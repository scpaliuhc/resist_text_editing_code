import tarfile
import torch
import matplotlib.pyplot as plt
import torchvision.io as tio
from src.models.model import Model
from torchvision.io.image import ImageReadMode
from util.path_list import get_weight
import torchvision.transforms as T
import pickle
from tqdm import tqdm
import numpy as np
import os
import torch.nn.functional as F
from src.models.layers.geometry.bbox import get_bbox,get_bb_level_features
from src.models.layers.geometry.shape import convert_shape
from src.dto.dto_model import TextInfo
from logzero import logger as logg
#######################loss#####################
def pert_loss(adv_img,img,mask=None,ord=1):
    if ord==1:
        if mask is None:
            loss=F.l1_loss(adv_img,img,reduction='mean')
        else:
            loss=F.l1_loss(adv_img[:,:,mask[1],mask[2]],img[:,:,mask[1],mask[2]])
    elif ord==2:
        if mask is None:
            loss=F.mse_loss(adv_img,img,reduction='mean')
        else:
            loss=F.mse_loss(adv_img[:,:,mask[1],mask[2]],img[:,:,mask[1],mask[2]])
    return loss
def inpaint_loss(out,target,mask):
    loss=F.l1_loss(out[:,:,mask[1],mask[2]],target[:,:,mask[1],mask[2]])
    return loss
def stroke_loss(stroke_param,GT,index=None):
    assert len(stroke_param.shape)==5
    if index is None:
        loss=F.cross_entropy(stroke_param[0,:,:,0,0],GT,reduction='mean')
    else:
        loss=F.cross_entropy(stroke_param[0,index,:,0,0],GT[index],reduction='mean')
    return -1*loss
def shadow_loss(sig,tanh,sig_GT,tanh_GT,index=None):
    assert len(sig.shape)==5
    assert len(tanh.shape)==5
    if index is None:
        loss=F.mse_loss(sig,sig_GT,reduction='mean')+F.mse_loss(tanh,tanh_GT,reduction='mean')
    else:
        loss=F.mse_loss(sig[0,index,:,:,:],sig_GT[0,index,:,:,:],reduction='mean')+F.mse_loss(tanh[0,index,:,:,:],tanh_GT[0,index,:,:,:],reduction='mean')
    return -1*loss
def visi_loss(visibility,index=None,T=True):
    #T True指定可见 False指定不可见
    vi_soft=torch.softmax(visibility[0,:,:,0,0],1)
    zero=torch.zeros_like(vi_soft[:,1])
    if index is None:
        if not T:
            loss=torch.max(vi_soft[:,1]-0.1,zero)
        else:
            loss=torch.max(0.1-vi_soft[:,1],zero)
    else:
        if not T:
            loss=torch.max(vi_soft[index,1]-0.1,zero[index])
        else:
            loss=torch.max(0.1-vi_soft[index,1],zero[index])
    return loss.mean()
def font_loss(fonts,GT,index=None):
    assert len(fonts.shape)==5
    if index is None:
        loss=F.cross_entropy(fonts[0,:,:,0,0],GT,reduction='mean')
    else:
        loss=F.cross_entropy(fonts[0,index,:,0,0],GT[index],reduction='mean')
    return -1*loss
def ocr_loss(pred_word_fg,mask):
    loss=torch.max(pred_word_fg[0,1,mask[1],mask[2]]-pred_word_fg[0,0,mask[1],mask[2]],torch.zeros_like(pred_word_fg[0,1,mask[1],mask[2]]))
    loss=loss.mean()
    return loss


def load_model(dev: torch.device):
    model = Model(dev).to(dev)
    model.load_state_dict(torch.load(get_weight()), strict=True)
    model.eval()
    return model

def get_parser_outs(img_norm,img_org,model):
    fe=model.backbone(img_norm)
    _,fe=model.down(fe)
    TxIn=model.text_parser(fe,img_org)
    # ocr_outs=model.text_parser.ocr(fe)
    return TxIn,fe

def get_parser_outs_with_fixed_ocr(img_norm,img_org,model,ocr_fixed):
    fe=model.backbone(img_norm)
    _,fe=model.down(fe)
    ocr_outs_ = model.text_parser.ocr(fe)
    ocr_outs=ocr_fixed
    bbox_information = get_bbox(
                ocr_outs, (img_org.shape[2], img_org.shape[3]))
    text_instance_mask = bbox_information.get_text_instance_mask()
    text_instance_mask = torch.from_numpy(
                text_instance_mask).to(model.text_parser.dev)
    features_box, text_num = get_bb_level_features(
            fe, text_instance_mask, model.training, 10, model.text_parser.dev
        )
    effect_visibility_outs = model.text_parser.effect_visibility(features_box)
    effect_param_outs = model.text_parser.effect_param(features_box)
    font_outs = model.text_parser.font(features_box)
        
        #font_size_outs = self.font_size(features_box)
    alpha_outs = model.text_parser.alpha(fe, img_org)

    batch_num = fe.shape[0]
    effect_visibility_outs = convert_shape(
            effect_visibility_outs, batch_num, text_num, False, 10
        )
    effect_param_outs = convert_shape(
            effect_param_outs, batch_num, text_num, False, 10)
    font_outs = convert_shape([font_outs], batch_num, text_num, False, 10)[0]
    TxIn=TextInfo(
            ocr_outs,#应该是替换后的还是替换前的？？
            bbox_information,
            effect_visibility_outs,
            effect_param_outs,
            font_outs,
            None,
            alpha_outs,
        )
    return TxIn,fe,ocr_outs_

def predict_with_fixed_ocr(img_norm,img_org,model,ocr_fixed):
    text_information,features,ocr=get_parser_outs_with_fixed_ocr(img_norm,img_org,model,ocr_fixed)
    inpaint = model.inpaintor(img_org, text_information)
    inpaint = F.interpolate(inpaint, img_org.shape[2:4], mode="bilinear")
    rec = model.reconstractor(features, img_org, inpaint, text_information)
    return text_information, inpaint, rec

def get_inpaint(img,TxInf,model):
    inpaint=model.inpaintor(img,TxInf)
    return inpaint

def load_img(imgfile: str):
    image = tio.read_image(imgfile, ImageReadMode.RGB)
    
    size = 640
    ratio = size / min(image.size()[1],image.size()[2])
    new_ysize = max(int(((ratio * image.size()[1]) // 128) * 128),128)
    new_xsize = max(int(((ratio * image.size()[2]) // 128) * 128),128)
    new_size = (new_ysize, new_xsize)
    pre_process = T.Compose(
        [
            T.Resize(new_size),
            T.ConvertImageDtype(torch.float),]
    )
    image=pre_process(image).unsqueeze(0)
    return image

def save_result(save_file,record):
    with open(save_file,'wb') as f:
        pickle.dump(record,f)

def get_mask_ind(textInf):
    bbox=textInf.bbox_information.get_text_rectangle()[0].astype(np.int)
    inds=[]
    for id_word in range(bbox.shape[0]):
        point_lt=bbox[id_word,0:2]
        point_rt=bbox[id_word,2:4]
        point_rb=bbox[id_word,4:6]
        point_lb=bbox[id_word,6:8]
        # points=[point_lt,point_rt,point_rb,point_lb]
        x1=min(point_lt[0],point_lb[0])
        x2=max(point_rt[0],point_rb[0])
        y1=min(point_lt[1],point_rt[1])
        y2=max(point_lb[1],point_rb[1])
        inds.append([y1,y2,x1,x2])
    return inds,bbox.shape[0]

def gradient_based_attack(model,img,mean,std,args,dev,save_dir,mask,log,GT_ocr,target_inpaint,GT_font,GT_stro,GT_shad_sig,GT_shad_tanh):
    if args.attack=='pgd':
        p=torch.Tensor(img.shape[0],img.shape[1],img.shape[2],img.shape[3]).uniform_(-args.epsilon,args.epsilon).to(dev)
        if args.GLI == 'glo':
            img_adv=img+p
        else:
            img_adv=img+p*mask[0][0]
        img_adv=torch.clamp(img_adv,0,1).detach_()
    else:
        img_adv=img.clone().detach()
    g_t=0
    iter=0
    alpha=args.epsilon/args.iter
    try:
    # if True:
        pert_l=0
        ocr_l=0
        inpaint_l=0
        visi_str_l=0
        visi_sha_l=0
        str_l=0
        sha_l=0
        font_l=0
        for i in tqdm(range(iter,args.iter)):
            img_adv.requires_grad=True
            img_adv_norm=(img_adv-mean)/std
            img_adv_org=(img_adv-0.5)*2
            ########待定
            # if 2 in args.attack_p or 3 in args.attack_p or 4 in args.attack_p or 6 in args.attack_p or 6 in args.attack_p:
            # ADV_TxT,_,ocr_outs_adv=get_parser_outs_with_fixed_ocr(img_adv_norm,img_adv_org,model,ocr_fixed=GT_ocr)
            # else:
            #     ADV_TxT,_=get_parser_outs(img_adv_norm,img_adv_org,model)
            ##########

            ###########
            if 1 in args.attack_p: #1和0不替换
                ADV_TxT,_=get_parser_outs(img_adv_norm,img_adv_org,model)
                ocr_outs_adv=ADV_TxT.ocr_outs
            else:
                ADV_TxT,_,ocr_outs_adv=get_parser_outs_with_fixed_ocr(img_adv_norm,img_adv_org,model,ocr_fixed=GT_ocr)
            ###########
            model.zero_grad()
            
            if args.GLI in ['loc','ind']: 
                pert_l=pert_loss(img_adv,img,mask[0],ord=2)
            else:
                pert_l=args.lbd_pert*pert_loss(img_adv,img,None,ord=2)
            
            if 0 in args.attack_p:
                ADV_word_out, _, _ = ocr_outs_adv # 之前是ADV_TxT.ocr_outs，但是现在ADV_TxT.ocr_outs是img的ocr结果了
                ADV_text_fg_pred, _, _ = ADV_word_out
                ocr_l=ocr_loss(ADV_text_fg_pred,mask[1])
            
            if 1 in args.attack_p:
                out=model.inpaintor((img_adv-0.5)*2,ADV_TxT) # 用的是fixed ocr！！！
                inpaint_l=inpaint_loss(out,target_inpaint,mask[2])

            if 2 in args.attack_p:#stro
                if args.GLI != 'ind':
                    visi_str_l=visi_loss(ADV_TxT.effect_visibility_outs[1],T=False,index=None)
                else:
                    visi_str_l=visi_loss(ADV_TxT.effect_visibility_outs[1],T=False,index=args.protect)
            
            if 3 in args.attack_p:#shad
                if args.GLI != 'ind':
                    visi_sha_l=visi_loss(ADV_TxT.effect_visibility_outs[0],T=False,index=None)
                else:
                    visi_sha_l=visi_loss(ADV_TxT.effect_visibility_outs[0],T=False,index=args.protect)

            if 4 in args.attack_p:#font
                if args.GLI != 'ind':
                    font_l=font_loss(ADV_TxT.font_outs,GT_font
                    ,index=None)
                else:
                    font_l=font_loss(ADV_TxT.font_outs,GT_font
                    ,index=args.protect)

            if 5 in args.attack_p:#stro
                if args.GLI != 'ind':
                    str_l=stroke_loss(ADV_TxT.effect_param_outs[2],GT_stro,index=None)
                else:
                    str_l=stroke_loss(ADV_TxT.effect_param_outs[2],GT_stro,index=args.protect)
            
            if 6 in args.attack_p:#shad
                if args.GLI != 'ind':
                    str_l=shadow_loss(ADV_TxT.effect_param_outs[0],ADV_TxT.effect_param_outs[1],GT_shad_sig,GT_shad_tanh,index=None)
                else:
                    str_l=shadow_loss(ADV_TxT.effect_param_outs[0],ADV_TxT.effect_param_outs[1],GT_shad_sig,GT_shad_tanh,index=args.protect)
            
            loss=args.lbd_ocr*ocr_l+args.lbd_inpaint*inpaint_l+args.lbd_visi_str*visi_str_l+args.lbd_visi_sha*visi_sha_l+args.lbd_str*str_l+args.lbd_sha*sha_l+args.lbd_font*font_l+args.lbd_pert*pert_l
            log.write(f'{i:4d}\t{loss:.4f}\t{pert_l:.4f}\t{ocr_l:.4f}\t{inpaint_l:.4f}\t{font_l:.4f}\t{visi_str_l:.4f}\t{visi_sha_l:.4f}\t{str_l:.4f}\t{sha_l:.4f}\n')
            log.flush()
            loss.backward()
            
            if args.attack != 'mi_fgsm':
                if args.GLI=='glo':
                    img_adv=img_adv-alpha*img_adv.grad.sign()
                else:
                    img_adv=img_adv-alpha*(img_adv.grad.sign()*mask[0][0])
            else:
                g_t=args.mu*g_t+img_adv.grad/torch.norm(img_adv.grad,p=1)
                if args.GLI=='glo':
                    img_adv=img_adv-alpha*g_t.sign()
                else:
                    img_adv=img_adv-alpha*(g_t.sign()*mask[0][0])
            perturb=torch.clamp(img_adv-img,-args.epsilon,args.epsilon)
            img_adv=torch.clamp(img+perturb,0,1).detach_()
    except Exception as e:
        logg.exception(e.args)
    finally:
        return img_adv