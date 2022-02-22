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
def get_parser_outs(img_norm,img,model):
    fe=model.backbone(img_norm)
    _,fe=model.down(fe)
    TxIn=model.text_parser(fe,img)
    # ocr_outs=model.text_parser.ocr(fe)
    return TxIn,fe

def get_parser_outs_with_fixed_ocr(img_norm,img,model,ocr):
    from src.models.layers.geometry.bbox import get_bbox,get_bb_level_features
    from src.models.layers.geometry.shape import convert_shape
    from src.dto.dto_model import TextInfo

    fe=model.backbone(img_norm)
    _,fe=model.down(fe)
    # TxIn=model.text_parser(fe,img)
    ocr_outs_1 = model.text_parser.ocr(fe)
    ocr_outs=ocr
    bbox_information = get_bbox(
                ocr_outs, (img.shape[2], img.shape[3]))
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
    alpha_outs = model.text_parser.alpha(fe, img)

    batch_num = fe.shape[0]
    effect_visibility_outs = convert_shape(
            effect_visibility_outs, batch_num, text_num, False, 10
        )
    effect_param_outs = convert_shape(
            effect_param_outs, batch_num, text_num, False, 10)
    font_outs = convert_shape([font_outs], batch_num, text_num, False, 10)[0]
    TxIn=TextInfo(
            ocr_outs_1,
            bbox_information,
            effect_visibility_outs,
            effect_param_outs,
            font_outs,
            None,
            alpha_outs,
        )
    return TxIn,fe

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
def calculate_ocr_loss(ADV,DIS,args,img,img_adv,mask_c,mask_c_): 
    ADV_word_out, _, _ = ADV.ocr_outs
    ADV_text_fg_pred, _, _ = ADV_word_out
    l1=torch.max(ADV_text_fg_pred[0,1,mask_c_[1],mask_c_[2]]-ADV_text_fg_pred[0,0,mask_c_[1],mask_c_[2]],torch.zeros_like(ADV_text_fg_pred[0,1,mask_c_[1],mask_c_[2]]))
    l1=l1.mean()
    # Char_L=torch.max(ADV_char_fg_pred[0,1,char_keep_rows,char_keep_cols]-ADV_char_fg_pred[0,0,char_keep_rows,char_keep_cols],torch.zeros_like(ADV_char_fg_pred[0,1,char_keep_rows,char_keep_cols]))
    # Char_L=Char_L.mean()
    # Char_L=0
    if args.GLI in ['loc','ind']:
        l2=DIS(img[:,:,mask_c[1],mask_c[2]],img_adv[:,:,mask_c[1],mask_c[2]])
    else:
        l2=DIS(img,img_adv)
    Loss=args.lbd1*l1+args.lbd2*l2
    return Loss,l1,l2

def calculate_inpaint_loss(ADV,DIS,args,img,img_adv,word_mask_c,word_mask_c_,word_mask_c_256,alpha_mask_c_,inpaintor,target_inpaint):   
    out=inpaintor(img_adv,ADV)
 
    #test alpha mask
    # l1=-1*DIS(out[:,:,alpha_mask_c_[1],alpha_mask_c_[2]],gt_inpaint[:,:,alpha_mask_c_[1],alpha_mask_c_[2]])
    l1=1*DIS(out[:,:,word_mask_c_256[1],word_mask_c_256[2]],target_inpaint[:,:,word_mask_c_256[1],word_mask_c_256[2]])
    if args.GLI in ['loc','ind']:
        l2=DIS(img[:,:,word_mask_c[1],word_mask_c[2]],img_adv[:,:,word_mask_c[1],word_mask_c[2]])
    else:
        l2=DIS(img,img_adv)
    Loss=args.lbd1*l1+args.lbd2*l2
    return Loss, l1, l2 

def calculate_visible_loss(ADV,DIS,args,img,img_adv,mask_c,target_vis,target_ocr):
    shadow_visibility, stroke_visibility=ADV.effect_visibility_outs
    # ocr_outs=ADV.ocr_outs
    # l3=DIS(ocr_outs[0][0],target_ocr[0])+DIS(ocr_outs[1][0],target_ocr[1])+DIS(ocr_outs[2],target_ocr[2])#word
    # l3+=#char
    # l3+=#rec

    if 3 in args.attack_p:
        visibility = F.softmax(shadow_visibility[0], 1)
    elif 2 in args.attack_p:
        visibility = F.softmax(stroke_visibility[0], 1)
    else:
        raise NotImplementedError
    l1=DIS(visibility,target_vis)
    if args.GLI in ['loc','ind']:
        l2=DIS(img[:,:,mask_c[1],mask_c[2]],img_adv[:,:,mask_c[1],mask_c[2]])
    else:
        l2=DIS(img,img_adv)
    Loss=args.lbd1*l1+args.lbd2*l2#+args.lbd3*l3
    return Loss,l1,l2#,l3


def save_fig(save_dir,name,img,img_adv,inpaint):
    fig=plt.figure(figsize=(30, 30))
    plt.subplot(1, 3, 1)
    plt.imshow(img.cpu().detach().numpy()[0].transpose((1,2,0)))
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(img_adv.cpu().detach().numpy()[0].transpose((1,2,0)))
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(inpaint.cpu().detach().numpy()[0].transpose((1,2,0))/2+0.5)
    plt.axis("off")
    plt.savefig(f"{save_dir}/{name}.png")
    plt.close()
def str2bool(str):
    return True if str.lower()=='true' else False
def cw(model,img,mean,std,DIS,args,dev,save_dir,word_mask_c,word_mask_c_,word_mask_c_256,alpha_mask_c,alpha_mask_c_,target_inpaint,log):
    if args.pre:
        try:
            with open(os.path.join(save_dir,f'adv_per_iter_{args.GLI}.b'),'rb') as f:
                r=pickle.load(f)
                w=r[3].to(dev)
                iter=r[2]
            print(f'load the last results from adv_per_iter_{args.GLI}.b')
        except:
            print(f'fail at loading adv_per_iter_{args.GLI}.b')
            w=torch.atanh(torch.clamp(2*img-1,-0.999999,0.999999)).to(dev).detach_()
            iter=0
    else:
        w=torch.atanh(torch.clamp(2*img-1,-0.999999,0.999999)).to(dev).detach_()
        iter=0
    w.requires_grad=True
    # optimizer = torch.optim.Adam([w], lr=args.lr)
    w_fixed=w.clone().detach()
    
    try:
        for i in tqdm(range(iter,args.iter)):
            img_adv=1/2*(torch.tanh(w)+1)
            # if args.GLI=='glo':
            #     img_adv=(torch.tanh(w)+1)/2
            # elif args.GLI=='loc':
            #     img_adv=(torch.tanh(w)+1)/2*word_bin_mask+img*(1-word_bin_mask)
            img_adv_norm=(img_adv-mean)/std
            ADV,fe=get_parser_outs(img_adv_norm,img_adv,model)
            if 0 in args.attack_p:
                Loss,l1,l2=calculate_ocr_loss(ADV,DIS,args,img,img_adv,word_mask_c,word_mask_c_)
            if 1 in args.attack_p:
                raise NotImplementedError
                # Loss,l1,l2=calculate_inpaint_loss(ADV,DIS,args,img,img_adv,word_mask_c,word_mask_c_,word_mask_c_256,alpha_mask_c_,model.inpaintor,gt_inpaint)
            log.write(f'{i:4d}\t{Loss:.4f}\t{l1:.8f}\t{l2:.8f}\n')
            log.flush()
            if i%20==0:
                save_fig(save_dir,i,img,img_adv)
            # optimizer.zero_grad()
            Loss.backward()
            print(w.grad[:,:,alpha_mask_c[1],alpha_mask_c[2]])
            w=(w-args.lr*w.grad).detach_()
            # optimizer.step()
            if args.GLI!='glo':
                w=(w_fixed*(1-word_mask_c[0])+w*word_mask_c[0]).detach_()
            w.requires_grad=True         
    except Exception as e:
        print(e.args)
        print(w.grad)
    finally:
        return img_adv, w, i
def gradient_attack(model,img,mean,std,DIS,args,dev,save_dir,word_mask_c,word_mask_c_,word_mask_c_256,alpha_mask_c,alpha_mask_c_,target_inpaint,target_visible,target_ocr,log):
    if args.pre:
        try:
            with open(os.path.join(save_dir,f'adv_per_iter_{args.GLI}.b'),'rb') as f:
                r=pickle.load(f)
                img_adv=r[0].to(dev)
                iter=r[2]
                g_t=r[3]
            print(f'load the last results from adv_per_iter_{args.GLI}.b')
        except:
            print(f'fail at loading adv_per_iter_{args.GLI}.b')
            if args.attack=='pgd':
                p=torch.Tensor(img.shape[0],img.shape[1],img.shape[2],img.shape[3]).uniform_(-args.epsilon,args.epsilon).to(dev)
                if args.GLI == 'glo':
                    img_adv=img+p
                else:
                    img_adv=img+p*word_mask_c[0]
                img_adv=torch.clamp(img_adv,0,1).detach_()
            else:
                img_adv=img.clone().detach()
            g_t=0
            iter=0         
    else:
        if args.attack=='pgd':
            p=torch.Tensor(img.shape[0],img.shape[1],img.shape[2],img.shape[3]).uniform_(-args.epsilon,args.epsilon).to(dev)
            if args.GLI == 'glo':
                img_adv=img+p
            else:
                img_adv=img+p*word_mask_c[0]
            img_adv=torch.clamp(img_adv,0,1).detach_()
        else:
            img_adv=img.clone().detach()
        g_t=0
        iter=0
    alpha=args.epsilon/args.iter
    l3=0.
    try:
    # if True:
        for i in tqdm(range(iter,args.iter)):
            img_adv.requires_grad=True
            img_adv_norm=(img_adv-mean)/std
            if 2 in args.attack_p or 3 in args.attack_p:
                ADV,fe=get_parser_outs_with_fixed_ocr(img_adv_norm,img_adv,model,ocr=target_ocr)
            else:
                ADV,fe=get_parser_outs(img_adv_norm,img_adv,model)
            model.zero_grad()
            if 0 in args.attack_p:
                Loss,l1,l2=calculate_ocr_loss(ADV,DIS,args,img,img_adv,word_mask_c,word_mask_c_)
            if 1 in args.attack_p:
                Loss,l1,l2=calculate_inpaint_loss(ADV,DIS,args,img,img_adv,word_mask_c,word_mask_c_,word_mask_c_256,alpha_mask_c_,model.inpaintor,target_inpaint)
            if 2 in args.attack_p or 3 in args.attack_p:
                Loss,l1,l2=calculate_visible_loss(ADV,DIS,args,img,img_adv,word_mask_c,target_visible,target_ocr)
            log.write(f'{i:4d}\t{Loss:.4f}\t{l1:.4f}\t{l2:.4f}\t{l3:.4f}\n')
            log.flush()
            Loss.backward()
            if args.attack != 'mi_fgsm':
                if args.GLI=='glo':
                    img_adv=img_adv-alpha*img_adv.grad.sign()
                else:
                    img_adv=img_adv-alpha*(img_adv.grad.sign()*word_mask_c[0])
            else:
                g_t=args.mu*g_t+img_adv.grad/torch.norm(img_adv.grad,p=1)
                if args.GLI=='glo':
                    img_adv=img_adv-alpha*g_t.sign()
                else:
                    img_adv=img_adv-alpha*(g_t.sign()*word_mask_c[0])
            perturb=torch.clamp(img_adv-img,-args.epsilon,args.epsilon)
            img_adv=torch.clamp(img+perturb,0,1).detach_()
    except Exception as e:
        print("!!!!")
        print(e.args)
    finally:
        img_adv_norm=(img_adv-mean)/std
        ADV,fe=get_parser_outs(img_adv_norm,img_adv,model)
        return img_adv,g_t,i,model.inpaintor(img_adv,ADV)
def save_result(save_file,record):
    with open(save_file,'wb') as f:
        pickle.dump(record,f)
def str2list_int(s):
    #'2,3,4,5'->[2,3,4,5]
    numbers=s.split(',')
    for i in len(numbers):
        numbers[i]=int(numbers[i])
    return numbers
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
            if 2 in args.attack_p or 3 in args.attack_p or 4 in args.attack_p or 6 in args.attack_p or 6 in args.attack_p:
                ADV_TxT,fe=get_parser_outs_with_fixed_ocr(img_adv_norm,(img_adv-0.5)*2,model,ocr=GT_ocr)
            else:
                ADV_TxT,fe=get_parser_outs(img_adv_norm,(img_adv-0.5)*2,model)
            model.zero_grad()
            
            if args.GLI in ['loc','ind']: 
                pert_l=pert_loss(img_adv,img,mask[0],ord=2)
            else:
                pert_l=args.lbd_pert*pert_loss(img_adv,img,None,ord=2)
            
            if 0 in args.attack_p:
                ADV_word_out, _, _ = ADV_TxT.ocr_outs
                ADV_text_fg_pred, _, _ = ADV_word_out
                ocr_l=ocr_loss(ADV_text_fg_pred,mask[1])
            
            if 1 in args.attack_p:
                out=model.inpaintor((img_adv-0.5)*2,ADV_TxT)
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
                    font_loss(ADV_TxT.font_outs,GT_font
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
        print('attack_box_util.py gradient_based_attack\n')
        print(e.args)
    finally:
        # img_adv_norm=(img_adv-mean)/std
        # if 2 in args.attack_p or 3 in args.attack_p or 4 in args.attack_p or 6 in args.attack_p or 6 in args.attack_p:
        #     ADV_TxT,fe=get_parser_outs_with_fixed_ocr(img_adv_norm,(img_adv-0.5)*2,model,ocr=GT_ocr)
        # else:
        #     ADV_TxT,fe=get_parser_outs(img_adv_norm,(img_adv-0.5)*2,model)
        # return img_adv,ADV_TxT,i,model.inpaintor((img_adv-0.5)*2,ADV_TxT)
        return img_adv