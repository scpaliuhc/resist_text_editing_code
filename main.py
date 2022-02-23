
from ast import Gt, arg
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import torch.nn.functional as F
from attack_box_util import *
import datetime
import shutil
from src.modules.postprocess.vector import  vectorize_postref
from PIL import Image
from src.modules.postprocess.renderer import render_vd
from logzero import logger as logg
parser = argparse.ArgumentParser()
parser.add_argument('-d','--data_dir',default='dataset')
parser.add_argument('-g','--gpuid',default=0,type=int)
parser.add_argument('-a','--attack',default='bim',choices=['pgd','bim','fgsm','mi_fgsm'])
parser.add_argument('-i','--iter',default=40,type=int)
parser.add_argument('--iter2',default=150,type=int)
parser.add_argument('-e','--epsilon',default=0.2,type=float)
parser.add_argument('--lbd_ocr',default=1,type=float)
parser.add_argument('--lbd_inpaint',default=1,type=float)
parser.add_argument('--lbd_visi_str',default=1,type=float)
parser.add_argument('--lbd_visi_sha',default=1,type=float)
parser.add_argument('--lbd_str',default=1,type=float)
parser.add_argument('--lbd_sha',default=1,type=float)
parser.add_argument('--lbd_font',default=1,type=float)
parser.add_argument('--lbd_pert',default=1,type=float)
parser.add_argument('--T_visi',default=False,type=bool)
parser.add_argument('--mu',default=0.8,type=float)
# parser.add_argument('-r','--lr',default=0.005,type=float)
parser.add_argument('-L','--log',default=True,type=bool)
parser.add_argument('-s','--reset',default=False,type=bool)
parser.add_argument('--GLI',default='glo',type=str,choices=['loc','glo','ind'])
parser.add_argument('--protect',type=int,nargs='+')
parser.add_argument('--attack_p',type=int,nargs='+') #0:detection #1:inpaint #2:stroke visi #3:shadow visi #4:font #5:stroke #6:shadow
args=parser.parse_args()
print(args)
if args.protect is not None:
    args.protect=np.array(args.protect,dtype=np.int8)
if args.GLI=='ind':
    assert len(args.protect)>0
############预处理###############
if args.gpuid<0:
    dev = torch.device(f"cpu")
else:
    dev = torch.device(f"cuda:{args.gpuid}")
print(dev)
files=os.listdir(args.data_dir)
total=len(files)
mean_=torch.tensor([0.485, 0.456, 0.406],dtype=torch.float32).reshape((1,3,1,1)).to(dev)
std_=torch.tensor([0.229, 0.224, 0.225],dtype=torch.float32).reshape((1,3,1,1)).to(dev)
model = load_model(dev)
save_dir=f'result/{args.GLI}_{args.attack_p}'
if args.reset:
    try:
        shutil.rmtree(save_dir)
    except:
        None
try:
    os.makedirs(save_dir)
except:
    None
if args.log:
    log=open(os.path.join(save_dir,'log.txt'),'a')
    t=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log.write('***'+t+'***\n')
    log.write(str(args)+'\n')
    log.flush()
finished=os.listdir(save_dir)
for id,file in enumerate(files):
    if f'{args.attack}_{file[:-4]}.b' in finished:
        continue
    try:
        logg.debug(f"{id}/{total} {file}")
        
        img=load_img(os.path.join(args.data_dir,file))
        img=img.to(dev)
        img_norm=(img-mean_)/std_
        img_org=(img-0.5)*2
        img_np=(img[0].data.cpu().permute(1, 2, 0).numpy()*255).astype(np.uint8)
        pil_img=Image.fromarray(img_np)
        img_size = torch.tensor([pil_img.size[1], pil_img.size[0]]).unsqueeze(0)
        inps = (img_norm, None, img_size)
        # with torch.no_grad():
        #     GT,fe=get_parser_outs(img_clean_norm,img_org,model)
        with torch.no_grad():
            outs = model(img_norm, img_org)
        vd, rec_img, op = vectorize_postref(
                    pil_img, inps, outs, model.reconstractor, args.iter2, dev=dev
                )
        rec_img = torch.max(
                            torch.min(
                                rec_img,
                                torch.zeros_like(rec_img) +
                                255),
                            torch.zeros_like(rec_img))
        rec_img=rec_img.data.cpu().numpy()[0].transpose(1, 2, 0)/255
        #用优化后结果作为GT
        GT=outs[0]
        GT_ocr=GT.ocr_outs
        GT_fonts=torch.argmax(torch.softmax(torch.tensor(op.font_outs[0,:,:,0,0],dtype=torch.float32).to(dev),1),1)
        GT_stroke=torch.argmax(torch.softmax(torch.tensor(op.stroke_param_outs[0,:,:,0,0],dtype=torch.float32).to(dev),1),1)
        GT_shadow_sig=op.shadow_param_sig_outs
        GT_shadow_tanh=op.shadow_param_tanh_outs
        logg.debug(vd.get_texts())
        logg.debug(vd.get_font_names())
        text_mask_clean=outs[0].bbox_information.get_text_instance_mask()[0]
        back_clean=vd.bg.astype(np.uint8)
        output_img = render_vd(vd)

        # 用一次ocr的结果作为GT
        # shadow_visibility, stroke_visibility = GT.effect_visibility_outs
        # shadow_param_sig, shadow_param_tanh, stroke_param = GT.effect_param_outs
        # fonts=GT.font_outs
        # font_GT=F.softmax(fonts[0], 1)
        # font_GT=font_GT.view(font_GT.shape[0],font_GT.shape[1])
        # font_GT=torch.argmax(font_GT.detach().clone(),1)
        # stroke_GT=F.softmax(stroke_param[0],1)
        # stroke_GT=stroke_GT.view(stroke_GT.shape[0],stroke_GT.shape[1])
        # stroke_GT=torch.argmax(stroke_GT.detach().clone(),1)
        # shadow_sig_GT=shadow_param_sig.detach().clone()
        # shadow_tanh_GT=shadow_param_tanh.detach().clone()
        
        GT_word_out, _, _ = GT.ocr_outs
        GT_text_fg_pred, _, _ = GT_word_out
        GT_text_fg_pred_ = F.softmax(GT_text_fg_pred.clone().detach(), 1)
        GT_text_fg_pred_np = GT_text_fg_pred_.data.cpu().numpy()[0,1]
        word_keep_rows_, word_keep_cols_ = np.where(GT_text_fg_pred_np > 0.8)
        word_mask_=torch.zeros((1,1,GT_text_fg_pred.shape[2],GT_text_fg_pred.shape[3]),dtype=torch.float32).to(dev)
        word_mask_[:,:,word_keep_rows_,word_keep_cols_]=1.
        word_mask=F.interpolate(word_mask_,scale_factor=img.shape[2]/GT_text_fg_pred.shape[2])
        word_mask_256=F.interpolate(word_mask_,(256,256))
        word_keep_rows, word_keep_cols = np.where(word_mask.data.cpu().numpy()[0,0] ==1.0)
        word_keep_rows_256, word_keep_cols_256 = np.where(word_mask_256.data.cpu().numpy()[0,0] ==1.0)
        if args.GLI=='ind':
            inds,word_num=get_mask_ind(GT)
            pro_mask=torch.zeros((1,1,img.shape[2],img.shape[3]),dtype=torch.float32).to(dev)# image.shape
            for i in args.protect:
                y1,y2,x1,x2=inds[i]
                pro_mask[:,:,y1:y2,x1:x2]=1.
                pro_mask_=F.interpolate(pro_mask,scale_factor=GT_text_fg_pred.shape[2]/img.shape[2])# pre.shape
                pro_mask_256=F.interpolate(pro_mask,(256,256))# 256
                word_mask_=word_mask_*pro_mask_
                word_mask=word_mask*pro_mask
                word_mask_256=word_mask_256*pro_mask_256
                word_keep_rows, word_keep_cols = np.where(word_mask.data.cpu().numpy()[0,0] ==1.0)
                word_keep_rows_, word_keep_cols_ = np.where(word_mask_.data.cpu().numpy()[0,0] ==1.0)
                word_keep_rows_256, word_keep_cols_256 = np.where(word_mask_256.data.cpu().numpy()[0,0] ==1.0)
        word_mask_c_=[word_mask_,word_keep_rows_,word_keep_cols_]
        word_mask_c=[word_mask,word_keep_rows,word_keep_cols]
        word_mask_c_256=[word_mask_256,word_keep_rows_256, word_keep_cols_256]
        mask=[word_mask_c,word_mask_c_,word_mask_c_256]
        target_inpaint=(F.interpolate(img.clone().detach(),(256,256)).detach_()-0.5)*2
        save_file=os.path.join(save_dir,f'{args.attack}_{file[:-4]}.b')
        
        img=img.detach().clone()
        log.write(f"{id}/{total} {file}\n")
        img_adv=gradient_based_attack(model,img,mean_,std_,args,dev,save_dir,mask,log,GT_ocr,target_inpaint,GT_fonts,GT_stroke,GT_shadow_sig,GT_shadow_tanh)
        
        #font_GT,stroke_GT,shadow_sig_GT,shadow_tanh_GT)
    except Exception as e:
        print(e.args)
        log.write(f'{e.args}\n')
        log.flush()
    finally:
        logg.debug('derendering for adv')
        
        img_adv_norm=(img_adv-mean_)/std_
        img_adv_orig=(img_adv-0.5)*2
        img_adv_np=(img_adv[0].data.cpu().permute(1, 2, 0).numpy()*255).astype(np.uint8)
        pil_img_adv = Image.fromarray(img_adv_np)
        inps = (img_adv_norm, None, img_size)
        img_adv=img_adv.data.cpu().numpy()[0].transpose(1,2,0)    
        if 0 in args.attack_p or 1 in args.attack_p:
            with torch.no_grad():
                outs_adv = model(img_adv_norm, img_adv_orig)
        else:
            logg.debug('predict_with_fixed_ocr')
            with torch.no_grad():
                outs_adv = predict_with_fixed_ocr(img_adv_norm, img_adv_orig, model,GT_ocr)
        if outs_adv[0].font_outs.shape[1]==0 and 0 in args.attack_p or 1 in args.attack_p:
            logg.error(f"{outs_adv[0].font_outs.shape}")
            log.write(f'[error]-adv: {outs_adv[0].font_outs.shape}\n')
            output_img_adv=np.zeros_like(img_adv)
            back_adv=output_img_adv
            vd_adv=None
        else:
            vd_adv, rec_img_adv, op_adv = vectorize_postref(
                    pil_img_adv, inps, outs_adv, model.reconstractor, args.iter2, dev=dev
                )
            rec_img_adv = torch.max(
                        torch.min(
                            rec_img_adv,
                            torch.zeros_like(rec_img_adv) +
                            255),
                        torch.zeros_like(rec_img_adv))
            rec_img_adv=rec_img_adv.data.cpu().numpy()[0].transpose(1, 2, 0)/255
            output_img_adv = render_vd(vd_adv)
            back_adv=vd_adv.bg.astype(np.uint8)
            logg.debug(vd_adv.get_texts())
            logg.debug(vd_adv.get_font_names())
        text_mask_adv=outs_adv[0].bbox_information.get_text_instance_mask()[0]
        
        
        img=img.data.cpu().numpy()[0].transpose(1,2,0)
        log.flush()
        fig=plt.figure(figsize=(65, 30))
        plt.subplot(2, 5, 1)
        plt.imshow(img)
        plt.axis("off")
        plt.subplot(2, 5, 2)
        plt.imshow(text_mask_clean)
        plt.axis("off")
        plt.subplot(2, 5, 3)
        plt.imshow(back_clean)
        plt.axis("off")
        plt.subplot(2, 5, 4)
        plt.imshow(output_img)
        plt.axis("off")
        plt.subplot(2, 5, 5)
        plt.imshow(rec_img)
        plt.axis("off")
        plt.subplot(2, 5, 6)
        plt.imshow(img_adv)
        plt.axis("off")
        plt.subplot(2, 5, 7)
        plt.imshow(text_mask_adv)
        plt.axis("off")
        plt.subplot(2, 5, 8)
        plt.imshow(back_adv)
        plt.axis("off")
        plt.subplot(2, 5, 9)
        plt.imshow(output_img_adv)
        plt.axis("off")
        plt.subplot(2, 5, 10)
        plt.imshow(rec_img_adv)
        plt.axis("off")
        save_result(save_file,[img_adv,vd_adv,vd])
        plt.savefig(os.path.join(save_dir, f'{args.attack}_{file[:-4]}.jpg'))
        plt.close()
        
        
log.close()
