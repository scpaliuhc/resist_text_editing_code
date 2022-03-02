import pickle
import torch.nn.functional as F
import numpy as np
from logzero import logger as logg
from attack_box_util import *
from PIL import Image
from src.modules.postprocess.vector import  vectorize_postref
from src.modules.postprocess.renderer import render_vd
from piq import ssim, psnr
#################攻击ocr####################
#IOU:攻击前和攻击后text instance mask的重叠
############################################
def load_record(file):
    with open(file,'rb') as f:
        record=pickle.load(f)
    if len(record)==4:
        img_adv,vd_adv,vd,protect=record
    else:
        img_adv,vd_adv,vd=record
        protect=None
    return img_adv,vd_adv,vd,protect
def ssim_psnr_L1(img1,img2):
    ssim=ssim(img1,img2, data_range=1.).item()
    psnr=psnr(img1,img2, data_range=1.).item()
    L1=F.l1_loss(img1,img2,reduction='mean').item()
    return ssim,psnr,L1
def draw_box(img,id,vd):
    x1,y1,x2,y2=vd.tb_param[id].box
    img=img.clone()
    img[y1,x1:x2,:]=1
    img[y2,x1:x2,:]=1
    img[y1:y2,x1,:]=1
    img[y1:y2,x2,:]=1
    return img  

def mask_occup(shape,num_text_instance,vd,num_text_instance_adv,vd_adv,index):
    mask1=np.zeros((shape[0],shape[1]),dtype=np.int8)
    mask2=np.zeros((shape[0],shape[1]),dtype=np.int8)
  
    for i in index:
        x1,y1,x2,y2=vd.tb_param[i].box
        mask1[y1:y2,x1:x2]=1
    for i in range(num_text_instance_adv):
        x1,y1,x2,y2=vd_adv.tb_param[i].box
        mask2[y1:y2,x1:x2]=1


    mask=np.logical_and(mask1,mask2).astype(np.int8)
    area_=mask.sum()
    area1=mask1.sum()
    area2=mask2.sum()
    return [mask1,mask2,mask], area_/area1

def distance_inpaint(inpaint1,inpaint2,mask):
    loss=F.mse_loss(inpaint1*mask,inpaint2*mask,reduction='sum')
    return loss/mask.sum()

def pars_with_fixed_text_pred():
    None

def pred_with_fixed_text_pred():
    None

def check_0(args,img_adv,vd,vd_adv,protect,):
        num_text_instance=len(vd.get_texts())
        num_text_instance_adv=len(vd_adv.get_texts())
            
        if protect is None:
            index=[i for i in range(num_text_instance)]
            logg.debug('protect is None')
        else:
            index=protect
            print(protect)
        if num_text_instance_adv==0:
            IOU=0.
        else:
                #mask_IOU
            re=mask_occup([img_adv.shape[2],img_adv.shape[3]],num_text_instance,vd,num_text_instance_adv,vd_adv,index)
        return re

def Cal_IOU(coor1,coor2):

    x1,y1,x2,y2=coor1
    x1_adv,y1_adv,x2_adv,y2_adv=coor2
    # min_y1=min(y1,y1_adv)
    max_y2=min(y2,y2_adv)
    # min_x1=min(x1,x1_adv)
    max_x2=max(x2,x2_adv)
    mask1=np.zeros((max_y2,max_x2),dtype=np.int8)
    mask2=np.zeros_like(mask1)

    mask1[y1:y2,x1:x2]=1
    mask2[y1_adv:y2_adv,x1_adv:x2_adv]=1
    mask=np.logical_and(mask1,mask2).astype(np.int8)

    L=mask.sum()
    K1=mask1.sum()
    K2=mask2.sum()
    IOU=L/(K1+K2-L)
    return IOU




def check_1(img_adv,vd,vd_adv):
    texts=vd.get_texts()
    texts_adv=vd_adv.get_texts()
    txt_content=0
    stroke=0
    font=0
    shadow_visibility_flag=[]#
    stroke_visibility_flag=[]
    index=[]
    for i in range(len(texts)):
        x1,y1,x2,y2=vd.tb_param[i].box
        for j in range(len(texts_adv)):
            x1_adv,y1_adv,x2_adv,y2_adv=vd_adv.tb_param[j].box
            iou=Cal_IOU((x1,y1,x2,y2),(x1_adv,y1_adv,x2_adv,y2_adv))
            if iou>0.9:#是同一个字
                index.append((i,j))
                if texts[i]==texts_adv[j]:#文字内容是否识别准确，但这个并不重要，内容可修改
                    txt_content+=1
                if vd.effect_visibility[i].shadow_visibility_flag==vd_adv.effect_visibility[j].shadow_visibility_flag:
                    shadow_visibility_flag.append(vd.effect_visibility[i].shadow_visibility_flag)
                if vd.effect_visibility[i].stroke_visibility_flag==vd_adv.effect_visibility[j].stroke_visibility_flag:
                    stroke_visibility_flag.append(vd.effect_visibility[i].stroke_visibility_flag)
                if vd.effect_param[i].stroke_param.border_weight==vd_adv.effect_param[j].stroke_param.border_weight:
                    stroke+=1
                if vd.tb_param[i].font_data.font_id==vd_adv.tb_param[j].font_data.font_id:
                    font+=1
                break
    return txt_content,stroke,shadow_visibility_flag,stroke_visibility_flag,font,index

def min_max_mean(lis):
    min_=1000
    max_=0
    num=0
    sum_=0
    for ele in lis:
        if ele is not None:
            num+=1
            sum_+=ele
            if ele<min_:
                min_=ele
            if ele>max_:
                max_=ele
    mean=sum_/num
    return min_,max_,mean,num    

def form_print(rows,value):
    s=',min,max,mean,num\n'
    for i in range(len(rows)):
        s=s+f'{rows[i]},{value[i][0]:.4f},{value[i][1]:.4f},{value[i][2]:.4f},{value[i][3]:.4f}\n'
    return s


