from cv2 import log
from torch import no_grad
from vis_sta_box import *
import argparse
import os
from logzero import logger as logg
from attack_box_util import *
from PIL import Image
from src.modules.postprocess.vector import  vectorize_postref
from src.modules.postprocess.renderer import render_vd
from piq import ssim, psnr

parser = argparse.ArgumentParser()
parser.add_argument('--pics',default='dataset')
parser.add_argument('--GLI',default='glo',type=str,choices=['loc','glo','ind'])
parser.add_argument('--attack',default='bim')
parser.add_argument('--attack_dir',default='result')
parser.add_argument('--check_dir',default='check')
parser.add_argument('--attack_p',type=int,nargs='+',required=True)#0 ocr
parser.add_argument('-g','--gpuid',default=0,type=int)
parser.add_argument('--T_visi',default=False,type=bool)

def generate(args):
    res=f'{args.GLI}_{args.attack_p}_visi{1 if args.T_visi else 0}'
    if args.gpuid<0:
        dev = torch.device(f"cpu")
    else:
        dev = torch.device(f"cuda:{args.gpuid}")
    save_dir=os.path.join(args.check_dir,res)
    try:
        os.makedirs(save_dir)
    except:
        logg.debug(f'{save_dir} has been existed')
    mean_=torch.tensor([0.485, 0.456, 0.406],dtype=torch.float32).reshape((1,3,1,1)).to(dev)
    std_=torch.tensor([0.229, 0.224, 0.225],dtype=torch.float32).reshape((1,3,1,1)).to(dev)
    model = load_model(dev)

    files=os.listdir(args.pics)
    files.sort()
    num=len(files)


    

    for id,file in enumerate(files):
        logg.debug(f'{id}/{num} {file[:-4]}')
        vs={}
        
        img = load_img(os.path.join(args.pics,file))
        try:
            record=load_record(os.path.join(args.attack_dir,res,f'{args.attack}_{file[:-4]}.b'))
            # print(len(record),type(record[0]),type(record[1]),type(record[2]))
            # exit()
            if len(record)==4:
                img_adv,vd_adv,vd,protect=record
            else:
                img_adv,vd_adv,vd=record
                protect=None
        except Exception as e:
            # img_adv,vd_adv,vd,protect
            print(e.args)
            print(os.path.join(args.attack_dir,res,f'{args.attack}_{file[:-4]}.b'))
            exit()
        img=img.to(dev)   
        img_adv=torch.tensor(img_adv.transpose((2,0,1)),dtype=torch.float32).unsqueeze(0)
        img_adv=img_adv.to(dev)
        ssim_i=ssim(img_adv, img, data_range=1.).item()
        psnr_i=psnr(img_adv, img, data_range=1.).item()
        L1_i=F.l1_loss(img_adv, img,reduction='mean').item()
        if vd_adv is not None:
            output_img = render_vd(vd)/255
            output_img_adv = render_vd(vd_adv)/255
            back_img = vd.bg/255
            back_img_adv = vd_adv.bg/255
            with torch.no_grad():
                output_img=torch.tensor(output_img).permute(2,0,1).unsqueeze(0).to(dev)
                output_img_adv=torch.tensor(output_img_adv).permute(2,0,1).unsqueeze(0).to(dev)
                back_img=torch.tensor(back_img).permute(2,0,1).unsqueeze(0)
                back_img_adv=torch.tensor(back_img_adv).permute(2,0,1).unsqueeze(0)
                ssim_o=ssim(output_img, img, data_range=1.).item()
                psnr_o=psnr(output_img, img, data_range=1.).item()
                L1_o=F.l1_loss(output_img, img,reduction='mean').item()
                ssim_oa=ssim(output_img_adv, img, data_range=1.).item()
                psnr_oa=psnr(output_img_adv, img, data_range=1.).item()
                L1_oa=F.l1_loss(output_img_adv, img,reduction='mean').item()
                ssim_b=ssim(back_img,back_img_adv, data_range=1.).item()
                psnr_b=psnr(back_img,back_img_adv, data_range=1.).item()
                L1_b=F.l1_loss(back_img,back_img_adv,reduction='mean').item()
        else:
            ssim_o=None
            psnr_o=None
            L1_o=None
            ssim_oa=None
            psnr_oa=None
            L1_oa=None
            ssim_b=None
            psnr_b=None
            L1_b=None
        

        if 0 in args.attack_p:
            if vd_adv is None:
                occup=0.
            else:     
                re=check_0(args,img_adv,vd,vd_adv,protect)
                occup=re[1]
            vs[file[:-4]]={'occup':occup,
                            'ssim_i':ssim_i,'psnr_i':psnr_i,'L1_i':L1_i,
                            'ssim_o':ssim_o,'psnr_o':psnr_o,'L1_i':L1_o,
                            'ssim_oa':ssim_oa,'psnr_oa':psnr_oa,'L1_oa':L1_oa,
                            'ssim_b':ssim_b,'psnr_b':psnr_b,'L1_b':L1_b,}
        
        else:            
            re=check_1(img_adv,vd,vd_adv)
            vs[file[:-4]]={'ssim_i':ssim_i,'psnr_i':psnr_i,'L1_i':L1_i,
                            'ssim_o':ssim_o,'psnr_o':psnr_o,'L1_i':L1_o,
                            'ssim_oa':ssim_oa,'psnr_oa':psnr_oa,'L1_oa':L1_oa,
                            'ssim_b':ssim_b,'psnr_b':psnr_b,'L1_b':L1_b,
                            'text_content':re[0],'stroke':re[1],
                            'shadow_visibility_flag':re[2],'stroke_visibility_flag':re[3],
                            'font':re[4],'iou_index':re[5]}
            


    save_file=os.path.join(save_dir,f'{args.attack_p[0]}_check.b')
    save_result(save_file,vs)

def static(args):
    res=f'{args.GLI}_{args.process}_visi{1 if args.T_visi else 0}'
    pick_dir1=os.path.join(args.check_dir,res)
    pick_dir2=os.path.join(args.attack_dir,res)
    check_file = f'{args.attack_p[0]}_check.b'
    files = os.listdir(args.pics)
    files.sort()
    with open(os.path.join(pick_dir1,check_file),'rb') as f:
        vs=pickle.load(f)
    text_num=[]
    ssim_i=[]
    psnr_i=[]
    L1_i=[]
    ssim_o=[]
    psnr_o=[]
    L1_o=[]
    ssim_oa=[]
    psnr_oa=[]
    L1_oa=[]
    ssim_b=[]
    psnr_b=[]
    L1_b=[]
    occup=[]
    text_content=[]
    stroke=[]
    shadow_visibility_flag=[]
    stroke_visibility_flag=[]
    font=[]
    iou_index=[]
    for file in files:
        file = file[:-4]
        try:
            img_adv,vd_adv,vd,protect=load_record(os.path.join(args.attack_dir,res,f'{args.attack}_{file}.b'))
        except Exception as e:
            # img_adv,vd_adv,vd,protect
            print(e.args)
            print(os.path.join(args.attack_dir,res,f'{args.attack}_{file[:-4]}.b'))
            exit()
        text_num.append(len(vd.get_text()))
        dic=vs[file]
        ssim_i.append(dic['ssim_i'])
        psnr_i.append(dic['psnr_i'])
        L1_i.append(dic['L1_i'])
        ssim_o.append(dic['ssim_o'])
        psnr_o.append(dic['psnr_o'])
        L1_o.append(dic['L1_o'])
        ssim_oa.append(dic['ssim_oa'])
        psnr_oa.append(dic['psnr_oa'])
        L1_oa.append(dic['L1_oa'])
        ssim_b.append(dic['ssim_b'])
        psnr_b.append(dic['psnr_b'])
        L1_b.append(dic['L1_b'])
        if 0 in args.attack_p:
            occup.append(dic['occup'])
        else:
            text_content.append(dic['text_content'])
            stroke.append(dic['stroke'])
            shadow_visibility_flag.append(dic['shadow_visibility_flag'])
            stroke_visibility_flag.append(dic['stroke_visibility_flag'])
            font.append(dic['font'])
            iou_index.append(dic['iou_index'])
    #distance between images
    min_max_mean_ssim_i=min_max_mean(ssim_i)
    min_max_mean_psnr_i=min_max_mean(psnr_i)
    min_L1_i,max_L1_i,mean_L1_i,num_L1_i=min_max_mean(L1_i)
    
    min_ssim_o,max_ssim_o,mean_ssim_o,num_ssim_o=min_max_mean(ssim_o)
    min_psnr_o,max_psnr_o,mean_psnr_o,num_psnr_o=min_max_mean(psnr_o)
    min_L1_o,max_L1_o,mean_L1_o,num_L1_o=min_max_mean(L1_o)
    
    min_ssim_oa,max_ssim_oa,mean_ssim_oa,num_ssim_oa=min_max_mean(ssim_oa)
    min_psnr_oa,max_psnr_oa,mean_psnr_oa,num_psnr_oa=min_max_mean(psnr_oa)
    min_L1_oa,max_L1_oa,mean_L1_oa,num_L1_oa=min_max_mean(L1_oa)
    
    min_ssim_b,max_ssim_b,mean_ssim_b,num_ssim_b=min_max_mean(ssim_b)
    min_psnr_b,max_psnr_b,mean_psnr_b,num_psnr_b=min_max_mean(psnr_b)
    min_L1_b,max_L1_b,mean_L1_b,num_L1_b=min_max_mean(L1_b)

    #mask
    if 0 in args.attack_p:
        min_occup,max_occup,num_occup=min_max_mean(occup)
        
    
    else:
        #同一个字的区域,mask
        num_same_area=[len(i) for i in range(iou_index)]
        ratio_same_area=np.array(num_same_area)/np.array(text_num)
        min_same_area,max_same_area,mean_same_area,_=min_max_mean(ratio_same_area)
        #相同内容的比例
        ratio_same_content=np.array(txt_content)/np.array(text_num)
        min_same_contetn,max_same_contetn,mean_same_contetn,_=min_max_mean(ratio_same_contetn)
        #可见性
        num_same_shadow_visibility_flag=[len(i) for i in range(shadow_visibility_flag)]
        ratio_same_shadow_visibility_flag=np.array(num_same_shadow_visibility_flag)/np.array(text_num)
        min_same_shadow_visibility_flag,max_same_shadow_visibility_flag,mean_same_shadow_visibility_flag,_=min_max_mean(ratio_same_shadow_visibility_flag)

        num_same_stroke_visibility_flag=[len(i) for i in range(stroke_visibility_flag)]
        ratio_same_stroke_visibility_flag=np.array(num_same_stroke_visibility_flag)/np.array(text_num)
        min_same_stroke_visibility_flag,max_same_stroke_visibility_flag,mean_same_stroke_visibility_flag,_=min_max_mean(ratio_same_stroke_visibility_flag)

        #stroke 
        ratio_same_stroke=np.array(stroke)/np.array(text_num)
        min_same_stroke,max_same_stroke,mean_same_stroke,_=min_max_mean(ratio_same_stroke)

        #font
        ratio_same_font=np.array(font)/np.array(text_num)
        min_same_font,max_same_font,mean_same_font,_=min_max_mean(ratio_same_font)



    
        
if __name__=='__main__':
    args=parser.parse_args()
    print(args)
    generate(args)