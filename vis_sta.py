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
parser.add_argument('--attack_p',type=int,nargs='+')#0 ocr
parser.add_argument('-g','--gpuid',default=0,type=int)
parser.add_argument('--T_visi',default=False,type=bool)

def generate(args):
    res=f'{args.GLI}_{args.process}_visi{1 if args.T_visi else 0}'
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
        img_adv,vd_adv,vd,protect=load_record(os.path.join(args.attack_dir,res,f'{args.attack}_{file[:-4]}.b'))
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
    for file in files:
        
if __name__=='__main__':
    args=parser.parse_args()
    print(args)
    generate(args)