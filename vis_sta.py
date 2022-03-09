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
parser.add_argument('--gs',default=0,type=int)
parser.add_argument('--replace',default=False,type=bool)

def generate(args):
    res=f'{args.GLI}_{args.attack_p}_visi{1 if args.T_visi else 0}'
    if args.gpuid<0:
        dev = torch.device(f"cpu")
    else:
        dev = torch.device(f"cuda:{args.gpuid}")
    if args.replace:
        save_dir=os.path.join(args.check_dir,res,'replace')
    else:
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
    # files=files[0:200]
    num=len(files)
    vs={}
    count=0
    for id,file in enumerate(files):
        logg.debug(f'{id}/{num} {file[:-4]}') 
        img = load_img(os.path.join(args.pics,file))
        try:
            img_adv,vd_adv,vd,protect=load_record(os.path.join(args.attack_dir,res,f'{args.attack}_{file[:-4]}.b'))
        except Exception as e:
            # img_adv,vd_adv,vd,protect
            print(e.args)
            print(os.path.join(args.attack_dir,res,f'{args.attack}_{file[:-4]}.b'))
            exit()
        
        img=img.to(dev)   
        img_adv=torch.tensor(img_adv.transpose((2,0,1)),dtype=torch.float32).unsqueeze(0)
        img_adv=img_adv.to(dev)
        if args.replace:
            vd_adv=get_vd_from_adv(img_adv,dev,model)
            save_file1=os.path.join(args.attack_dir,res,f'{args.attack}_{file[:-4]}_r.b')
            save_result(save_file1,vd_adv)
        ssim_i,psnr_i,L1_i=ssim_psnr_L1(torch.clamp(img_adv,0.,1.), torch.clamp(img,0.,1.))
        if vd_adv is not None:
            count+=1
            output_img = render_vd(vd)/255
            output_img_adv = render_vd(vd_adv)/255
            back_img = vd.bg/255
            back_img_adv = vd_adv.bg/255
            with torch.no_grad():
                output_img=torch.tensor(output_img).permute(2,0,1).unsqueeze(0).to(dev)
                output_img_adv=torch.tensor(output_img_adv).permute(2,0,1).unsqueeze(0).to(dev)
                back_img=torch.tensor(back_img).permute(2,0,1).unsqueeze(0)
                back_img_adv=torch.tensor(back_img_adv).permute(2,0,1).unsqueeze(0)
                ssim_o,psnr_o,L1_o=ssim_psnr_L1(torch.clamp(output_img,0.,1.), torch.clamp(img,0.,1.))
                # ssim_o=ssim(output_img, img, data_range=1.).item()
                # psnr_o=psnr(output_img, img, data_range=1.).item()
                # L1_o=F.l1_loss(output_img, img,reduction='mean').item()
                ssim_oa,psnr_oa,L1_oa=ssim_psnr_L1(torch.clamp(output_img_adv,0.,1.), torch.clamp(img,0.,1.))
                # ssim_oa=ssim(output_img_adv, img, data_range=1.).item()
                # psnr_oa=psnr(output_img_adv, img, data_range=1.).item()
                # L1_oa=F.l1_loss(output_img_adv, img,reduction='mean').item()
                ssim_b,psnr_b,L1_b=ssim_psnr_L1(torch.clamp(back_img,0.,1.),torch.clamp(back_img_adv,0.,1.))
                # ssim_b=ssim(back_img,back_img_adv, data_range=1.).item()
                # psnr_b=psnr(back_img,back_img_adv, data_range=1.).item()
                # L1_b=F.l1_loss(back_img,back_img_adv,reduction='mean').item()
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
        

        # if 0 in args.attack_p:
        if vd_adv is None:
                # occup=0.
                # txt_content=0
                # stroke=0
                # font=0
                # shadow_visibility_flag=[]#
                # stroke_visibility_flag=[]
                # iou_index=[]
                # blur=0
                # offset=0
                re=[0,0,[],[],0,None,None,[],0,0,0,0]
        else:     
                re=check(vd,vd_adv,protect)
        vs[file[:-4]]={'ssim_i':ssim_i,'psnr_i':psnr_i,'L1_i':L1_i,
                            'ssim_o':ssim_o,'psnr_o':psnr_o,'L1_o':L1_o,
                            'ssim_oa':ssim_oa,'psnr_oa':psnr_oa,'L1_oa':L1_oa,
                            'ssim_b':ssim_b,'psnr_b':psnr_b,'L1_b':L1_b,
                            'text_content':re[0],'stroke':re[1],
                            'shadow_visibility_flag':re[2],'stroke_visibility_flag':re[3],
                            'font':re[4],'blur':re[5],'offset':re[6],'iou_index':re[7],'shadow_v':re[8],'shadow_v_adv':re[9],'stroke_v':re[10],'stroke_v_adv':re[11]}
        
        # else:            
        #     re=check_1(img_adv,vd,vd_adv,protect)
        #     vs[file[:-4]]={'ssim_i':ssim_i,'psnr_i':psnr_i,'L1_i':L1_i,
        #                     'ssim_o':ssim_o,'psnr_o':psnr_o,'L1_o':L1_o,
        #                     'ssim_oa':ssim_oa,'psnr_oa':psnr_oa,'L1_oa':L1_oa,
        #                     'ssim_b':ssim_b,'psnr_b':psnr_b,'L1_b':L1_b,
        #                     'text_content':re[0],'stroke':re[1],
        #                     'shadow_visibility_flag':re[2],'stroke_visibility_flag':re[3],
        #                     'font':re[4],'blur':re[5],'offset':re[6],'iou_index':re[7]}
            

    logg.debug(count)
    save_file=os.path.join(save_dir,f'{args.attack_p[0]}_check.b')
    
    save_result(save_file,vs)
    

def static(args):
    res=f'{args.GLI}_{args.attack_p}_visi{1 if args.T_visi else 0}'
    if args.replace:
        pick_dir1=os.path.join(args.check_dir,res,'replace')
    else:
        pick_dir1=os.path.join(args.check_dir,res)
    # pick_dir1=os.path.join(args.check_dir,res)
    pick_dir2=os.path.join(args.attack_dir,res)
    check_file = f'{args.attack_p[0]}_check.b'
    files = os.listdir(args.pics)
    files.sort()

    try:
        with open(os.path.join(pick_dir1,check_file),'rb') as f:
            vs=pickle.load(f)
    except Exception as e:
        logg.error(f'{e.args} {os.path.join(pick_dir1,check_file)}')
        exit(1)
    
    # print(vs)

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
    shadow_v=[]
    shadow_v_adv=[]
    stroke_v=[]
    stroke_v_adv=[]
    font=[]
    iou_index=[]
    blur=[]
    offset=[]
    logg.debug(len(files))
    for file in files:
        logg.debug(file)
        file = file[:-4]
        try:
            img_adv,vd_adv,vd,protect=load_record(os.path.join(pick_dir2,f'{args.attack}_{file}.b'))
        except Exception as e:
            # img_adv,vd_adv,vd,protect
            print(e.args)
            print(os.path.join(args.attack_dir,res,f'{args.attack}_{file[:-4]}.b'))
            exit()
        if protect is None:
            text_num.append(len(vd.get_texts()))
        else:
            text_num.append(len(protect))
        dic=vs[file]
        # if dic['text_content']!=0:
        #     logg.debug(f"{file} {dic['text_content']} {dic['iou_index']}")
        #     for item in dic['iou_index']:
        #         logg.debug(f"{item} {vd.get_texts()[item[0]]} {vd_adv.get_texts()[item[1]]}")
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
        # if 0 in args.attack_p:
        #     occup.append(dic['occup'])
        #     if dic['occup']>0.98:
        #         logg.error(f'{file}')
        # else:
        text_content.append(dic['text_content'])
        stroke.append(dic['stroke'])
        shadow_visibility_flag.append(dic['shadow_visibility_flag'])
        stroke_visibility_flag.append(dic['stroke_visibility_flag'])
        font.append(dic['font'])
        iou_index.append(dic['iou_index'])
        blur.append(dic['blur'])
        offset.append(dic['offset'])
        shadow_v.append(dic['shadow_v'])
        shadow_v_adv.append(dic['shadow_v_adv'])
        stroke_v.append(dic['stroke_v'])
        stroke_v_adv.append(dic['stroke_v_adv'])
    #distance between images
    min_max_mean_ssim_i=min_max_mean(ssim_i)
    min_max_mean_psnr_i=min_max_mean(psnr_i)
    min_max_mean_L1_i=min_max_mean(L1_i)
    
    min_max_mean_ssim_o=min_max_mean(ssim_o)
    min_max_mean_psnr_o=min_max_mean(psnr_o)
    min_max_mean_L1_o=min_max_mean(L1_o)
    
    min_max_mean_ssim_oa=min_max_mean(ssim_oa)
    min_max_mean_psnr_oa=min_max_mean(psnr_oa)
    min_max_mean_L1_oa=min_max_mean(L1_oa)
    
    min_max_mean_ssim_b=min_max_mean(ssim_b)
    min_max_mean_psnr_b=min_max_mean(psnr_b)
    min_max_mean_L1_b=min_max_mean(L1_b)

    rows=['ssim_i','psnr_i','L1_i',
          'ssim_o','psnr_o','L1_o',
          'ssim_oa','psnr_oa','L1_oa',
          'ssim_b','psnr_b','L1_b']
    values=[min_max_mean_ssim_i,min_max_mean_psnr_i,min_max_mean_L1_i,
            min_max_mean_ssim_o,min_max_mean_psnr_o,min_max_mean_L1_o,
            min_max_mean_ssim_oa,min_max_mean_psnr_oa,min_max_mean_L1_oa,
            min_max_mean_ssim_b,min_max_mean_psnr_b,min_max_mean_L1_b]

    # if 0 in args.attack_p:
    #     min_max_mean_occup=min_max_mean(occup)
    #     rows=rows+['occup']
    #     values=values+[min_max_mean_occup]

    # else:
    if True:
        #同一个字的区域,mask
        num_area=[len(i) for i in iou_index]
        ratio_area=np.array(num_area)/np.array(text_num)
        num_area=[len(i) if len(i)>0 else 10000000 for i in iou_index]
        min_max_mean_area=min_max_mean(ratio_area)
        
        #相同内容的比例
        ratio_content=np.array(text_content)/np.array(num_area)
        min_max_mean_content=min_max_mean(ratio_content)
        

        #可见性
        # num_shadow_visibility_flag=[len(i) for i in shadow_visibility_flag]
        # ratio_shadow_visibility_flag=np.array(num_shadow_visibility_flag)/np.array(text_num)
        # min_max_mean_shadow_visibility_flag=min_max_mean(ratio_shadow_visibility_flag)

        # num_stroke_visibility_flag=[len(i) for i in stroke_visibility_flag]
        # ratio_stroke_visibility_flag=np.array(num_stroke_visibility_flag)/np.array(text_num)
        # min_max_mean_stroke_visibility_flag=min_max_mean(ratio_stroke_visibility_flag)

        stroke=sum(stroke_v_adv)/sum(stroke_v)
        shadow=sum(shadow_v_adv)/sum(shadow_v)
        min_max_mean_stroke_v=[0,0,stroke,len(stroke_v)]
        min_max_mean_shadow_v=[0,0,shadow,len(shadow_v)]
        
        #stroke 
        ratio_stroke=np.array(stroke)/np.array(num_area)
        min_max_mean_stroke=min_max_mean(ratio_stroke)

        #font
        ratio_font=np.array(font)/np.array(num_area)
        min_max_mean_font=min_max_mean(ratio_font)
        
        #blur
        min_max_mean_blur=min_max_mean(blur)

        #offset
        min_max_mean_offset=min_max_mean(offset)

        rows=rows+['area','content','shadow_v','stroke_v','stroke','shadow_blur','shadow_off','font']
        values=values+[min_max_mean_area,min_max_mean_content,min_max_mean_shadow_v,min_max_mean_stroke_v,min_max_mean_stroke,min_max_mean_blur,min_max_mean_offset,min_max_mean_font]
    
    s=form_print(rows,values)
    logg.debug(pick_dir1)
    with open(os.path.join(pick_dir1,'format.txt'),'w') as f:
        f.write(s)
        f.flush()

    a=[]
    b=[]
    for item in shadow_visibility_flag:
        a+=item
    for item in stroke_visibility_flag:
        b+=item
    a_t=0
    a_f=0
    b_f=0
    b_t=0
    for item in a:
        if item:
            a_t+=1
        else:
            a_f+=1
    for item in b:
        if item:
            b_t+=1
        else:
            b_f+=1

    logg.debug('shadow_vis:')
    logg.debug(f'True:{a_t} False:{a_f}')
    logg.debug('stroke_vis:')
    logg.debug(f'True:{b_t} False:{b_f}')   
if __name__=='__main__':
    args=parser.parse_args()
    print(args)
    if args.gs==0:
        generate(args)
    elif args.gs==1:
        logg.debug('static')
        static(args)
    else:
        raise NotImplementedError