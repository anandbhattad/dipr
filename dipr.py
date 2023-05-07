from __future__ import print_function
import os
import math
import numpy as np
import torch
import torch.optim
from torchvision import transforms, datasets, models
from torch.nn import functional as F
from models.Attention_reshade_inpaint_model import Inpaint_Color_Net, Inpaint_Color_Net_lite #inpainting network
from utils import *
from spherical_harmonics_helper import *
import matplotlib.pyplot as plt
from scipy.io import loadmat
from models.multi_task_model import net #surface normal predictor
from models.unet_shading_consist import UNet_small #shading consistency network

import argparse
import time
import os
import json
import random
import csv

start_time = time.time()

## runtime arguments
parser = argparse.ArgumentParser(description='Object Insertion by reshading')
parser.add_argument('--img', default=None, type=str, help='Specify image path where an object needs to be added')
parser.add_argument('--obj', default=None, nargs='+', type=str, help='Specify object path, + for multiple objects')
parser.add_argument('--mask', default=None, nargs='+', type=str, help='Specify object mask path + for multiple object masks')
parser.add_argument('--transparent', default=0, type=int, help='if background is transparent but no alpha image')
parser.add_argument('--loss_type', default="all", type=str, help='Specify type of losses -- all, spherical_harmonics, shading_consistency, base')
parser.add_argument('--outdir', default=None, type=str, help='Specify Directory to Save Models etc')
parser.add_argument('--load_size', default=1024, type=int, help='size of the image to be loaded')
parser.add_argument('--render_scale', default=0.25, type=float, help='size of the image to be rendered wrt load size')
parser.add_argument('--obj_size', default=256,  nargs='+', type=int, help='Specify which ball to remove; -1 to consider all balls' )
parser.add_argument('--seed', default=1024, type=int, help='From left at what weight the object needs to be placed in percentage wrt object center' )
parser.add_argument('--center_crop', default=None, type=str, help='crop obj at the center')
parser.add_argument('--std_partial', default=None, type=str, help='image_decomp_checkpoint')
parser.add_argument('--lr', default=0.0005, type=float, help='image_decomp_checkpoint')
parser.add_argument('--depth_aware', default=0, type=int, help='rerun_experiment')


args = parser.parse_args()


from models.image_decomposition import mlcrossenc #image decomposition network

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

args.outdir = os.path.join(args.outdir, args.img[:-4])
try:
    os.makedirs(args.outdir)
except OSError:
    pass


save_img_dir = os.path.join(args.outdir, args.loss_type)
try:
    os.makedirs(save_img_dir)
except OSError:
    pass



# parser = ArgumentParser()
# args = parser.parse_args()

# with open(args.outdir + '/commandline_args.txt', 'w') as f:
#     json.dump(args.__dict__, f, indent=2)



print(args)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PLOT = True

### Load all Pretrained Models 
## Image Decomposition 

allmodel=mlcrossenc(0, device).to(device)
checkpoint=torch.load('pretrained_models/intrinsic_decomp_checkpoint15.pth.tar')
allmodel.load_state_dict(checkpoint['allstate_dict'])
for param in allmodel.parameters():
    param.requires_grad = False
allmodel.eval();
allmodel.to(device);

## SurfaceNromal Predictor
multi_task_ckpt = torch.load('pretrained_models/ExpNYUD_three.ckpt')
NUM_CLASSES = 40
NUM_TASKS = 3 # segm + depth + normals
multi_task_model = net(num_classes=NUM_CLASSES, num_tasks=NUM_TASKS)
multi_task_model.load_state_dict(multi_task_ckpt['state_dict'])
for param in multi_task_model.parameters():
    param.requires_grad = False
    
multi_task_model.to(device)
multi_task_model.eval();
huberloss = torch.nn.SmoothL1Loss()


### Shading Consistency Network
shconsist = 'geometric'

Shc_model = UNet_small(6, 2)
Shc_model.load_state_dict(torch.load("pretrained_models/unet_Shc_small_multi-task_normals.pth"))
print("Loaded Pretrained Model")
for param in Shc_model.parameters():
    param.requires_grad = False
Shc_model = Shc_model.to(device)
Shc_model.eval();
CELoss = nn.CrossEntropyLoss()


imsize = (args.load_size, args.load_size)

import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable

def normalize(im):
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    for i in range(3):
        im[:,i,...] = (im[:,i,...] - mean[i])/std[i]
    return im


img_path  = args.img
obj_path = args.obj
mask_path = args.mask
outdir = args.outdir
run_type = args.loss_type 



img_pil, img_np = get_image(img_path, imsize)
img_np = img_np[:3, ...]


NET_TYPE = 'skip_depth6' # one of skip_depth4|skip_depth2|UNET|ResNet

def random_circular_obj(x, y, r, scale=4):
    #y = np.random.randint(200, 240, 1)
    #x = np.random.randint(200, 250, 1)

    #r = np.random.choice([10, 15, 20])#np.random.randint(40)
    mask = Image.new('RGB', (256*scale, 256*scale), (0, 0, 0))
    draw = ImageDraw.Draw(mask)
    draw.ellipse(((x-r)*scale, (y-r)*scale, (x+r)*scale, (y+r)*scale), fill=(255, 255, 255,0))
    obj = Image.new('RGB', (256*scale, 256*scale), (0, 0, 0))
    draw = ImageDraw.Draw(obj)
    color = np.random.choice([255])
    draw.ellipse(((x-r)*scale, (y-r)*scale, (x+r)*scale, (y+r)*scale), fill=(color, color, color,0))
    return obj, mask

obj0, _ = random_circular_obj(70, 225, 15)
obj1, _ = random_circular_obj(185, 155, 15)
obj2, _ = random_circular_obj(100, 165, 15)
obj3, _ = random_circular_obj(140, 200, 15)
obj4, _ = random_circular_obj(35, 145, 15)
obj5, _ = random_circular_obj(240, 160, 15)
obj6, _ = random_circular_obj(220, 210, 15)
obj7, _ = random_circular_obj(138, 40, 15)
obj8, _ = random_circular_obj(50, 50, 15)
obj9, _ = random_circular_obj(200, 50, 15)
obj10, _ = random_circular_obj(20, 240, 15)


obj0_np = pil_to_np(obj0)
obj1_np = pil_to_np(obj1)
obj2_np = pil_to_np(obj2)
obj3_np = pil_to_np(obj3)
obj4_np = pil_to_np(obj4)
obj5_np = pil_to_np(obj5)
obj6_np = pil_to_np(obj6)
obj7_np = pil_to_np(obj7)
obj8_np = pil_to_np(obj8)
obj9_np = pil_to_np(obj9)
obj10_np = pil_to_np(obj10)
x = [70, 185, 100, 140, 35, 240, 220, 138, 50, 200, 20]
y = [225, 155, 165, 200, 145, 160, 210, 40, 50, 50, 240]
all_obj = [obj0_np, obj1_np, obj2_np, obj3_np, obj4_np, obj5_np, obj6_np, obj7_np, obj8_np, obj9_np, obj10_np]

img_mask_np = sum(all_obj)
#else:
    #img_mask_np = all_obj[args.pos_add]
# img_mask_np = obj6_np
#man_np = (1-img_mask_np)*(img_np)+img_mask_np
img_mask_var = np_to_torch(img_mask_np).type(dtype)

img_var = np_to_torch(img_np).type(dtype)
mask_var = np_to_torch(img_mask_np).type(dtype)
manipulated_var = ((1-mask_var)*img_var+mask_var)


img_hr = torch.nn.functional.interpolate(
            img_var, (1024, 1024), mode="bilinear", align_corners=False).to(device)
man_hr = torch.nn.functional.interpolate(
            manipulated_var, (1024, 1024), mode="bilinear", align_corners=False).to(device)
mask_hr = torch.nn.functional.interpolate(
            mask_var, (1024, 1024), mode="bilinear", align_corners=False).to(device)


img_var = torch.nn.functional.interpolate(
            img_var, (256, 256), mode="bilinear", align_corners=False).to(device)
manipulated_var = torch.nn.functional.interpolate(
            manipulated_var, (256, 256), mode="bilinear", align_corners=False).to(device)
mask_var = torch.nn.functional.interpolate(
            mask_var, (256, 256), mode="bilinear", align_corners=False).to(device)

mask_var = mask_var.clamp(min=0, max=1.)

torch_all_mask = []
for iter, mask_items in enumerate(all_obj):
    torch_all_mask.append(torch.nn.functional.interpolate( 
            np_to_torch(mask_items).type(dtype).to(device), (256, 256), mode="bilinear", align_corners=False))

save_image(outdir, mask_var, 'mask')


# print(img_var.shape, mask_var.shape, manipulated_var.shape)
# plot_image_grid([torch_to_np(mask_var), torch_to_np(img_var), torch_to_np(manipulated_var)], 3,11); 
save_image(outdir, man_hr, 'cut_paste')
save_image(outdir, img_hr, "input")
save_image(outdir, mask_hr, "all_masks")

# obj_reshape = np.zeros_like(img_np)
# mask_reshape = np.zeros_like(img_np)
# obj_dummy = np.zeros_like(img_np)
# mask_dummy = np.zeros_like(img_np)


# man_np = img_mask_np[:3, ...]*obj_np + (1-img_mask_np[:3, ...])*img_np
# img_var = np_to_torch(img_np).type(dtype).to(device)

# manipulated_var = np_to_torch(man_np).type(dtype).to(device)
# mask_var = np_to_torch(img_mask_np).type(dtype).to(device)

# render_size = int(args.load_size*args.render_scale)
# render_size = (render_size, render_size)
# img_var = torch.nn.functional.interpolate(
#             img_var, render_size, mode="bilinear", align_corners=False).to(device)
# manipulated_var = torch.nn.functional.interpolate(
#             manipulated_var, render_size, mode="bilinear", align_corners=False).to(device)
# mask_var = torch.nn.functional.interpolate(
#             mask_var, render_size, mode="bilinear", align_corners=False).to(device)
# mask_var = mask_var.clamp(min=0, max=1.)
# print(img_var.shape, mask_var.shape, manipulated_var.shape)
# plot_image_grid([torch_to_np(mask_var), torch_to_np(img_var), torch_to_np(manipulated_var)], 3,11); 
# save_image(outdir, manipulated_var, 'cut_paste')
# save_image(outdir, mask_var, 'mask')

def optimize(optimizer_type, parameters, closure, LR, num_iter):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')        
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        
        optimizer = torch.optim.Adam(parameters, lr=LR)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                    step_size=200, gamma=0.9)
        lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer,
                                    step_size=500, gamma=0.9)       
        for j in range(num_iter):
            optimizer.zero_grad()
            closure()
            optimizer.step()
            if j<5000:
                lr_scheduler.step()
            else:
                lr_scheduler2.step()
            
            if (j+1)%500==0:
                print("learning rate now:", optimizer.param_groups[0]['lr'])

    else:
        assert False

def reshading(img_var, manipulated_var, mask_var, run_type, model_dir, inpaint=True):
    
    if run_type == "all":
        shading_consistency = True
        spherical_harmonics = True
    elif run_type == "shading_consistency":
        shading_consistency = True
        spherical_harmonics = False
    elif run_type == "spherical_harmonics":
        shading_consistency = False
        spherical_harmonics = True
    elif run_type == "base":
        shading_consistency = False
        spherical_harmonics = False
    else:
        return print("Run type error: please check run type")


    # pad = 'reflection' # 'zero'
    OPT_OVER = 'net'
    OPTIMIZER = 'adam'
    # INPUT = 'noise'
    # input_depth = 3

    num_iter = 10001
    show_every = 1000
    figsize = 8
    reg_noise_std = 0.00
    param_noise = True


    net = Inpaint_Color_Net_lite(channel_att=True, spatial_att=True, depth_aware=args.depth_aware)

    # Compute number of parameters
    s  = sum(np.prod(list(p.size())) for p in net.parameters())
    print ('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)
    L1 = torch.nn.L1Loss().type(dtype)
    net.to(device);
 
    if inpaint:
        print("Using Inapinting Masks")
        mask_tensor = torch.cat([mask_var, mask_var], 0).to(device)
    else:
        mask_tensor = torch.cat([torch.zeros_like(mask_var), mask_var], 0).to(device)

    image_tensor = torch.cat([img_var, manipulated_var], 0).to(device)
    net_input = image_tensor.clone()

    if args.std_partial:
        net_input = torch.cat([net_input, 1-mask_tensor[:, 1].unsqueeze(1)], 1)
    else:
        net_input = torch.cat([net_input, mask_tensor[:, 1].unsqueeze(1)], 1)
    #net_input = torch.cat([image_tensor, mask_tensor[:, 1].unsqueeze(1)], 1)


    albedo, shading, _, gloss, _, _, _ = allmodel(image_tensor)
    
    albedo = albedo.clamp(min=0, max=1)
    #albedo[1]*mask_var.squeeze(0) = (albedo[1]-gloss[1])*mask_var.squeeze(0)

    _, _, out_norm = multi_task_model(normalize(image_tensor.detach().clone()))

    #img_norm = torch.nn.functional.interpolate(
        #img_norm, imsize, mode="bilinear", align_corners=False)
    out_norm = torch.nn.functional.interpolate(
         out_norm, (256, 256), mode="bilinear", align_corners=False)
    out_norm = F.normalize(out_norm, dim=1)

    if spherical_harmonics:

        SH_basis_src1 = get_SH_basis_no_coeff(out_norm[0].unsqueeze(0))

        SH_coeff1 = least_square_solver_SHCoeff(SH_basis_src1, shading[0].unsqueeze(0)+gloss[0].unsqueeze(0))

        SH_basis_tar1 = get_SH_basis_no_coeff(out_norm[1].unsqueeze(0))


    out_norm[:, 0, ...] = ((out_norm[:, 0, ...] + 1.) / 2.) 
    out_norm[:, 1, ...] = ((out_norm[:, 1, ...] + 1.) / 2.)
    out_norm[:, 2, ...] = ((1. - out_norm[:, 2, ...]) / 2.)

    if shading_consistency:
        
        consist_spatial_label = torch.ones(2, 256, 256).type(torch.LongTensor).to(device)
        consist_code_label = torch.tensor(1).type(torch.LongTensor).unsqueeze(0).to(device)
        consist_code_label = torch.cat([consist_code_label, consist_code_label], 0)
        
    albedo = albedo.clamp(min=0, max=1)

    albedo, shading, gloss, out_norm = albedo.detach(), shading.detach(), gloss.detach(), out_norm.detach()

    i = 0
    is_best = True
    albedo_weight = torch.tensor(1.0).requires_grad_().to(device)
    alambda = nn.Parameter(albedo_weight, requires_grad=True)
    is_best = True
    LR = args.lr 
    def closure():    
        global i
        global a_loss
        global s_loss 
        global g_loss
        global rec_loss
        global res_loss
        global p_loss
        global shc_loss
        global sph_loss
        global lam_loss
        global t_loss
        global best_loss
        if param_noise:
            for n in [x for x in net.parameters() if len(x.size()) == 4]:
                n = n + n.detach().clone().normal_() * n.std() / 50

        net_input = net_input_saved
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)


        out = net(net_input)
        recons_loss = mse(out*(1-mask_tensor), image_tensor*(1-mask_tensor))

        out_apred, out_spred, _, out_gpred, _, _, residual_loss = allmodel(out)
        comp_out = (out_apred*out_spred+out_gpred)
            
        residual_loss = 0.2*mse(comp_out*(1-mask_tensor), image_tensor*(1-mask_tensor))

        albedo_loss = mse(out_apred*(1-mask_tensor), albedo*(1-mask_tensor)) + mse(out_apred*mask_tensor*alambda, albedo*mask_tensor*alambda) ##5 times mask albedo
        lambda_loss = 0.1*mse(alambda, torch.ones_like(alambda))
        shading_loss = mse(out_spred*(1-mask_tensor), shading*(1-mask_tensor))

        total_loss = recons_loss + residual_loss + albedo_loss + lambda_loss + shading_loss #+perceptual_loss


        gloss_loss = mse(out_gpred*(1-mask_tensor), gloss*(1-mask_tensor)) 
        total_loss+=gloss_loss



        if shading_consistency:

            out_shnorm = torch.cat([(out_spred), out_norm], dim=1)
            out_spatial, out_code = Shc_model(out_shnorm)

            shading_consist_loss = CELoss(out_spatial, consist_spatial_label)+ CELoss(out_code, consist_code_label) 

            total_loss +=shading_consist_loss
            
        else:
            shading_consist_loss = torch.Tensor([0])

        if spherical_harmonics:

            SH_coeff_pred1 = least_square_solver_SHCoeff(SH_basis_tar1,  out_spred[1].unsqueeze(0)+out_gpred[1].unsqueeze(0))
            SH_loss = huberloss(SH_coeff_pred1, SH_coeff1)
            total_loss+=SH_loss

        else:
            SH_loss = torch.Tensor([0])
            
            
        total_loss.backward()


        print ('Iteration %05d Loss %f lambda  %f  shconsist_loss  %f SHLoss  %f'% (i, total_loss.item(), alambda.item(), shading_consist_loss.item(), SH_loss.item()), '\r', end='')
        if  PLOT and (i+1) % show_every == 0:
            plot_image_grid([torch_to_np(out[0].unsqueeze(0)), torch_to_np(out[1].unsqueeze(0))], factor=figsize)


        if (i+1)%100==0:
            a_loss.append(albedo_loss.item())
            s_loss.append(shading_loss.item()) 
            g_loss.append(gloss_loss.item())
            rec_loss.append(recons_loss.item())
            res_loss.append(residual_loss.item())
            shc_loss.append(shading_consist_loss.item())
            sph_loss.append(SH_loss.item())
            lam_loss.append(lambda_loss.item())
            t_loss.append(total_loss.item())


            if best_loss >= shading_loss.item()+shading_consist_loss.item()+SH_loss.item():
                best_loss = shading_loss+shading_consist_loss+SH_loss
                if args.std_partial:
                    torch.save(net, model_dir + '/%sLosses_attention_inpainting_std_partial_conv%s_lite_checkpoint.pth'%(run_type, inpaint)) 
                else:
                    torch.save(net, model_dir + '/%sLosses_attention_inpainting_partial_conv%s_lite_checkpoint.pth'%(run_type, inpaint)) 
        i += 1

        return total_loss

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    p = get_params(OPT_OVER, net, net_input)
    print("learning_Rate", LR)
    optimize(OPTIMIZER, p+[alambda], closure, LR, num_iter)

i=0
best_loss=9999
a_loss = []
s_loss = []
g_loss = []

rec_loss = []
res_loss = []

shc_loss = []
p_loss = []
sph_loss = [] 

lam_loss = []
learning_rate = []
t_loss = []
b_loss = []
reshading(img_var, manipulated_var, mask_var, run_type, model_dir=outdir, inpaint=True)
print("--- %s seconds ---" % (time.time() - start_time))
sum_list = [sum(x) for x in zip(*[s_loss, shc_loss, sph_loss])]
plt.figure(figsize=(8,8))
plt.plot(np.array(t_loss[1:])/t_loss[1])
plt.plot(np.array(s_loss[1:])/t_loss[1])
plt.plot(np.array(shc_loss[1:])/t_loss[1])
plt.plot(np.array(sph_loss[1:])/t_loss[1])
plt.plot(np.array(sum_list[1:])/t_loss[1])
plt.savefig(outdir + "/%s_losses.png"%(run_type))


# Post Processing --> Shading is smooth in our case, so we can upscale the predicted shading
# also we will render with and without object using the same network
with torch.no_grad():
    #albedo_hr, shading_man, _, gloss_man,  _, _, res_cut_paste = allmodel(man_hr)
    #albedo_hr = albedo_hr.clamp(min=0, max=1)
    image_tensor = torch.cat([img_var, manipulated_var], 0)
    mask_tensor = torch.cat([mask_var, mask_var], 0).to(device)
    net_input = image_tensor.clone()
    net_input = torch.cat([net_input, mask_tensor[:, 1].unsqueeze(1)], 1)
    albedo, shading, _, gloss, _, _, _ = allmodel(image_tensor)
    albedo = albedo.clamp(min=0, max=1)
    net = torch.load(outdir + '/%sLosses_attention_inpainting_partial_conv%s_lite_checkpoint.pth'%(run_type, True))
    net.eval();
    #net = torch.load("results/sphere_balls_ade_val_516/all_inpainting_inv_partial_convTrue_lite_checkpoint.pth")
    out = net(net_input)
    I = mask_var*out[1] + (1 - mask_var)*(img_var + out[1] - out[0]) #to account for any shading changes outside the mask
    final_apred, final_spred, _, final_gpred, _, _, _ = allmodel(I)
    final_apred = final_apred.clamp(min=0, max=1)
    res = I - (final_apred*final_spred+final_gpred)
    I1  = (albedo[1].unsqueeze(0)*final_spred + final_gpred)*mask_var + img_var*(1-mask_var)
    # shading_hr = torch.nn.functional.interpolate(
    #         final_spred, (1024, 1024), mode="bilinear", align_corners=False)

    # gloss_hr = torch.nn.functional.interpolate(
    #         final_gpred, (1024, 1024), mode="bilinear", align_corners=False)
    
    # res_hr = torch.nn.functional.interpolate(
    #         res, (1024, 1024), mode="bilinear", align_corners=False)
    # I2 = ((albedo_hr)*shading_hr+gloss_hr)*mask_hr + (img_hr)*(1-mask_hr) 
    # final1 = torch_to_np(I2)
    # I3 = ((albedo_hr)*shading_hr+gloss_hr+res_cut_paste)*mask_hr + (img_hr)*(1-mask_hr) 
    # final2 = torch_to_np(I3)
    # plot_image_grid([man_np, final1, final2], 6, factor=25);
    # plot_image_grid([img_np, man_np, final2], 6, factor=25);
    save_image(outdir, img_var, "target_%s"%(args.img.split("/")[-1][:-4]))
    save_image(outdir, manipulated_var, "target_%s_source_%s_cut_and_paste"%(args.img.split("/")[-1][:-4], "spheres_solid"))
    save_image(outdir, I1, "target_%s_source_%s_reshaded_%s"%(args.img.split("/")[-1][:-4], "spheres_solid", args.loss_type))