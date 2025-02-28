import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import cv2
import torch
# from basicsr.models.sr_model import SRModel
from basicsr.archs.safmn_arch import SAFMN
from copy import deepcopy


from My_alpha_clip_loss import AL_CLIPLoss

from collections import OrderedDict


from basicsr.metrics import calculate_metric

from basicsr.data.paired_image_dataset import PairedImageDataset, Target_PairedImageDataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data import build_dataloader

from basicsr.utils import tensor2img
import sys
import datetime
import time

def main(checkpoint,epoch,Source_lr_root,Source_hr_root,Target_lr_root,Target_hr_root,test_root,batch_size,target_image_size,Weight_root,mask_root,clip_path):

    # Set up networks, optimizers.
    print("Initializing networks...")
    model_source =SAFMN(dim=128, n_blocks=16, ffn_scale=2.0, upscaling_factor=4).cuda()
    load_net = torch.load(checkpoint,map_location=lambda storage, loc: storage)
    print(model_source.load_state_dict(load_net["params"], strict=True))
    for param in model_source.parameters():
        param.requires_grad = False
    model_source.eval()



    model_target =SAFMN(dim=128, n_blocks=16, ffn_scale=2.0, upscaling_factor=4).cuda()
    load_net = torch.load(checkpoint,map_location=lambda storage, loc: storage)
    print(model_target.load_state_dict(load_net["params"], strict=True))



    CLIP_LOSS = AL_CLIPLoss(device="cuda", path=clip_path, clip_model='ViT-B/16')
    for param in CLIP_LOSS.parameters():
        param.requires_grad = False


    g_optim = torch.optim.Adam(
        model_target.parameters(),
        lr=0.00001,
        betas=(0 ** 0.8, 0.99 ** 0.8),
    )


    opt_tar_dataset = OrderedDict([('name', 'DF2K'), ('type', 'PairedImageDataset'), ('dataroot_gt', Target_hr_root),
                                   ('dataroot_lq', Target_lr_root), ('filename_tmpl', '{}'),
                                   ('io_backend', OrderedDict([('type', 'disk')])), ('gt_size', target_image_size*4),
                                   ('use_hflip', True),
                                   ('use_rot', True), ('use_shuffle', True), ('num_worker_per_gpu', 0),
                                   ('batch_size_per_gpu', batch_size),
                                   ('dataset_enlarge_ratio', 10), ('prefetch_mode', None), ('phase', 'train'),
                                   ('scale', 4),('dataroot_gt_source', Source_hr_root),('dataroot_lr_source', Source_lr_root),('instance_mask',mask_root)])

    Target_dataset_ori = Target_PairedImageDataset(opt_tar_dataset)
    Target_sampler = EnlargedSampler(Target_dataset_ori, num_replicas=1, rank=0, ratio=10)
    target_loader = build_dataloader(
        Target_dataset_ori,
        opt_tar_dataset,
        num_gpu=1,
        dist=False,
        sampler=Target_sampler,
        seed=10)


    test_lr = test_root + os.sep + "LR"
    test_gt = test_root + os.sep + "GT"
    opt_test_dataset = OrderedDict([('name', 'DIV2K_val100'), ('type', 'PairedImageDataset'),
                                   ('dataroot_gt', test_gt),
                                   ('dataroot_lq', test_lr),
                                   ('filename_tmpl', '{}'), ('io_backend', OrderedDict([('type', 'disk')])),
                                   ('phase', 'val'), ('scale', 4)])

    test_dataset_ori = Target_PairedImageDataset(opt_test_dataset)
    test_loader = build_dataloader(
        test_dataset_ori, opt_test_dataset, num_gpu=1, dist=False, sampler=None, seed=10)



    # Record PSNR
    current_time = datetime.datetime.now()
    text_file = open(str(current_time) + ".txt", "w")
    source_path = Source_lr_root
    target_path = Target_lr_root
    text_file.write(source_path)
    text_file.write("\n")
    text_file.write(target_path)
    text_file.write("\n")
    text_file.write("============================================")
    text_file.write("\n")
    timestamp = time.time()
    current_time = time.ctime(timestamp)
    text_file.write(current_time)
    text_file.write("\n")





    IS_initial_test = True
    if IS_initial_test == True:
        model_target.eval()
        with (torch.no_grad()):

            psnr_all = 0.
            # ssim_all = 0.

            for i_test, input_data in enumerate(test_loader):
                hr_image = input_data['gt']
                lr_image = input_data['lq'].cuda()

                target_hr_iamges_eval = model_target(lr_image)

                target_hr_iamges_eval = target_hr_iamges_eval.detach().cpu()
                hr_image = hr_image.cpu()
                target_hr_iamges_eval = tensor2img(target_hr_iamges_eval)
                hr_image = tensor2img(hr_image)

                res = {'img': target_hr_iamges_eval, "img2": hr_image}

                opt_psnr = OrderedDict([('type', 'calculate_psnr'), ('crop_border', 4), ('test_y_channel', True)])
                # opt_ssim = OrderedDict([('type', 'calculate_ssim'), ('crop_border', 4), ('test_y_channel', True)])
                psnr = calculate_metric(res, opt_psnr)
                # ssim = calculate_metric(res, opt_ssim)

                psnr_all += psnr
                # ssim_all += ssim
                print("\r", end="")
                print("Percent" + str(i_test/len(test_loader)),end="")
                # sys.stdout.write("\r{0}".format(i_tset/all_length))
                sys.stdout.flush()

            size = len(test_loader)
            # print("Zero Epoch"+"PSNR:", psnr_all / size, "SSIM:", ssim_all / size)
            print("Zero Epoch"+"PSNR:", psnr_all / size)

            Save_path = Weight_root + os.sep + "initial"+"_"+str(psnr_all / size) + ".pth"
            torch.save(model_target.state_dict(), Save_path)

            acc_str = "step_0:"+ str(psnr_all / size)
            text_file.write(acc_str)
            text_file.write("\n")
            timestamp = time.time()
            current_time = time.ctime(timestamp)
            text_file.write(current_time)
            text_file.write("\n")



    step =0
    for i_epoch in range(epoch):
            for datas_input in target_loader:


                source_lr_images = datas_input["lq_source"].cuda()
                target_lr_images = datas_input["lq"].cuda()


                with torch.no_grad(): 
                    source_sr_images = model_source(source_lr_images)


                model_target.train()
                target_hr_iamges = model_target(target_lr_images)

                mask = datas_input['mask']
                loss = CLIP_LOSS.my_clip_directional_loss(source_lr_images, source_sr_images, target_lr_images,
                                                          target_hr_iamges, mask)
                print(float(loss))

                model_target.zero_grad()
                loss.backward()
                g_optim.step()
                step += 1


                if step % 10 ==0:
                    model_target.eval()
                    with (torch.no_grad()):

                        psnr_all = 0.
                        ssim_all = 0.
                        niqe_all = 0.

                        all_length = len(test_loader)
                        for i_test, input_data in enumerate(test_loader):


                            hr_image = input_data['gt']
                            lr_image = input_data['lq'].cuda()

                            target_hr_iamges_eval = model_target(lr_image)

                            target_hr_iamges_eval = target_hr_iamges_eval.detach().cpu()
                            hr_image = hr_image.cpu()
                            target_hr_iamges_eval = tensor2img(target_hr_iamges_eval)
                            hr_image = tensor2img(hr_image)

                            res = {'img':target_hr_iamges_eval, "img2":hr_image}

                            opt_psnr = OrderedDict([('type', 'calculate_psnr'), ('crop_border', 4), ('test_y_channel', True)])
                            # opt_ssim = OrderedDict([('type', 'calculate_ssim'), ('crop_border', 4), ('test_y_channel', True)])
                            # opt_niqe = OrderedDict(
                            #     [('type', 'calculate_niqe'), ('crop_border', 4), ('test_y_channel', True)])
                            psnr = calculate_metric(res, opt_psnr)

                            # ssim = calculate_metric(res, opt_ssim)
                            # niqe = calculate_metric(res, opt_niqe)

                            psnr_all+=psnr
                            # ssim_all+=ssim
                            # niqe_all+=niqe

                            print("\r", end="")
                            print("percent:" + str(i_test / all_length), end="")
                            # sys.stdout.write("\r{0}".format(i_tset/all_length))
                            sys.stdout.flush()



                        size = all_length
                        # print("PSNR:",psnr_all/size,"SSIM:",ssim_all/size,"NIQE:",niqe_all/size)
                        print("PSNR:",psnr_all/size)



                        Save_path = Weight_root + os.sep + str(step) + "_" + str(psnr_all/size) + ".pth"
                        torch.save(model_target.state_dict(), Save_path)

                        acc_str = "step_" + str(step) + ":" + str(psnr_all / size)
                        text_file.write(acc_str)
                        text_file.write("\n")
                        timestamp = time.time()
                        current_time = time.ctime(timestamp)
                        text_file.write(current_time)
                        text_file.write("\n")



#Pretrain_model
checkpoint = "Path_to.../SAFMN_P_net_g_25000.pth"

#Alpha_CLIP weight
clip_path = "Path_to.../clip_b16_grit+mim_fultune_4xe.pth"


Source_lr_root = "Path_to...Source/LR"
Source_hr_root = "Path_to...Source/HR"
Target_lr_root = "Path_to...Target/LR"
Target_hr_root = "Path_to...Target/GT"
mask_root = "Path_to.../Mask"


# Test_Path
Test_Target_root= "Path_to.../Test_IODA"


target_image_size = 128  #128
batch_size = 2


Weight_root = "Path_to.../Save_weight_root"



epoch = 300


main(checkpoint,epoch,Source_lr_root,Source_hr_root,Target_lr_root,Target_hr_root,Test_Target_root,batch_size,target_image_size,Weight_root,mask_root,clip_path)