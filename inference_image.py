import os
from collections import OrderedDict
from basicsr.data.paired_image_dataset import Target_PairedImageDataset
from basicsr.data import build_dataloader
from basicsr.archs.safmn_arch import SAFMN
import torch
import sys
from basicsr.utils import tensor2img
from basicsr.metrics import calculate_metric
import cv2

import PIL.Image
import numpy as np

import time

def test_samples(Weight,Test_root,Output_root):


    test_lr = Test_root + os.sep + "LR"
    test_gt = Test_root + os.sep + "HR"
    opt_test_dataset = OrderedDict([('name', 'DIV2K_val100'), ('type', 'PairedImageDataset'),
                                    ('dataroot_gt', test_gt),
                                    ('dataroot_lq', test_lr),
                                    ('filename_tmpl', '{}'), ('io_backend', OrderedDict([('type', 'disk')])),
                                    ('phase', 'val'), ('scale', 4)])

    test_dataset_ori = Target_PairedImageDataset(opt_test_dataset)
    test_loader = build_dataloader(
        test_dataset_ori, opt_test_dataset, num_gpu=1, dist=False, sampler=None, seed=10)


    model_target = SAFMN(dim=128, n_blocks=16, ffn_scale=2.0, upscaling_factor=4).cuda()
    load_net = torch.load(Weight, map_location=lambda storage, loc: storage)
    print(model_target.load_state_dict(load_net, strict=True))

    


    Output_GT_path = Output_root + os.sep + "GT"
    Output_LR_path = Output_root + os.sep + "LR"
    Output_SR_path = Output_root + os.sep + "SR"
    if not os.path.exists(Output_GT_path):
        os.mkdir(Output_GT_path)
    if not os.path.exists(Output_LR_path):
        os.mkdir(Output_LR_path)
    if not os.path.exists(Output_SR_path):
        os.mkdir(Output_SR_path)


    time_all = 0
    num = 0



    model_target.eval()
    with (torch.no_grad()):
        all_length = len(test_loader)

        for i_test, input_data in enumerate(test_loader):
            hr_image = input_data['gt']
            lr_image = input_data['lq'].cuda()

            num += 1
            time1 = time.time()
            target_hr_iamges_eval = model_target(lr_image)
            time2 = time.time()
            time_res = time2 - time1
            time_all += time_res


            target_hr_iamges_eval = target_hr_iamges_eval.detach().cpu()
            hr_image = hr_image.cpu()

            sr_image = tensor2img(target_hr_iamges_eval)
            hr_image = tensor2img(hr_image)

            gt_path = os.path.basename(input_data['gt_path'][0])
            lr_path = os.path.basename(input_data['lq_path'][0])

            sr_path = lr_path.replace("LR4","SR")


            gt_path = Output_GT_path + os.sep + gt_path
            sr_path = Output_SR_path + os.sep + sr_path
            lr_path = Output_LR_path + os.sep + lr_path


            lr_image = tensor2img(input_data['lq'])

            cv2.imwrite(lr_path, lr_image)
            cv2.imwrite(gt_path,hr_image)
            cv2.imwrite(sr_path, sr_image)


            print("\r", end="")
            print("percent:" + str(i_test / all_length), end="")
            # sys.stdout.write("\r{0}".format(i_tset/all_length))
            sys.stdout.flush()

    print(time_all/num)






# weight_path = input("请输入：")


#Night
weight_path = "/disk2/AIM_D/Weight/IODA/IODA_P/50_30.897628388334084.pth"
Weight = weight_path

Test_root = "/disk2/AIM_D/DATASET/SODA_DRealSR/panasonic/Test"
Output_root = "/disk2/AIM_D/ZODA/IODA/IODA_pan"

test_samples(Weight,Test_root,Output_root)


#有mask掩码
#PSNR: 38.91548402154208 SSIM: 0.9618092533118343 NIQE: 6.674977540006495
#无mask掩码




