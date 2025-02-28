import os
from collections import OrderedDict
from basicsr.data.paired_image_dataset import Target_PairedImageDataset
from basicsr.data import build_dataloader
from basicsr.archs.safmn_arch import SAFMN
import torch
import sys
from basicsr.utils import tensor2img
from basicsr.metrics import calculate_metric

def test_samples(Weight,Test_root):


    test_lr = Test_root + os.sep + "LR"
    test_gt = Test_root + os.sep + "GT"
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


    psnr_all = 0.
    ssim_all = 0.
    niqe_all = 0.
    model_target.eval()
    with (torch.no_grad()):
        all_length = len(test_loader)

        for i_test, input_data in enumerate(test_loader):
            hr_image = input_data['gt']
            lr_image = input_data['lq'].cuda()

            target_hr_iamges_eval = model_target(lr_image)

            target_hr_iamges_eval = target_hr_iamges_eval.detach().cpu()
            hr_image = hr_image.cpu()
            target_hr_iamges_eval = tensor2img(target_hr_iamges_eval)
            hr_image = tensor2img(hr_image)

            res = {'img': target_hr_iamges_eval, "img2": hr_image}
            # psnr = torch.mean((target_hr_iamges_eval-hr_image)**2)

            opt_psnr = OrderedDict([('type', 'calculate_psnr'), ('crop_border', 4), ('test_y_channel', True)])
            opt_ssim = OrderedDict([('type', 'calculate_ssim'), ('crop_border', 4), ('test_y_channel', True)])
            opt_niqe = OrderedDict(
                [('type', 'calculate_niqe'), ('crop_border', 4), ('test_y_channel', True)])

            psnr = calculate_metric(res, opt_psnr)
            ssim = calculate_metric(res, opt_ssim)
            niqe = calculate_metric(res, opt_niqe)

            psnr_all += psnr
            ssim_all += ssim
            niqe_all += niqe

            print("\r", end="")
            print("percent:" + str(i_test / all_length), end="")
            # sys.stdout.write("\r{0}".format(i_tset/all_length))
            sys.stdout.flush()

        size = all_length
        print("PSNR:", psnr_all / size, "SSIM:", ssim_all / size, "NIQE:", niqe_all / size)





weight_path = input("Input weight path:")

Weight = weight_path

# Weight = "/home/tzz/Work_file/CODE/SAFMN/ZSSR_CLIP/Weight/ACDC_Cityscapes/Noema_Nomask_20_37.11114204892411.pth"
Test_root = "/disk2/AIM_D/DATASET/SODA_DRealSR/panasonic/Test_IODA"
# Test_root = "/media/tzz/Other/data/Cityscapes/Image_all/test/Valid"

test_samples(Weight,Test_root)







