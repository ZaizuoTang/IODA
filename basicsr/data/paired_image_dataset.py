import os
import random

import cv2
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop, paired_three_random_crop, augment_three
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY

import numpy as np
from torchvision import transforms

@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)




@DATASET_REGISTRY.register()
class Target_PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(Target_PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']


        if opt['phase'] == 'train':
            self.source_gt, self.source_lr = opt['dataroot_gt_source'], opt['dataroot_lr_source']


        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
            if opt['phase'] == 'train':
                self.path_source = paired_paths_from_lmdb([self.source_lr, self.source_gt], ['lq', 'gt'])


        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
            if opt['phase'] == 'train':
                self.path_source = paired_paths_from_meta_info_file([self.source_lr, self.source_gt], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)

        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
            if opt['phase'] == 'train':
                self.path_source = paired_paths_from_folder([self.source_lr, self.source_gt], ['lq', 'gt'], self.filename_tmpl)

        if opt['phase'] == 'train':
            self.mask_root = opt['instance_mask']


    def get_mask(self,image):

        # image = image*255.
        # mask = cv2.Canny(image.astype(np.uint8), 25, 80)
        h, w = image.shape[:2]
        # mask = np.zeros(shape=[h,w])

        mask = np.zeros(shape=[h, w])

        # #获取尺寸，随意画方框
        random_num = random.random()
        if random_num > 0.1:

            random_grad = False
            if random_grad == True:

                left_h = random.randrange(0,h-1)
                left_w = random.randrange(0, w - 1)
                right_h = random.randrange(left_h,h-1)
                right_w = random.randrange(left_w,w-1)
                mask[left_h:right_h,left_w:right_w]= 255
            else:
                grad_size = 16
                grad_h = h // grad_size
                grad_w = w // grad_size
                for i_h in range(grad_h):
                    for i_w in range(grad_w):
                        random_is = random.random()
                        if random_is>=0.5:
                            mask[i_h*grad_size:(i_h+1)*grad_size,i_w*grad_size:(i_w+1)*grad_size] = 255


        if len(mask.shape) == 2: binary_mask = (mask == 255)
        if len(mask.shape) == 3: binary_mask = (mask[:, :, 0] == 255)

        return binary_mask



    def get_paired_mask(self,imagelr,image_hr,mask_image):

        mask_image = mask_image.squeeze()

        num_all = np.unique(mask_image)

        h_lr, w_lr = imagelr.shape[:2]
        h_hr, w_hr = image_hr.shape[:2]

        # mask_lr = np.zeros(shape=[h_lr, w_lr])
        # mask_hr = np.zeros(shape=[h_hr, w_hr])


        #这是用来对掩码进行过滤的，仅仅保留像素点数量大于1000的掩码

        #用来统计掩码数量，将数量位于20%以下的舍弃


        num_new = []
        num_num = []
        Threshold = 1000
        for num in num_all:
            num_alone = len(np.where(mask_image==num)[0])
            if num_alone>= Threshold:
                num_new.append(num)
                num_num.append(num_alone)

        mask_blank = np.zeros(shape=[h_lr, w_lr])
        length_all = len(num_new)

        Num_mask_random = True

        if Num_mask_random == False:
            # 选取一个掩码
            mask_num = random.randint(0, length_all - 1)
            index_mask = np.where(mask_image==num_new[mask_num])
            mask_blank[index_mask] =255
        else:
            #随机选取多个掩码
            mask_index_list = []
            mask_num = random.randint(1, length_all)   #我比较担心万一数量全部没有达标咋个半，就是没有掩码数量超过1000
            for i_mask in range(mask_num):
                mask_index = random.randint(0, length_all - 1)

                while mask_index in mask_index_list:
                    mask_index = random.randint(0, length_all - 1)

                index_m = np.where(mask_image == num_new[mask_index])
                mask_blank[index_m] = 255
                mask_index_list.append(mask_index)


        mask_lr = mask_blank
        mask_hr = cv2.resize(mask_lr,[h_hr,w_hr])

        if len(mask_lr.shape) == 2: binary_mask = (mask_lr == 255)
        if len(mask_lr.shape) == 3: binary_mask = (mask_lr[:, :, 0] == 255)

        if len(mask_hr.shape) == 2: binary_mask_hr = (mask_hr == 255)
        if len(mask_hr.shape) == 3: binary_mask_hr = (mask_hr[:, :, 0] == 255)

        return binary_mask, binary_mask_hr


    # def get_paired_box_mask(self, imagelr, image_hr):
    #
    #
    #     h_lr, w_lr = imagelr.shape[:2]
    #     h_hr, w_hr = image_hr.shape[:2]
    #
    #     mask_lr = np.zeros(shape=[h_lr, w_lr])
    #     mask_hr = np.zeros(shape=[h_hr, w_hr])
    #
    #
    #     grad_size = 16
    #     grad_size_hr = grad_size*4
    #
    #
    #     random_num = random.random()
    #     if random_num > 0.0:
    #
    #         #随机画网格
    #         grad_h = h_lr // grad_size
    #         grad_w = w_lr // grad_size
    #         for i_h in range(grad_h):
    #             for i_w in range(grad_w):
    #                 random_is = random.random()
    #                 if random_is>=0.5:
    #                     mask_lr[i_h*grad_size:(i_h+1)*grad_size,i_w*grad_size:(i_w+1)*grad_size] = 255
    #                     mask_hr[i_h * grad_size_hr:(i_h + 1) * grad_size_hr, i_w * grad_size_hr:(i_w + 1) * grad_size_hr] = 255
    #
    #     if len(mask_lr.shape) == 2: binary_mask = (mask_lr == 255)
    #     if len(mask_lr.shape) == 3: binary_mask = (mask_lr[:, :, 0] == 255)
    #
    #     if len(mask_hr.shape) == 2: binary_mask_hr = (mask_hr == 255)
    #     if len(mask_hr.shape) == 3: binary_mask_hr = (mask_hr[:, :, 0] == 255)
    #
    #     return binary_mask, binary_mask_hr


























    def get_full_mask(self,image):
        h, w = image.shape[:2]
        mask = np.ones(shape=[h, w])*255
        # mask = np.zeros(shape=[h, w])

        if len(mask.shape) == 2: binary_mask = (mask == 255)
        if len(mask.shape) == 3: binary_mask = (mask[:, :, 0] == 255)

        return binary_mask



    #
    # def add_mask(self, image):
    #
    #     h, w = image.shape[:2]
    #
    #     random_num = random.random()
    #     if random_num > 0.5:
    #
    #         random_grad = False
    #         if random_grad == True:
    #
    #             left_h = random.randrange(0, h - 1)
    #             left_w = random.randrange(0, w - 1)
    #             right_h = random.randrange(left_h, h - 1)
    #             right_w = random.randrange(left_w, w - 1)
    #             image[left_h:right_h, left_w:right_w,:] = 0
    #         else:
    #             grad_size = 32  #64效果还好按
    #             grad_h = h // grad_size
    #             grad_w = w // grad_size
    #             for i_h in range(grad_h):
    #                 for i_w in range(grad_w):
    #                     random_is = random.random()
    #                     if random_is >= 0.5:
    #                         image[i_h * grad_size:(i_h+1) * grad_size, i_w * grad_size:(i_w+1) * grad_size,:] = 0
    #
    #
    #
    #     return image

    def add_paired_mask(self, img_lr,img_hr):

        h, w = img_lr.shape[:2]
        random_num = random.random()
        if random_num > 0.5:
            grad_size = 32
            grad_size_hr = grad_size * 4
            grad_h = h // grad_size
            grad_w = w // grad_size
            for i_h in range(grad_h):
                for i_w in range(grad_w):
                    random_is = random.random()
                    if random_is >= 0.5:
                        img_lr[i_h * grad_size:(i_h + 1) * grad_size, i_w * grad_size:(i_w + 1) * grad_size, :] = 0
                        img_hr[i_h * grad_size_hr:(i_h + 1) * grad_size_hr, i_w * grad_size_hr:(i_w + 1) * grad_size_hr, :] = 0


        return img_lr,img_hr









    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        if self.opt['phase'] == 'train':
            file_name = os.path.basename(lq_path).split(".")[0]
            mask_path = self.mask_root + os.sep + file_name + ".npy"
            mask_image = np.load(mask_path)



        if self.opt['phase'] == 'train':
            #随机获取原始域中的高低分辨率图片
            length_source = len(self.path_source)
            index_source = random.randint(0,length_source-1)

            gt_path_src = self.path_source[index_source]['gt_path']
            img_bytes = self.file_client.get(gt_path_src, 'gt')
            img_gt_source = imfrombytes(img_bytes, float32=True)

            lq_path_source = self.path_source[index_source]['lq_path']
            img_bytes = self.file_client.get(lq_path_source, 'lq')
            img_lq_source = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq, mask_image = paired_three_random_crop(img_gt, img_lq, gt_size, scale, gt_path, mask_image)
            img_gt_source,img_lq_source = paired_random_crop(img_gt_source, img_lq_source, gt_size, scale, gt_path_src)

            mask_image = np.expand_dims(mask_image,-1)
            # flip, rotation
            img_gt, img_lq, mask_image= augment([img_gt, img_lq,mask_image], self.opt['use_hflip'], self.opt['use_rot'])
            img_gt_source, img_lq_source = augment([img_gt_source, img_lq_source], self.opt['use_hflip'], self.opt['use_rot'])


            #percent:0.98PSNR: 25.708743666819572 SSIM: 0.7750117853580223 NIQE: 4.5309410475287155
            mask_dict = dict()


            mask_dict["tar_lr"],mask_dict["tar_gt"]  = self.get_paired_mask(img_lq,img_gt,mask_image)
            mask_dict["src_lr"] = self.get_full_mask(img_lq_source)
            mask_dict["src_gt"] = self.get_full_mask(img_gt_source)


            Add_mask = True
            if Add_mask == True:
                img_lq, img_gt = self.add_paired_mask(img_lq, img_gt)
                img_lq_source, img_gt_source = self.add_paired_mask(img_lq_source, img_gt_source)


            # cv2.imshow("1",img_gt)
            # cv2.imshow("2", img_lq)
            # cv2.imshow("3", img_gt_source)
            # cv2.imshow("4", img_lq_source)
            # cv2.imshow("5", (mask_dict["tar_gt"]*255).astype(np.uint8) )
            # cv2.imshow("6", (mask_dict["tar_lr"]*255).astype(np.uint8))
            # cv2.waitKey(0)
            # print("OK")





        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]
            img_gt_source = rgb2ycbcr(img_gt_source, y_only=True)[..., None]
            img_lq_source = rgb2ycbcr(img_lq_source, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]


        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        if self.opt['phase'] == 'train':
            img_gt_source, img_lq_source = img2tensor([img_gt_source, img_lq_source], bgr2rgb=True, float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_lq_source, self.mean, self.std, inplace=True)
            normalize(img_gt_source, self.mean, self.std, inplace=True)

        if self.opt['phase'] =='train':
            return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path,'lq_source':img_lq_source, 'gt_source':img_gt_source, 'mask':mask_dict}
        else:
            return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}


    def __len__(self):
        return len(self.paths)