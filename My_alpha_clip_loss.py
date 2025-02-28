import torch
import torchvision.transforms as transforms
import numpy as np

import clip
from PIL import Image
import alpha_clip
import torch.nn.functional as F
import math


class DirectionLoss(torch.nn.Module):

    def __init__(self, loss_type='mse'):
        super(DirectionLoss, self).__init__()

        self.loss_type = loss_type

        self.loss_func = {
            'mse': torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae': torch.nn.L1Loss
        }[loss_type]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)

        return self.loss_func(x, y)






class AL_CLIPLoss(torch.nn.Module):
    def __init__(self, device, path, direction_loss_type='cosine',
                 clip_model='ViT-B/32'):
        super(AL_CLIPLoss, self).__init__()

        self.device = device

        self.model, preprocess = alpha_clip.load(clip_model,
                                            alpha_vision_ckpt_pth=path,
                                            device=device)


        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0,
                                                                                                2.0])] +  # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                             preprocess.transforms[:2] +  # to match CLIP input scale assumptions
                                             preprocess.transforms[4:])  # + skip convert PIL to tensor  # + skip convert PIL to tensor


        self.direction_loss = DirectionLoss(direction_loss_type)

        self.mask_transform = transforms.Compose([

            transforms.Normalize([-1.0], [2.0]),
            transforms.Resize((224, 224)), # change to (336,336) when using ViT-L/14@336px
            transforms.Normalize([0.5], [0.26])
        ])


    def encode_images(self, images: torch.Tensor, mask:torch.Tensor) -> torch.Tensor:


        images = self.preprocess(images).half().cuda()

        alpha = self.mask_transform((mask).to(float)).unsqueeze(1)
        alpha = alpha.half().cuda()

        # return self.model.visual(images,alpha)
        return self.model.encode_image(images,alpha)

    def get_image_features(self, img: torch.Tensor,mask:torch.Tensor, norm: bool = True) -> torch.Tensor:

        image_features = self.encode_images(img, mask)

        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features





    def my_clip_directional_loss(self, source_lr_images, source_hr_images, target_lr_images,
                                 target_hr_iamges, mask) -> torch.Tensor:


        tar_lr_mask = mask['tar_lr']
        tar_gt_mask = mask['tar_gt']
        src_gt_mask = mask['src_gt']
        src_lr_mask = mask['src_lr']

        src_lr_encoding = self.get_image_features(source_lr_images,src_lr_mask)
        tar_lr_encoding = self.get_image_features(target_lr_images,tar_lr_mask)
        edit_direction_lr = (tar_lr_encoding - src_lr_encoding)
        edit_direction_lr /= (edit_direction_lr.clone().norm(dim=-1, keepdim=True))

        src_encoding = self.get_image_features(source_hr_images,src_gt_mask)
        target_encoding = self.get_image_features(target_hr_iamges,tar_gt_mask)

        edit_direction = (target_encoding - src_encoding)
        if edit_direction.sum() == 0:
            target_encoding = self.get_image_features(target_hr_iamges + 1e-6)
            edit_direction = (target_encoding - src_encoding)

        edit_direction /= (edit_direction.clone().norm(dim=-1, keepdim=True))

        sim = F.cosine_similarity(src_lr_encoding,tar_lr_encoding,dim=1,eps=1e-6)
        return (sim * self.direction_loss(edit_direction, edit_direction_lr)).mean()



