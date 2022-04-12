import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import traceback
import json
import pyexr


class Dataset:
    def __init__(self, data_dir):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.data_dir = data_dir

        try:
            self.images_lis = sorted(
                glob(os.path.join(self.data_dir, 'image/*.png')))
            images_np = np.stack([cv.imread(im_name)
                                 for im_name in self.images_lis]) / 255.0
        except:
            traceback.print_exc()
            ic('Loading png images failed; try loading exr images')
            self.images_lis = sorted(
                glob(os.path.join(self.data_dir, 'image/*.exr')))
            images_np = np.clip(np.power(np.stack([pyexr.open(im_name).get()[
                                :, :, ::-1] for im_name in self.images_lis]), 1./2.2), 0., 1.)
        self.n_images = len(self.images_lis)

        no_mask = True
        self.masks_lis = None
        masks_np = np.ones_like(images_np)
        if not no_mask:
            try:
                self.masks_lis = sorted(
                    glob(os.path.join(self.data_dir, 'mask/*.png')))
                assert (len(self.masks_lis) == len(
                    self.images_lis)), f'# masks {len(self.masks_lis)} != # images {len(self.images_lis)}'
                masks_np = np.stack([cv.imread(im_name)
                                    for im_name in self.masks_lis]) / 255.0
            except:
                # traceback.print_exc()
                ic('Loading mask images failed')
        if self.masks_lis is None:
            ic('Not using masks!')

        self.images = torch.from_numpy(images_np[..., :3].astype(
            np.float32)).cpu()  # [n_images, H, W, 3]
        self.masks = torch.from_numpy(masks_np[..., :3].astype(
            np.float32)).cpu()   # [n_images, H, W, 3]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        ic(self.images.shape, self.masks.shape)
        ic(self.images.min().item(), self.images.max().item())

        # we assume the scene to render is inside a unit sphere at origin.
        # put all the camera parameters on cpu using float64
        self.camera_dict = json.load(
            open(os.path.join(self.data_dir, 'cam_dict_norm.json')))
        self.intrinsics_all_inv = []
        self.pose_all = []
        for x in self.images_lis:
            x = os.path.basename(x)[:-4] + '.png'
            K = np.array(self.camera_dict[x]['K']).reshape((4, 4))
            W2C = np.array(self.camera_dict[x]['W2C']).reshape((4, 4))
            K_inv = np.linalg.inv(K)
            C2W = np.linalg.inv(W2C)
            # ic(K_inv.dtype, torch.from_numpy(K_inv).dtype)
            self.intrinsics_all_inv.append(torch.from_numpy(K_inv))
            self.pose_all.append(torch.from_numpy(C2W))

        self.intrinsics_all_inv = torch.stack(
            self.intrinsics_all_inv)   # [n_images, 4, 4]
        self.pose_all = torch.stack(self.pose_all)  # [n_images, 4, 4]
        assert (self.intrinsics_all_inv.dtype == torch.float64), 'self.intrinsics_all_inv.dtype: {}'.format(
            self.intrinsics_all_inv.dtype)
        assert (self.pose_all.dtype == torch.float64), 'self.poses_all.dtype: {}'.format(
            self.pose_all.dtype)

        # region of interest to **extract mesh**
        self.object_bbox_min = np.array([-1.01, -1.01, -1.01])
        self.object_bbox_max = np.array([1.01,  1.01,  1.01])
        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        # compute on cpu with float64 precision
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack(
            [pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).cpu().double()  # W, H, 3
        p = torch.matmul(
            self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2,
                                       dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(
            self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None,
                               :3, 3].expand(rays_v.shape)  # W, H, 3
        # shift rays_o along rays_v towards the scene to shrink the huge depth
        rays_o = rays_o + rays_v * \
            (torch.norm(rays_o, dim=-1, keepdim=True) - 5.)
        assert (rays_o.dtype == torch.float64), 'rays_o.dtype: {}'.format(
            rays_o.dtype)
        assert (rays_v.dtype == torch.float64), 'rays_v.dtype: {}'.format(
            rays_v.dtype)
        rays_o = rays_o.float().to(self.device)
        rays_v = rays_v.float().to(self.device)
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask = self.masks[img_idx][(pixels_y, pixels_x)]      # batch_size, 3
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(
            pixels_y)], dim=-1).cpu().double()  # batch_size, 3
        # batch_size, 3
        p = torch.matmul(
            self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze()
        # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)
        rays_v = torch.matmul(
            self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(
            rays_v.shape)  # batch_size, 3
        # shift rays_o along rays_v towards the scene to shrink the huge depth
        rays_o = rays_o + rays_v * \
            (torch.norm(rays_o, dim=-1, keepdim=True) - 5.)
        assert (rays_o.dtype == torch.float64), 'rays_o.dtype: {}'.format(
            rays_o.dtype)
        assert (rays_v.dtype == torch.float64), 'rays_v.dtype: {}'.format(
            rays_v.dtype)
        rays_o = rays_o.float().to(self.device)
        rays_v = rays_v.float().to(self.device)
        # batch_size, 10
        return torch.cat([rays_o, rays_v, color.to(self.device), mask[:, :1].to(self.device)], dim=-1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        # handle the case where the camera is inside the unit sphere
        near = near.clamp_(min=1e-5)
        # clip the bbx z=-0.3 to 0.3?
        
        return near, far

    def image_at(self, idx, resolution_level):
        if self.images_lis[idx].endswith('.exr'):
            img = np.power(pyexr.open(self.images_lis[idx]).get()[
                           :, :, ::-1], 1./2.2) * 255.
        else:
            img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255).astype(np.uint8)
