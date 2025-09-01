import os
import os.path as osp
import json
import copy
import numpy as np
import PIL.Image as pil
import pickle as pkl
import torch
from torch.utils.data import Dataset

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from .data_util import img_loader, mask_loader_scene, align_dataset, transform_mask_sample
from utils.visualize import show_data

def is_numpy(data):
    """Checks if data is a numpy array."""
    return isinstance(data, np.ndarray)

def is_tensor(data):
    """Checks if data is a torch tensor."""
    return type(data) == torch.Tensor

def is_list(data):
    """Checks if data is a list."""
    return isinstance(data, list)

def stack_sample(sample):
    """Stack a sample from multiple sensors"""
    # If there is only one sensor don't do anything
    if len(sample) == 1:
        return sample[0]

    # Otherwise, stack sample
    stacked_sample = {}
    for key in sample[0]:
        # Global keys (do not stack)
        if key in ['idx', 'dataset_idx', 'sensor_name', 'filename', 'token']:
            stacked_sample[key] = sample[0][key]
        else:
            # Stack torch tensors
            if is_tensor(sample[0][key]):
                stacked_sample[key] = torch.stack([s[key] for s in sample], 0)
            # Stack numpy arrays
            elif is_numpy(sample[0][key]):
                stacked_sample[key] = np.stack([s[key] for s in sample], 0)
            # Stack list
            elif is_list(sample[0][key]):
                stacked_sample[key] = []
                # Stack list of torch tensors
                if is_tensor(sample[0][key][0]):
                    for i in range(len(sample[0][key])):
                        stacked_sample[key].append(
                            torch.stack([s[key][i] for s in sample], 0))
                # Stack list of numpy arrays
                if is_numpy(sample[0][key][0]):
                    for i in range(len(sample[0][key])):
                        stacked_sample[key].append(
                            np.stack([s[key][i] for s in sample], 0))

    # Return stacked sample
    return stacked_sample


def load_info(info):  # TODO: 似乎不应该再 flip_yz
    img_path = info["data_path"]
    # use lidar coordinate of the key frame as the world coordinate
    c2w = info["sensor2lidar_transform"]
    # opencv cam -> opengl cam, maybe not necessary!
    # flip_yz = np.eye(4)
    # flip_yz[1, 1] = -1
    # flip_yz[2, 2] = -1
    # c2w = c2w@flip_yz

    lidar2cam_r = np.linalg.inv(info["sensor2lidar_rotation"])
    lidar2cam_t = info["sensor2lidar_translation"] @ lidar2cam_r.T
    w2c = np.eye(4)
    w2c[:3, :3] = lidar2cam_r.T
    w2c[3, :3] = -lidar2cam_t
    
    return img_path, c2w, w2c


class OmnisceneDataset(Dataset):
    """
    Loaders for Omniscene dataset
    """
    def __init__(self, path, split,
                 cameras=None,
                 back_context=0,
                 forward_context=0,
                 data_transform=None,
                 depth_type=None,
                 scale_range=2,
                 with_pose=None,
                 with_ego_pose=None,
                 with_mask=None,
                 ):        
        super().__init__()
        self.version = 'interp_12Hz_trainval'
        self.dataset_prefix = "/datasets/nuScenes/"
        self.path = path
        self.split = split
        self.dataset_idx = 0

        self.cameras = cameras
        self.scales = np.arange(scale_range+2) 
        self.num_cameras = len(cameras)

        self.bwd = back_context
        self.fwd = forward_context
        
        self.has_context = back_context + forward_context > 0
        self.data_transform = data_transform

        self.with_depth = depth_type is not None
        self.with_pose = with_pose
        self.with_ego_pose = with_ego_pose

        self.loader = img_loader

        self.with_mask = with_mask
        cur_path = os.path.dirname(os.path.realpath(__file__))        
        self.mask_path = os.path.join(cur_path, 'nuscenes_mask')
        self.mask_loader = mask_loader_scene

        # load OmniScene dataset bin tokens
        if split == "train":
            #for training
            self.bin_tokens = json.load(open(osp.join(self.path, self.version, "bins_train_3.2m.json")))["bins"]
        elif split == "eval_SF" or split == "eval_OS" or split == "eval_MF":
            # for evaluation
            self.bin_tokens = json.load(open(osp.join(self.path, self.version, "bins_val_3.2m.json")))["bins"]
            self.bin_tokens = self.bin_tokens[0::14][:2048]  # 每隔 14 个取一个，取 2048 个
        else:
            raise NotImplementedError("Invalid split: {}".format(self.split))

    def get_current(self, key, img_path):
        """
        This function returns samples for current contexts
        """        
        # get current timestamp rgb sample
        if key == 'rgb':
            rgb_path = img_path.replace("samples", "samples_small")
            rgb_path = rgb_path.replace("sweeps", "sweeps_small")
            return self.loader(rgb_path)
        # get current timestamp camera intrinsics
        elif key == 'intrinsics':
            param_path = img_path.replace("samples", "samples_param_small") # 224x400 resolution
            param_path = param_path.replace("sweeps", "sweeps_param_small")
            param_path = param_path.replace(".jpg", ".json")
            param = json.load(open(param_path))
            return np.array(param["camera_intrinsic"], dtype=np.float32)
        # get current timestamp metric depth map
        elif key == 'depth':
            depthm_path = img_path.replace("sweeps", "sweeps_dptm_small")
            depthm_path = depthm_path.replace("samples", "samples_dptm_small")
            depthm_path = depthm_path.replace(".jpg", "_dpt.npy")
            dptm = np.load(depthm_path).astype(np.float32)
            return dptm
        else:
            raise ValueError('Unknown key: ' +key)

    def get_context(self, key, cam, sensor_info_fwd, sensor_info_bwd):
        """
        This function returns samples for backward and forward contexts
        """
        bwd_context, fwd_context = [], []
        if self.bwd != 0:
            info = copy.deepcopy(sensor_info_bwd[cam])
            img_path, c2w_bwd, _ = load_info(info)
            bwd_path = img_path.replace(self.dataset_prefix, self.path)
            bwd_context = [self.get_current(key, bwd_path)]

        if self.fwd != 0:
            info = copy.deepcopy(sensor_info_fwd[cam])
            img_path, c2w_fwd, _ = load_info(info)
            fwd_path = img_path.replace(self.dataset_prefix, self.path)
            fwd_context = [self.get_current(key, fwd_path)]
        
        return bwd_context + fwd_context, c2w_bwd, c2w_fwd
    
    def get_cam_T_cam(self, c2w_center, c2w_bwd, c2w_fwd):
        cam_T_cam = []

        # cam_T_cam, 0, -1
        if self.bwd != 0:
            cam_T_cam_bwd = np.linalg.inv(c2w_bwd) @ c2w_center
            cam_T_cam.append(cam_T_cam_bwd)

        # cam_T_cam, 0, 1
        if self.fwd != 0:
            cam_T_cam_fwd = np.linalg.inv(c2w_fwd) @ c2w_center
            cam_T_cam.append(cam_T_cam_fwd)

        return cam_T_cam


    def __len__(self):
        return len(self.bin_tokens)
    
    def __getitem__(self, idx):
        bin_token = self.bin_tokens[idx]
        with open(osp.join(self.path, self.version, "bin_infos_3.2m", bin_token + ".pkl"), "rb") as f:
            bin_info = pkl.load(f)
        sensor_info_center = {sensor: bin_info["sensor_info"][sensor][0] for sensor in self.cameras + ["LIDAR_TOP"]}
        sensor_info_bwd = {sensor: bin_info["sensor_info"][sensor][1] for sensor in self.cameras + ["LIDAR_TOP"]}
        sensor_info_fwd = {sensor: bin_info["sensor_info"][sensor][2] for sensor in self.cameras + ["LIDAR_TOP"]}
        
        sample = []
        contexts = []
        if self.bwd:
            contexts.append(-1)
        if self.fwd:
            contexts.append(1)

        # loop over all cameras            
        for cam in self.cameras:
            # 输入视角
            info = copy.deepcopy(sensor_info_center[cam])
            img_path, c2w_center, _ = load_info(info)
            img_path = img_path.replace(self.dataset_prefix, self.path)
            
            data = {
                'idx': idx,
                'token': bin_token,
                'sensor_name': cam,
                'contexts': contexts,
                'filename': img_path,
                'rgb': self.get_current('rgb', img_path),
                'intrinsics': self.get_current('intrinsics', img_path)
            }

            # if depth is returned            
            if self.with_depth:
                data.update({
                    'depth': self.get_current('depth', img_path)  # 尺度深度
                })
            # if pose is returned
            if self.with_pose:
                data.update({
                    'extrinsics': c2w_center  # cam to lidar
                })
            # if mask is returned
            if self.with_mask:
                data.update({
                    'mask': self.mask_loader(self.mask_path, '', cam)  # 自车掩码
                })  
                  
            # 输出视角
            context, c2w_bwd, c2w_fwd = self.get_context('rgb', cam, sensor_info_fwd, sensor_info_bwd)
            # if context is returned
            if self.has_context:
                data.update({
                    'rgb_context': context
                })
            # if ego_pose is returned
            if self.with_ego_pose:
                data.update({
                    'ego_pose': self.get_cam_T_cam(c2w_center, c2w_bwd, c2w_fwd)  # [cam_T_cam_bwd, cam_T_cam_fwd]，当前相机到过去/未来相机的变换
                })

            sample.append(data)

        # apply same data transformations for all sensors
        if self.data_transform:
            sample = [self.data_transform(smp) for smp in sample]
            sample = [transform_mask_sample(smp, self.data_transform) for smp in sample]

        # stack and align dataset for our trainer
        sample = stack_sample(sample)
        sample = align_dataset(sample, self.scales, contexts)

        # 添加 ('cam_T_cam', 0, 0) 到 sample，以使用当前时刻图像进行监督和评估
        sample[('cam_T_cam', 0, 0)] = np.expand_dims(np.eye(4), 0).repeat(self.num_cameras, axis=0)

        return sample
                