import time
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch import Tensor

from utils import Logger

from lpips import LPIPS
from jaxtyping import Float, UInt8
from skimage.metrics import structural_similarity
from einops import reduce

from PIL import Image
from pathlib import Path
from einops import rearrange, repeat
from typing import Union
import numpy as np

from tqdm import tqdm

FloatImage = Union[
    Float[Tensor, "height width"],
    Float[Tensor, "channel height width"],
    Float[Tensor, "batch channel height width"],
]

class DrivingForwardTrainer:
    """
    Trainer class for training and evaluation
    """
    def __init__(self, cfg, rank, use_tb=True):
        self.read_config(cfg)
        self.rank = rank        
        if rank == 0:
            self.logger = Logger(cfg, use_tb)
            self.depth_metric_names = self.logger.get_metric_names()

        self.lpips = LPIPS(net="vgg").cuda(rank)

    def read_config(self, cfg):
        for attr in cfg.keys(): 
            for k, v in cfg[attr].items():
                setattr(self, k, v)

    def learn(self, model):
        """
        This function sets training process.
        """        
        train_dataloader = model.train_dataloader()
        if self.rank == 0:
            val_dataloader = model.val_dataloader()
            self.val_iter = iter(val_dataloader)
        
        self.step = 0
        self.start_epoch = 0

        assert self.train_target in ['epoch', 'global_iter'], f'train_target {self.train_target} should be one of [epoch, global_iter]'
        
        # load model
        if model.load_weights_dir is not None:  # 断点续训
            model.load_weights()
            if self.train_target == 'epoch':  # 以达成若干轮数为训练目标
                self.start_epoch = int(model.load_weights_dir.split('_')[-2])  # 已完成训练的 epoch 数
                self.start_epoch = self.start_epoch + 1  # 从下一个 epoch 开始训练
                self.step = self.start_epoch * len(train_dataloader)  # 已完成训练的 step 数
            elif self.train_target == 'global_iter':  # 以达成若干迭代次数为训练目标
                self.step = int(model.load_weights_dir.split('_')[-1])  # 已完成训练的 step 数
        
        start_time = time.time()
        for self.epoch in range(self.start_epoch, self.num_epochs):
                
            self.train(model, train_dataloader, start_time)  # 训练一轮 epoch
            
            # 若以达成若干轮数为训练目标，每结束一轮 epoch 保存一次
            if self.train_target == 'epoch' and self.rank == 0:
                model.save_model(self.epoch, self.step)
                print(f'Save model at epoch {self.epoch} at step {self.step} !')

            # 若以达成若干迭代次数为训练目标，达成后直接退出
            if self.train_target == 'global_iter' and self.step > self.num_global_iters:
                break

            if self.ddp_enable:
                dist.barrier()
                
        if self.rank == 0:
            self.logger.close_tb()
        
    def train(self, model, data_loader, start_time):
        """
        This function trains models.
        """
        model.set_train()
        pbar = tqdm(total=len(data_loader), desc='training on epoch {}'.format(self.epoch), mininterval=100)
        for batch_idx, inputs in enumerate(data_loader):         
            before_op_time = time.time()
            model.optimizer.zero_grad(set_to_none=True)
            outputs, losses = model.process_batch(inputs, self.rank)
            losses['total_loss'].backward()
            model.optimizer.step()

            if self.rank == 0: 
                self.logger.update('train', self.epoch, self.world_size, batch_idx, self.step, start_time, before_op_time, inputs, outputs, losses)

                # 若以达成若干迭代次数为训练目标，每 10k 次迭代保存一次模型
                if self.train_target == 'global_iter' and self.step % 10000 == 0:
                    model.save_model(self.epoch, self.step)
                    print(f'Save model at epoch {self.epoch} at step {self.step} !')

                # 若以达成若干迭代次数为训练目标，达成后直接退出
                if self.train_target == 'global_iter' and self.step > self.num_global_iters:
                    break
                
                if self.logger.is_checkpoint(self.step):
                    self.validate(model)

            self.step += 1
            pbar.update(1)

        pbar.close()
        model.lr_scheduler.step()
        
    @torch.no_grad()
    def validate(self, model, vis_results=False):
        """
        This function validates models on the validation dataset to monitor training process.
        """
        val_dataloader = model.val_dataloader()
        val_iter = iter(val_dataloader)
        
        # Ensure the model is in validation mode
        model.set_val()

        avg_reconstruction_metric = defaultdict(float)

        inputs = next(val_iter)
        outputs, _ = model.process_batch(inputs, self.rank)
            
        psnr, ssim, lpips= self.compute_reconstruction_metrics(inputs, outputs)

        avg_reconstruction_metric['psnr'] += psnr   
        avg_reconstruction_metric['ssim'] += ssim
        avg_reconstruction_metric['lpips'] += lpips

        print('Validation reconstruction result...')
        print(f"{inputs['token'][0]}")
        self.logger.print_perf(avg_reconstruction_metric, 'reconstruction')
        print('')

        # Set the model back to training mode
        model.set_train()

    @torch.no_grad()
    def evaluate(self, model):
        """
        This function evaluates models on validation dataset of samples with context.
        """
        eval_dataloader = model.eval_dataloader()

        # load model
        model.load_weights()
        model.set_eval()

        avg_reconstruction_metric = defaultdict(float)

        count = 0

        process = tqdm(eval_dataloader)
        for batch_idx, inputs in enumerate(process):
            outputs, _ = model.process_batch(inputs, self.rank)
            
            psnr, ssim, lpips= self.compute_reconstruction_metrics(inputs, outputs)

            avg_reconstruction_metric['psnr'] += psnr   
            avg_reconstruction_metric['ssim'] += ssim
            avg_reconstruction_metric['lpips'] += lpips
            count += 1

            process.set_description(f"PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, LPIPS: {lpips:.4f}")

            print(f"\n{inputs['token'][0]}")
            print(f"avg PSNR: {avg_reconstruction_metric['psnr']/count:.4f}, avg SSIM: {avg_reconstruction_metric['ssim']/count:.4f}, avg LPIPS: {avg_reconstruction_metric['lpips']/count:.4f}")
            
        avg_reconstruction_metric['psnr'] /= len(eval_dataloader)
        avg_reconstruction_metric['ssim'] /= len(eval_dataloader)
        avg_reconstruction_metric['lpips'] /= len(eval_dataloader)

        print('Evaluation reconstruction result...\n')
        self.logger.print_perf(avg_reconstruction_metric, 'reconstruction')

    def save_image(
        self,
        image: FloatImage,
        path: Union[Path, str],
    ) -> None:
        """Save an image. Assumed to be in range 0-1."""

        # Create the parent directory if it doesn't already exist.
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)

        # Save the image.
        Image.fromarray(self.prep_image(image)).save(path)

    def save_ply(self, outputs, path, compatible=True):
        # compatible: save pre-activated gaussians as in the original paper

        # 从 outputs 中提取高斯
        for i in range(self.eval_batch_size):
            xyz_i_valid = []
            # rgb_i_valid = []
            rot_i_valid = []
            scale_i_valid = []
            opacity_i_valid = []
            sh_i_valid = []
            if self.novel_view_mode == 'SF' or self.novel_view_mode == 'OS':
                for frame_id in [0]:
                    for cam in range(self.num_cams):
                        valid_i = outputs[('cam', cam)][('pts_valid', frame_id, 0)][i, :]
                        xyz_i = outputs[('cam', cam)][('xyz', frame_id, 0)][i, :, :]
                        # rgb_i = inputs[('color', frame_id, 0)][:, cam, ...][i, :, :, :].permute(1, 2, 0).view(-1, 3) # HWC
                        
                        rot_i = outputs[('cam', cam)][('rot_maps', frame_id, 0)][i, :, :, :].permute(1, 2, 0).view(-1, 4)
                        scale_i = outputs[('cam', cam)][('scale_maps', frame_id, 0)][i, :, :, :].permute(1, 2, 0).view(-1, 3)
                        opacity_i = outputs[('cam', cam)][('opacity_maps', frame_id, 0)][i, :, :, :].permute(1, 2, 0).view(-1, 1)
                        sh_i = rearrange(outputs[('cam', cam)][('sh_maps', frame_id, 0)][i, :, :, :], "p srf r xyz d_sh -> (p srf r) d_sh xyz").contiguous()

                        xyz_i_valid.append(xyz_i[valid_i].view(-1, 3))
                        # rgb_i_valid.append(rgb_i[valid_i].view(-1, 3))
                        rot_i_valid.append(rot_i[valid_i].view(-1, 4))
                        scale_i_valid.append(scale_i[valid_i].view(-1, 3))
                        opacity_i_valid.append(opacity_i[valid_i].view(-1, 1))
                        sh_i_valid.append(sh_i[valid_i])

            elif self.novel_view_mode == 'MF':
                for frame_id in [-1, 1]:
                    for cam in range(self.num_cams):
                        valid_i = outputs[('cam', cam)][('pts_valid', frame_id, 0)][i, :]
                        xyz_i = outputs[('cam', cam)][('xyz', frame_id, 0)][i, :, :]
                        # rgb_i = inputs[('color', frame_id, 0)][:, cam, ...][i, :, :, :].permute(1, 2, 0).view(-1, 3) # HWC
                            
                        rot_i = outputs[('cam', cam)][('rot_maps', frame_id, 0)][i, :, :, :].permute(1, 2, 0).view(-1, 4)
                        scale_i = outputs[('cam', cam)][('scale_maps', frame_id, 0)][i, :, :, :].permute(1, 2, 0).view(-1, 3)
                        opacity_i = outputs[('cam', cam)][('opacity_maps', frame_id, 0)][i, :, :, :].permute(1, 2, 0).view(-1, 1)
                        sh_i = rearrange(outputs[('cam', cam)][('sh_maps', frame_id, 0)][i, :, :, :], "p srf r xyz d_sh -> (p srf r) d_sh xyz").contiguous()

                        xyz_i_valid.append(xyz_i[valid_i].view(-1, 3))
                        # rgb_i_valid.append(rgb_i[valid_i].view(-1, 3))
                        rot_i_valid.append(rot_i[valid_i].view(-1, 4))
                        scale_i_valid.append(scale_i[valid_i].view(-1, 3))
                        opacity_i_valid.append(opacity_i[valid_i].view(-1, 1))
                        sh_i_valid.append(sh_i[valid_i])

            pts_xyz_i = torch.concat(xyz_i_valid, dim=0)
            # pts_rgb_i = torch.concat(rgb_i_valid, dim=0)
            # pts_rgb_i = pts_rgb_i * 0.5 + 0.5
            rot_i = torch.concat(rot_i_valid, dim=0)
            scale_i = torch.concat(scale_i_valid, dim=0)
            opacity_i = torch.concat(opacity_i_valid, dim=0)
            sh_i = torch.concat(sh_i_valid, dim=0)

        from plyfile import PlyData, PlyElement
     
        means3D = pts_xyz_i.contiguous().float()
        opacity = opacity_i.contiguous().float()
        scales = scale_i.contiguous().float()
        rotations = rot_i.contiguous().float()
        shs = sh_i.unsqueeze(1).contiguous().float() # [N, 1, 3]

        # prune by opacity
        mask = opacity.squeeze(-1) >= 0.005
        means3D = means3D[mask]
        opacity = opacity[mask]
        scales = scales[mask]
        rotations = rotations[mask]
        shs = shs[mask]

        # invert activation to make it compatible with the original ply format
        if compatible:
            def inverse_sigmoid(x):
                return torch.log(x/(1-x))
            opacity = inverse_sigmoid(opacity)
            scales = torch.log(scales + 1e-8)
            shs = (shs - 0.5) / 0.28209479177387814

        xyzs = means3D.detach().cpu().numpy()
        f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = opacity.detach().cpu().numpy()
        scales = scales.detach().cpu().numpy()
        rotations = rotations.detach().cpu().numpy()

        l = ['x', 'y', 'z']
        # All channels except the 3 DC
        for i in range(f_dc.shape[1]):
            l.append('f_dc_{}'.format(i))
        l.append('opacity')
        for i in range(scales.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(rotations.shape[1]):
            l.append('rot_{}'.format(i))

        dtype_full = [(attribute, 'f4') for attribute in l]

        elements = np.empty(xyzs.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)


    def prep_image(self, image: FloatImage) -> UInt8[np.ndarray, "height width channel"]:
        # Handle batched images.
        if image.ndim == 4:
            image = rearrange(image, "b c h w -> c h (b w)")

        # Handle single-channel images.
        if image.ndim == 2:
            image = rearrange(image, "h w -> () h w")

        # Ensure that there are 3 or 4 channels.
        channel, _, _ = image.shape
        if channel == 1:
            image = repeat(image, "() h w -> c h w", c=3)
        assert image.shape[0] in (3, 4)

        image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
        return rearrange(image, "c h w -> h w c").cpu().numpy()

    @torch.no_grad()
    def compute_reconstruction_metrics(self, inputs, outputs):
        """
        This function computes reconstruction metrics.
        """
        psnr = 0.0
        ssim = 0.0
        lpips = 0.0
        if self.novel_view_mode == 'SF':
            frame_ids = [1]
        elif self.novel_view_mode == 'MF':
            frame_ids = [0]
        elif self.novel_view_mode == 'OS':
            frame_ids = [0, -1, 1]
        else:
            raise ValueError(f"Invalid novel view mode: {self.novel_view_mode}")
        for cam in range(self.num_cams):
            for frame_id in frame_ids:
                rgb_gt = inputs[('color', frame_id, 0)][:, cam, ...]
                image = outputs[('cam', cam)][('gaussian_color', frame_id, 0)]
                psnr += self.compute_psnr(rgb_gt, image).mean()
                ssim += self.compute_ssim(rgb_gt, image).mean()
                lpips += self.compute_lpips(rgb_gt, image).mean()
                if self.save_images:
                    assert self.eval_batch_size == 1
                    if self.novel_view_mode == 'SF':
                        self.save_image(image, Path(self.save_path) / inputs['token'][0] / f"{cam}.png")
                        self.save_image(rgb_gt, Path(self.save_path) / inputs['token'][0] / f"{cam}_gt.png")
                        self.save_image(inputs[('color', 0, 0)][:, cam, ...], Path(self.save_path) / inputs['token'][0] / f"{cam}_0_gt.png")
                    elif self.novel_view_mode == 'MF':
                        self.save_image(image, Path(self.save_path) / inputs['token'][0] / f"{cam}.png")
                        self.save_image(rgb_gt, Path(self.save_path) / inputs['token'][0] / f"{cam}_gt.png")
                        self.save_image(inputs[('color', -1, 0)][:, cam, ...], Path(self.save_path) / inputs['token'][0] / f"{cam}_prev_gt.png")
                        self.save_image(inputs[('color', 1, 0)][:, cam, ...], Path(self.save_path) / inputs['token'][0] / f"{cam}_next_gt.png")
                    elif self.novel_view_mode == 'OS':
                        self.save_image(image, Path(self.save_path) / inputs['token'][0] / f"{cam}_{frame_id}.png")
                        self.save_image(rgb_gt, Path(self.save_path) / inputs['token'][0] / f"{cam}_{frame_id}_gt.png")
                        self.save_image(inputs[('color', 0, 0)][:, cam, ...], Path(self.save_path) / inputs['token'][0] / f"{cam}_{frame_id}_0_gt.png")
        
        if self.save_plys:
            assert self.eval_batch_size == 1
            self.save_ply(outputs, Path(self.save_path) / inputs['token'][0] / "driving_forward.ply")
        
        psnr /= (self.num_cams * len(frame_ids))
        ssim /= (self.num_cams * len(frame_ids))
        lpips /= (self.num_cams * len(frame_ids))
        return psnr, ssim, lpips
    
    @torch.no_grad()
    def compute_psnr(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, " batch"]:
        ground_truth = ground_truth.clip(min=0, max=1)
        predicted = predicted.clip(min=0, max=1)
        mse = reduce((ground_truth - predicted) ** 2, "b c h w -> b", "mean")
        return -10 * mse.log10()
    
    @torch.no_grad()
    def compute_lpips(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, " batch"]:
        value = self.lpips.forward(ground_truth, predicted, normalize=True)
        return value[:, 0, 0, 0]
    
    @torch.no_grad()
    def compute_ssim(
        self,
        ground_truth: Float[Tensor, "batch channel height width"],
        predicted: Float[Tensor, "batch channel height width"],
    ) -> Float[Tensor, " batch"]:
        ssim = [
            structural_similarity(
                gt.detach().cpu().numpy(),
                hat.detach().cpu().numpy(),
                win_size=11,
                gaussian_weights=True,
                channel_axis=0,
                data_range=1.0,
            )
            for gt, hat in zip(ground_truth, predicted)
        ]
        return torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)
