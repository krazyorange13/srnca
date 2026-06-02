import os
import sys
import math
import time
from dataclasses import dataclass

import cv2
import numpy as np
import visdom

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from torchvision.models import vgg19, VGG19_Weights
from torchvision.transforms import Normalize

from tqdm import tqdm


class VGGFeatureExtractor(nn.Module):
    def __init__(self, slice=16):
        super().__init__()
        self.slice = slice

        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.feature_extractor = vgg[:slice]  # type: ignore # :9, :16, :23, :36

        for param in self.parameters():
            param.requires_grad = False

        # imagenet normalization
        self.normalize = Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.eval()

    def forward(self, x):
        # (b, c, h, w) bgr to rgb
        x = x[:, [2, 1, 0], :, :]

        if x.shape[2] < 224 or x.shape[3] < 224:
            x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)

        x = self.normalize(x)

        return self.feature_extractor(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        spectral_norm = nn.utils.parametrizations.spectral_norm
        self.seq = nn.Sequential(
            spectral_norm(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.seq(x)


class Sampler:
    def __init__(self, x_dir, y_dir, limit=100, channels=16):
        self.x_paths = [os.path.join(x_dir, f) for f in sorted(os.listdir(x_dir))[:limit]]
        self.y_paths = [os.path.join(y_dir, f) for f in sorted(os.listdir(y_dir))[:limit]]
        self.channels = channels
        self.xs = self.load_images(self.x_paths)
        self.ys = self.load_images(self.y_paths)

    def load_images(self, img_paths):
        imgs = []
        for path in img_paths:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            imgs.append(img_t)
        imgs_t = torch.stack(imgs, dim=0)
        return imgs_t

    def sample(self, batch_size=8, crop_size=64):
        self.idxs = torch.randperm(len(self.xs))[:batch_size]
        self.x_batch = self.xs[self.idxs]
        self.y_batch = self.ys[self.idxs]

        # hidden layers
        _b, _c, _h, _w = self.x_batch.shape
        _c_ = self.channels - _c
        hid = torch.zeros(_b, _c_, _h, _w)
        self.x_batch = torch.cat([self.x_batch, hid], dim=1)

        # upscale
        self.x_batch = F.interpolate(self.x_batch, scale_factor=2, mode="nearest")

        # crop
        top = torch.randint(0, _h - crop_size + 1, (1,)).item()
        left = torch.randint(0, _w - crop_size + 1, (1,)).item()
        self.x_batch = self.x_batch[:, :, top : top + crop_size, left : left + crop_size]
        self.y_batch = self.y_batch[:, :, top : top + crop_size, left : left + crop_size]

        # print(self.x_batch.shape, self.y_batch.shape)
        return self.x_batch, self.y_batch


class PerceptionFilter(nn.Module):
    def __init__(self, in_channels):
        super(PerceptionFilter, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 4

        self.conv = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.reset_params()

    def reset_params(self):
        identity = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        grad_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
        grad_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]) / 8.0
        laplacian = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])

        kernel = torch.stack([identity, grad_x, grad_y, laplacian])[:, None, :]
        with torch.no_grad():
            self.conv.weight.copy_(kernel.repeat(self.in_channels, 1, 1, 1))

    def forward(self, x):
        return self.conv(x)


class NCA(nn.Module):
    def __init__(self, channels, update_rate=0.5):
        super(NCA, self).__init__()
        self.channels = channels
        self.update_rate = update_rate

        self.perception = PerceptionFilter(self.channels)

        self.seq = nn.Sequential(
            nn.Conv2d(self.perception.out_channels, 256, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, self.channels, kernel_size=1, bias=False),
        )

    def step(self, x, update_rate=None):
        y = self.perception(x)
        y = self.seq(y)

        update_mask = self.get_update_mask(y.shape, update_rate)
        x = x + y * update_mask

        return x

    def get_update_mask(self, shape, update_rate=None):
        b, _, h, w = shape
        update_rate = update_rate or self.update_rate
        update_mask = (torch.rand(b, 1, h, w) < update_rate).float()
        return update_mask

    def forward_chunk(self, x, steps, update_rate):
        for i in range(steps):
            # tqdm.write(f"\t\t\tnca step {i}")
            x = self.step(x, update_rate=update_rate)
        return x

    def forward(self, x, steps, update_rate=None):
        x = self.forward_chunk(x, steps, update_rate)
        # chunk_size = 8
        # for i in range(0, steps, chunk_size):
        #     tqdm.write(f"\t\tnca chunk {i}")
        #     x = checkpoint(self.forward_chunk, x, chunk_size, update_rate, use_reentrant=False)
        return x


@dataclass
class SRNCAConfig:
    model_name: str
    model_dir: str
    hr_dir: str
    lr_dir: str
    img_limit: int
    epochs: int
    batch_size: int
    crop_size: int
    nca_steps: tuple[int, int]
    nca_channels: int = 16
    nca_update_rate: float = 0.5
    nca_optim_lr: float = 1e-4
    nca_optim_lr_gamma: float = 0.99995
    nca_optim_weight_decay: float = 1e-4
    nca_optim_betas: tuple[float, float] = (0.0, 0.999)
    gan_optim_lr: float = 2e-4
    gan_optim_lr_gamma: float = 0.99995
    gan_optim_weight_decay: float = 0.0
    gan_optim_betas: tuple[float, float] = (0.0, 0.999)
    vgg_slice: int = 16
    gan_start: int = 400
    lambda_pxl: float = 1.0
    lambda_vgg: float = 1.0
    lambda_gan: float = 1e-3


class SRNCA:
    """Super Resolution Neural Cellular Automata"""

    def __init__(self, config: SRNCAConfig, state: None, vis: visdom.Visdom | None):
        self.config = config

        self.sampler = Sampler(self.config.lr_dir, self.config.hr_dir, self.config.img_limit, self.config.nca_channels)
        self.nca = NCA(channels=self.config.nca_channels)
        self.vgg = VGGFeatureExtractor(self.config.vgg_slice)
        self.gan = Discriminator()
        self.nca_optimizer = optim.AdamW(self.nca.parameters(), lr=self.config.nca_optim_lr, weight_decay=self.config.nca_optim_weight_decay, betas=self.config.nca_optim_betas)
        self.gan_optimizer = optim.AdamW(self.gan.parameters(), lr=self.config.gan_optim_lr, weight_decay=self.config.gan_optim_weight_decay, betas=self.config.gan_optim_betas)
        self.nca_scheduler = optim.lr_scheduler.ExponentialLR(self.nca_optimizer, self.config.nca_optim_lr_gamma)
        self.gan_scheduler = optim.lr_scheduler.ExponentialLR(self.gan_optimizer, self.config.gan_optim_lr_gamma)

        self.pxl_criterion = nn.L1Loss()
        self.vgg_criterion = nn.MSELoss()
        self.gan_criterion = nn.BCEWithLogitsLoss()

        self.min_loss = float("inf")
        self.min_model = ""
        self.last_model = ""
        self.curr_epoch = 0
        self.loaded_epoch = 0
        self.nca_acc_loss = 0
        self.gan_acc_loss = 0
        self.acc_epochs = 0

        if state is not None:
            self.nca.load_state_dict(state["nca"])
            self.vgg.load_state_dict(state["vgg"])
            self.gan.load_state_dict(state["gan"])
            self.nca_optimizer.load_state_dict(state["nca_optimizer"])
            self.gan_optimizer.load_state_dict(state["gan_optimizer"])
            self.nca_scheduler.load_state_dict(state["nca_scheduler"])
            self.gan_scheduler.load_state_dict(state["gan_scheduler"])
            self.min_loss = state["min_loss"]
            self.loaded_epoch = state["epoch"]

    def get_nca_steps_random(self):
        return torch.randint(self.config.nca_steps[0], self.config.nca_steps[1], [1]).item()

    def get_nca_steps_avg(self):
        return (self.config.nca_steps[0] + self.config.nca_steps[1]) // 2

    def train(self):
        print(f"model: {self.config.model_name}")

        try:
            # with torch.profiler.profile(
            #     activities=[torch.profiler.ProfilerActivity.CPU],
            #     profile_memory=True,
            # ) as prof:
            self.loop()
            # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
        except KeyboardInterrupt:
            print("training cancelled")

        print(f"model saved: {self.last_model}")
        if self.min_model:
            print(f"best model: {self.min_model}")

    def loop(self):
        for i in tqdm(
            range(self.loaded_epoch, self.config.epochs),
            initial=self.loaded_epoch,
            total=self.config.epochs,
            leave=False,
            dynamic_ncols=True,
        ):
            self.curr_epoch = i

            # nca (generator)

            # tqdm.write(f"{i}\tgenerate NCA")
            x, hr_imgs = self.sampler.sample(self.config.batch_size, self.config.crop_size)
            steps = self.get_nca_steps_random()
            x = self.nca(x, steps)
            sr_imgs = x[:, :3, :, :]

            # optimize gan (discriminator)

            if self.curr_epoch >= self.config.gan_start:
                # tqdm.write(f"{i}\toptimize GAN")
                self.gan_optimizer.zero_grad()
                hr_preds = self.gan(hr_imgs)
                sr_preds = self.gan(sr_imgs.detach())
                # hr_loss = self.gan_criterion(hr_preds, torch.ones_like(hr_preds))
                # sr_loss = self.gan_criterion(sr_preds, torch.zeros_like(sr_preds))
                # gan_loss = (hr_loss + sr_loss) / 2
                hr_loss = torch.mean(torch.relu(1.0 - hr_preds))
                sr_loss = torch.mean(torch.relu(1.0 + sr_preds))
                gan_loss = hr_loss + sr_loss
                gan_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.gan.parameters(), max_norm=1.0)
                self.gan_optimizer.step()
                self.gan_scheduler.step()
            else:
                gan_loss = torch.tensor([0])

            # optimize nca

            self.nca_optimizer.zero_grad()

            if self.curr_epoch < self.config.gan_start:
                # tqdm.write(f"{i}\toptimize NCA (L2)")
                nca_loss = self.pxl_criterion(sr_imgs, hr_imgs)
                nca_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nca.parameters(), max_norm=1.0)
                self.nca_optimizer.step()
                self.nca_scheduler.step()
            else:
                # tqdm.write(f"{i}\toptimize NCA (VGG + GAN)")
                pxl_loss = self.pxl_criterion(sr_imgs, hr_imgs)
                sr_preds_for_nca = self.gan(sr_imgs)
                # gan_loss_for_nca = self.gan_criterion(sr_preds_for_nca, torch.ones_like(sr_preds_for_nca))
                gan_loss_for_nca = -torch.mean(sr_preds_for_nca)
                hr_vgg = self.vgg(hr_imgs)
                sr_vgg = self.vgg(sr_imgs)
                vgg_loss = self.vgg_criterion(sr_vgg, hr_vgg)
                nca_loss = (pxl_loss * self.config.lambda_pxl) + (vgg_loss * self.config.lambda_vgg) + (gan_loss_for_nca * self.config.lambda_gan)
                nca_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.nca.parameters(), max_norm=1.0)
                self.nca_optimizer.step()
                self.nca_scheduler.step()

            # tracking

            self.gan_acc_loss += gan_loss.item()
            self.nca_acc_loss += nca_loss.item()
            self.acc_epochs += 1

            # started VGG + GAN training, reset loss baseline
            if self.curr_epoch == self.config.gan_start:
                self.min_loss = float("inf")
                self.min_model = ""
                self.last_model = ""

            if vis:
                vis.line(
                    X=[self.curr_epoch],
                    Y=[nca_loss.item()],
                    win="nca_loss",
                    update="append" if self.curr_epoch > 0 else None,
                )
                vis.line(
                    X=[self.curr_epoch],
                    Y=[gan_loss.item()],
                    win="gan_loss",
                    update="append" if self.curr_epoch > 0 else None,
                )

            if self.curr_epoch % 100 == 99:
                nca_avg_loss = self.nca_acc_loss / self.acc_epochs
                gan_avg_loss = self.gan_acc_loss / self.acc_epochs
                tqdm.write(f"epoch {self.curr_epoch + 1} nca loss: {nca_avg_loss} gan loss: {gan_avg_loss}")

                if nca_avg_loss < self.min_loss:
                    if self.min_model and self.min_model != self.last_model:
                        os.remove(self.min_model)
                    self.min_loss = nca_avg_loss
                    self.min_model = self.save()
                if not math.isnan(nca_avg_loss):
                    if self.last_model and self.min_model != self.last_model:
                        os.remove(self.last_model)
                    self.last_model = self.save()

                self.gan_acc_loss = 0
                self.nca_acc_loss = 0
                self.acc_epochs = 0

    def save(self):
        state = {
            "config": self.config,
            "nca": self.nca.state_dict(),
            "vgg": self.vgg.state_dict(),
            "gan": self.gan.state_dict(),
            "nca_optimizer": self.nca_optimizer.state_dict(),
            "gan_optimizer": self.gan_optimizer.state_dict(),
            "nca_scheduler": self.nca_scheduler.state_dict(),
            "gan_scheduler": self.gan_scheduler.state_dict(),
            "min_loss": self.min_loss,
            "epoch": self.curr_epoch,
        }
        save_path = f"{self.config.model_dir}/{self.config.model_name}-{self.curr_epoch + 1}.tar"
        torch.save(state, save_path)
        return save_path


if __name__ == "__main__":
    try:
        vis = visdom.Visdom(port=8097)
        if not vis.check_connection():
            vis = None
    except:
        vis = None

    if len(sys.argv) in [2, 3]:
        state = torch.load(sys.argv[1], weights_only=False)
        config = state["config"]

        if len(sys.argv) == 3:
            new_epochs = int(sys.argv[2])
            config.epochs = new_epochs

        srnca = SRNCA(config, state, vis)

    else:
        config = SRNCAConfig(
            model_name="zeta",
            model_dir="models",
            hr_dir="data/hr",
            lr_dir="data/lr",
            img_limit=1000,
            epochs=10000,
            batch_size=8,
            crop_size=80,
            nca_steps=(48, 64),
            nca_channels=12,
            nca_update_rate=0.5,
            nca_optim_lr=1e-4,
            nca_optim_lr_gamma=0.99995,
            nca_optim_weight_decay=1e-4,
            nca_optim_betas=(0.0, 0.999),
            gan_optim_lr=2e-4,
            gan_optim_lr_gamma=0.99995,
            gan_optim_weight_decay=0.0,
            gan_optim_betas=(0.0, 0.999),
            vgg_slice=16,
            gan_start=400,
            lambda_pxl=1.0,
            lambda_vgg=1.0,
            lambda_gan=1e-3,
        )
        srnca = SRNCA(config, None, vis)

    srnca.train()
