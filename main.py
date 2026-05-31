import os
import sys
import math
import time
from dataclasses import dataclass
import tracemalloc

import cv2
import numpy as np
import visdom

import torch
from torch import optim
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from tqdm import tqdm


tracemalloc.start()


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
        self.x_batch = F.interpolate(self.x_batch, scale_factor=2, mode="bilinear")

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
    nca_channels: int = 16
    optim_lr: float = 1e-3
    optim_weight_decay: float = 0.01
    optim_lr_gamma: float = 0.999


class SRNCA:
    """Super Resolution Neural Cellular Automata"""

    def __init__(self, config: SRNCAConfig, state: None, vis: visdom.Visdom | None):
        self.config = config

        self.sampler = Sampler(self.config.lr_dir, self.config.hr_dir, self.config.img_limit, self.config.nca_channels)
        self.nca = NCA(channels=self.config.nca_channels)
        self.optimizer = optim.AdamW(self.nca.parameters(), lr=self.config.optim_lr, weight_decay=self.config.optim_weight_decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, self.config.optim_lr_gamma)

        self.min_loss = float("inf")
        self.min_model = ""
        self.last_model = ""
        self.curr_epoch = 0
        self.loaded_epoch = 0
        self.acc_loss = 0
        self.acc_epochs = 0

        if state is not None:
            self.nca.load_state_dict(state["nca"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
            self.min_loss = state["min_loss"]
            self.loaded_epoch = state["epoch"]

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
            range(0, self.config.epochs),
            initial=self.loaded_epoch,
            total=self.config.epochs,
            leave=False,
            dynamic_ncols=True,
        ):
            self.curr_epoch = i
            self.optimizer.zero_grad()

            x, y = self.sampler.sample(self.config.batch_size, self.config.crop_size)

            steps = torch.randint(8, 16, [1]).item()
            # tqdm.write(f"{i}\trun nca")
            x = self.nca(x, steps)
            loss = F.mse_loss(x[:, :3, :, :], y)
            # tqdm.write(f"{i}\trun loss backprop")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.nca.parameters(), max_norm=1.0)
            # tqdm.write(f"{i}\trun optim")
            self.optimizer.step()
            self.scheduler.step()

            self.acc_loss += loss.item()
            self.acc_epochs += 1

            # tqdm.write(f"{i}\tloss {loss.item()}")
            if vis:
                vis.line(
                    X=[self.curr_epoch],
                    Y=[loss.item()],
                    win="loss",
                    update="append" if self.curr_epoch > 0 else None,
                )

            if self.curr_epoch % 100 == 99:
                avg_loss = self.acc_loss / self.acc_epochs
                tqdm.write(f"epoch {self.curr_epoch + 1} loss: {avg_loss}")
                if avg_loss < self.min_loss:
                    if self.min_model and self.min_model != self.last_model:
                        os.remove(self.min_model)
                    self.min_loss = avg_loss
                    self.min_model = self.save()
                if not math.isnan(avg_loss):
                    if self.last_model and self.min_model != self.last_model:
                        os.remove(self.last_model)
                    self.last_model = self.save()
                self.acc_loss = 0
                self.acc_epochs = 0

    def save(self):
        state = {
            "config": self.config,
            "nca": self.nca.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
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
            model_name="alpha",
            model_dir="models",
            hr_dir="data/hr",
            lr_dir="data/lr",
            img_limit=100,
            epochs=1000,
            batch_size=8,
            crop_size=64,
            nca_channels=8,
            optim_lr=1e-3,
            optim_weight_decay=0.01,
            optim_lr_gamma=0.997,
        )
        srnca = SRNCA(config, None, vis)

    srnca.train()
