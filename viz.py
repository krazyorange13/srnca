import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from main import SRNCA, NCA, SRNCAConfig, Sampler


class Visualization:
    def __init__(
        self,
        model_path: str,
        lr_path: str,
        hr_path: str | None,
        out_path: str | None,
        steps: int,
        update_rate: float = 0.5,
    ):
        self.model_path = model_path
        self.lr_path = lr_path
        self.hr_path = hr_path
        self.out_path = out_path
        self.steps = steps
        self.update_rate = update_rate

        print(f"model: {self.model_path}")
        print(f"LR img: {self.lr_path}")
        if self.hr_path:
            print(f"HR img: {self.hr_path}")
        if self.out_path:
            print(f"SR img: {self.out_path}")

        state = torch.load(model_path, weights_only=False)
        self.config: SRNCAConfig = state["config"]

        # self.nca = NCA(channels=self.config.nca_channels, update_rate=self.config.nca_update_rate)
        self.srnca = SRNCA(self.config, state, None)

    def viz(self):
        x = self.img_to_nca(self.lr_path)
        print("upscaling... ", end="", flush=True)
        with torch.no_grad():
            steps = (self.config.nca_steps[0] + self.config.nca_steps[1]) // 2
            x = self.srnca.nca(x, steps=steps)
        print("done!")
        img = self.nca_to_img(x)

        if self.out_path:
            cv2.imwrite(self.out_path, (img * 255).astype(np.uint8))

        print(self.lr_path, self.hr_path, self.out_path)

        quit = False
        cv2.namedWindow("upscaled", cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.lr_path, cv2.WINDOW_NORMAL)
        lr_img = cv2.imread(self.lr_path, cv2.IMREAD_COLOR)
        if self.hr_path:
            cv2.namedWindow(self.hr_path, cv2.WINDOW_NORMAL)
            hr_img = cv2.imread(self.hr_path, cv2.IMREAD_COLOR)

        while not quit:
            cv2.imshow("upscaled", img)
            cv2.imshow(self.lr_path, lr_img)  # type: ignore
            if self.hr_path:
                cv2.imshow(self.hr_path, hr_img)  # type: ignore
            if cv2.waitKey(int(1000 / 30)) & 0xFF == ord("q"):
                quit = True
                break
        cv2.destroyAllWindows()

    def img_to_nca(self, img_path):
        # load img
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = img_t.unsqueeze(dim=0)

        # hidden layers
        _b, _c, _h, _w = img_t.shape
        _c_ = self.config.nca_channels - _c
        hid = torch.zeros(_b, _c_, _h, _w)
        img_t = torch.cat([img_t, hid], dim=1)

        # upscale
        img_t = F.interpolate(img_t, scale_factor=2, mode="nearest")

        return img_t

    def nca_to_img(self, x: torch.Tensor):
        b, c, h, w = x.shape
        x_ = x[0, :3, :, :]
        x_ = x_.permute(1, 2, 0)
        x_ = x_.clamp(0, 1)
        return x_.numpy()


if __name__ == "__main__":
    if not len(sys.argv) > 3:
        print("usage: model_path lr=lowres_path [hr=highres_path] [out=output_path]")
        exit(1)

    argv = sys.argv[:]

    lr = [arg for arg in sys.argv if arg.startswith("lr=")]
    if lr:
        lr = lr[0]
        argv.remove(lr)
        lr = lr[3:]
    else:
        print("usage: model_path lr=lowres_path [hr=highres_path] [out=output_path]")
        exit(1)

    hr = [arg for arg in sys.argv if arg.startswith("hr=")]
    if hr:
        hr = hr[0]
        argv.remove(hr)
        hr = hr[3:]
    else:
        hr = None

    out = [arg for arg in sys.argv if arg.startswith("out=")]
    if out:
        out = out[0]
        argv.remove(out)
        out = out[4:]
    else:
        out = None

    viz = Visualization(model_path=argv[1], lr_path=lr, hr_path=hr, out_path=out, steps=32)
    viz.viz()
