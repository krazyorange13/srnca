import os
import sys

import cv2
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

RATIO = 2
INTERPOLATION = cv2.INTER_LANCZOS4


def downscale(input_dir, output_dir, image):
    in_path = os.path.join(input_dir, image)
    out_path = os.path.join(output_dir, image)
    img = cv2.imread(in_path)
    if img is not None:
        h, w, c = img.shape
        img_resized = cv2.resize(img, (w // RATIO, h // RATIO), interpolation=INTERPOLATION)
        cv2.imwrite(out_path, img_resized)


def batch_downscale(input_dir, output_dir):
    images = os.listdir(input_dir)

    with Pool(cpu_count()) as pool:
        _ = list(tqdm(pool.imap(partial(downscale, input_dir, output_dir), images), total=len(images)))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: input_dir output_dir")
        exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    batch_downscale(input_dir, output_dir)
