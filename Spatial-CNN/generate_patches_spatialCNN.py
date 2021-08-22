import argparse
import os
import random

import cv2
import numpy as np

from utilis import crop_image
from utilis import data_augmentation
from utilis import get_videonames
from utilis import mkdir_p


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script to generate patches from data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('data_dir', type=str, help='Data root directory')
    parser.add_argument(
        'save_dir',
        type=str,
        help='Directory to save generated patches',
    )
    parser.add_argument(
        '--patch_size',
        type=int,
        default=50,
        help='Patch size',
    )
    parser.add_argument('--stride', type=int, default=100, help='Stride size')
    parser.add_argument(
        '--num_aug',
        type=int,
        default=3,
        help='Number of augmentations',
    )

    args = parser.parse_args()

    return args


class IndexedVideoReader(object):

    def __init__(self, filename):
        self._filename = filename
        self._video = cv2.VideoCapture(filename)

    @property
    def filename(self):
        return self._filename

    def __del__(self):
        self._video.release()

    def __len__(self):
        return int(self._video.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(self, index):
        self._video.set(cv2.CAP_PROP_POS_FRAMES, index)

        return self._video.read()


def gen_patch_filename(
    videoname,
    img_idx,
    scale,
    xmin,
    ymin,
    aug_mode,
    extension,
):
    basename = os.path.basename(videoname).replace('.', '_')

    return '{}_{}_{}_{}_{}_{}.{}'.format(
        basename,
        img_idx,
        scale,
        xmin,
        ymin,
        aug_mode,
        extension,
    )


def generate_patches(
    video_noisy,
    video_clean,
    patch_size,
    stride,
    num_aug,
    out_dir_noisy,
    out_dir_clean,
    scales=(1, 0.8),
    output_extension='png',
):
    assert len(video_noisy) == len(video_clean), \
        'Number of frames are not equal for noisy and clean videos pair'

    count_img_pairs = 0
    count_patch_pairs = 0

    for img_idx in range(len(video_noisy)):
        success_noisy, img_noisy = video_noisy.read(img_idx)
        success_clean, img_clean = video_clean.read(img_idx)

        if (not success_noisy) or (not success_clean):
            continue

        count_img_pairs += 1

        for scale in scales:
            new_size = (
                int(img_noisy.shape[0] * scale),
                int(img_noisy.shape[1] * scale),
            )
            img_scaled_noisy = cv2.resize(
                img_noisy,
                new_size,
                interpolation=cv2.INTER_CUBIC,
            )
            img_scaled_clean = cv2.resize(
                img_clean,
                new_size,
                interpolation=cv2.INTER_CUBIC,
            )

            im_width = img_scaled_noisy.shape[1]
            im_height = img_scaled_noisy.shape[0]
            for xmin in range(0, im_width - patch_size, stride):
                for ymin in range(0, im_height - patch_size, stride):
                    bbox = (
                        xmin,
                        ymin,
                        xmin + patch_size,
                        ymin + patch_size,
                    )
                    patch_noisy = crop_image(img_scaled_noisy, bbox)
                    patch_clean = crop_image(img_scaled_clean, bbox)

                    for aug_mode in random.sample(range(7), 3):
                        patch_aug_noisy = data_augmentation(
                            patch_noisy,
                            aug_mode,
                        )
                        patch_aug_clean = data_augmentation(
                            patch_clean,
                            aug_mode,
                        )

                        filename_noisy = os.path.join(
                            out_dir_noisy,
                            gen_patch_filename(
                                video_noisy.filename,
                                img_idx,
                                scale,
                                xmin,
                                ymin,
                                aug_mode,
                                output_extension,
                            ),
                        )
                        cv2.imwrite(filename_noisy, patch_aug_noisy)
                        filename_clean = os.path.join(
                            out_dir_clean,
                            gen_patch_filename(
                                video_clean.filename,
                                img_idx,
                                scale,
                                xmin,
                                ymin,
                                aug_mode,
                                output_extension,
                            ),
                        )
                        cv2.imwrite(filename_clean, patch_aug_clean)

                        count_patch_pairs += 1

    return count_img_pairs, count_patch_pairs


def main(args):
    out_dir_noisy = os.path.join(args.save_dir, 'before')
    out_dir_clean = os.path.join(args.save_dir, 'after')
    mkdir_p(out_dir_noisy)
    mkdir_p(out_dir_clean)

    videonames_noisy = get_videonames(os.path.join(args.data_dir, 'before'))
    videonames_clean = get_videonames(os.path.join(args.data_dir, 'after'))

    num_img_pairs = 0
    num_patch_pairs = 0
    for videoname_noisy, videoname_clean in \
        zip(videonames_noisy, videonames_clean):
        video_noisy = IndexedVideoReader(videoname_noisy)
        video_clean = IndexedVideoReader(videoname_clean)

        count_img_pairs, count_patch_pairs = generate_patches(
            video_noisy,
            video_clean,
            args.patch_size,
            args.stride,
            args.num_aug,
            out_dir_noisy,
            out_dir_clean,
        )

        num_img_pairs += count_img_pairs
        num_patch_pairs += count_patch_pairs

    print ('[*] Number of training image pairs: {}'.format(num_img_pairs))
    print ('[*] Number of training patch pairs: {}'.format(num_patch_pairs))
    print('[*] Patches generated and saved!')


if __name__ == '__main__':
    main(parse_args())
