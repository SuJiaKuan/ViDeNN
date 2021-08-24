import argparse
import glob
import os
import math
import random
from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from tqdm import tqdm

from alignment import align_images
from utilis import save_pickle
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
    parser.add_argument('--margin', type=int, default=100, help='Margin size')
    parser.add_argument(
        '--num_aug',
        type=int,
        default=3,
        help='Number of augmentations',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='Number of workers',
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


def calc_diff_value(img_noisy, img_clean, patch_size):
    img_diff = cv2.absdiff(img_noisy, img_clean)
    diff_value = np.sum(img_diff) / (patch_size ** 2)

    return diff_value


def generate_patches_imp(
    worker_id,
    videoname_noisy,
    videoname_clean,
    img_idxes,
    patch_size,
    stride,
    margin,
    num_aug,
    out_dir_noisy,
    out_dir_clean,
    scales,
    output_extension,
):
    video_noisy = IndexedVideoReader(videoname_noisy)
    video_clean = IndexedVideoReader(videoname_clean)

    count_img_pairs = 0
    count_patch_pairs = 0
    diff_value_mapping = {}

    for idx, img_idx in enumerate(img_idxes):
        print('[Worker {}] Processing {} / {}'.format(
            worker_id,
            idx,
            len(img_idxes),
        ))

        success_noisy, img_noisy = video_noisy.read(img_idx)
        success_clean, img_clean = video_clean.read(img_idx)

        if (not success_noisy) or (not success_clean):
            continue

        success_align, img_clean = align_images(img_clean, img_noisy)

        if not success_align:
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
            xmins = range(margin, im_width - patch_size - margin, stride)
            ymins = range(margin , im_height - patch_size - margin, stride)
            for xmin in xmins:
                for ymin in ymins:
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

                        diff_value = calc_diff_value(
                            patch_aug_noisy,
                            patch_aug_clean,
                            patch_size,
                        )
                        filename = os.path.basename(filename_noisy)
                        diff_value_mapping[filename] = diff_value

                        count_patch_pairs += 1

    return count_img_pairs, count_patch_pairs, diff_value_mapping


def generate_patches(
    videoname_noisy,
    videoname_clean,
    patch_size,
    stride,
    margin,
    num_aug,
    num_workers,
    out_dir_noisy,
    out_dir_clean,
    scales=(1, 0.8),
    output_extension='png',
):
    video_noisy = IndexedVideoReader(videoname_noisy)
    video_clean = IndexedVideoReader(videoname_clean)

    assert len(video_noisy) == len(video_clean), \
        'Number of frames are not equal for noisy and clean videos pair'

    chunk_size = math.ceil(len(video_noisy) / num_workers)
    img_idxes = list(range(len(video_noisy)))
    img_idxes_chunks = [
        img_idxes[i:i + chunk_size]
        for i in range(0, len(video_noisy), chunk_size)
    ]

    count_img_pairs = 0
    count_patch_pairs = 0
    diff_value_mapping = {}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for worker_id, img_idxes in enumerate(img_idxes_chunks):
            future = executor.submit(
                generate_patches_imp,
                worker_id,
                videoname_noisy,
                videoname_clean,
                img_idxes,
                patch_size,
                stride,
                margin,
                num_aug,
                out_dir_noisy,
                out_dir_clean,
                scales,
                output_extension,
            )
            futures.append(future)

        for future in as_completed(futures):
            results = future.result()
            count_img_pairs += results[0]
            count_patch_pairs += results[1]
            diff_value_mapping.update(results[2])

    return count_img_pairs, count_patch_pairs, diff_value_mapping


def validate_pairs(dir_noisy, dir_clean, extension='png'):
    print('Validate the generated pairs...')

    filenames = [
        os.path.basename(f)
        for f in glob.glob(os.path.join(dir_noisy, '*.{}'.format(extension)))
    ]

    valid_filenames = []
    for filename in tqdm(filenames):
        if os.path.exists(os.path.join(dir_clean, filename)):
            valid_filenames.append(filename)

    return valid_filenames


def generate_diff_values(filenames, diff_value_mapping):
    print('Generate diff values...')

    diff_values = []
    for filename in tqdm(filenames):
        diff_values.append(diff_value_mapping[filename])

    return np.array(diff_values)


def main(args):
    out_dir_noisy = os.path.join(args.save_dir, 'before')
    out_dir_clean = os.path.join(args.save_dir, 'after')
    filenames_path = os.path.join(args.save_dir, 'filenames.pickle')
    diff_values_path = os.path.join(args.save_dir, 'diff_values.npy')

    mkdir_p(out_dir_noisy)
    mkdir_p(out_dir_clean)

    videonames_noisy = get_videonames(os.path.join(args.data_dir, 'before'))
    videonames_clean = get_videonames(os.path.join(args.data_dir, 'after'))

    num_img_pairs = 0
    num_patch_pairs = 0
    diff_value_mapping_all = {}
    for videoname_noisy, videoname_clean in \
            zip(videonames_noisy, videonames_clean):
        count_img_pairs, count_patch_pairs, diff_value_mapping = generate_patches(
            videoname_noisy,
            videoname_clean,
            args.patch_size,
            args.stride,
            args.margin,
            args.num_aug,
            args.num_workers,
            out_dir_noisy,
            out_dir_clean,
        )

        num_img_pairs += count_img_pairs
        num_patch_pairs += count_patch_pairs
        diff_value_mapping_all.update(diff_value_mapping)

    valid_filenames = validate_pairs(out_dir_noisy, out_dir_clean)
    save_pickle(valid_filenames, filenames_path)

    diff_values = generate_diff_values(
        valid_filenames,
        diff_value_mapping_all,
    )
    np.save(diff_values_path, diff_values)

    print('[*] Number of detected image pairs: {}'.format(num_img_pairs))
    print('[*] Number of generated patch pairs: {}'.format(num_patch_pairs))
    print('[*] Number of valid patch pairs: {}'.format(len(valid_filenames)))
    print('[*] Patches generated and saved!')


if __name__ == '__main__':
    main(parse_args())
