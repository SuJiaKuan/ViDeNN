# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:32:05 2019

@author: clausmichele
"""

import argparse
from glob import glob
import tensorflow as tf
import os
from model_spatialCNN import denoiser
from utilis import load_data
from utilis import load_images
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('data_dir', type=str, help='Data root directory')
    parser.add_argument(
        '--epoch',
        dest='epoch',
        type=int,
        default=50,
        help='# of epoch',
    )
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        type=int,
        default=64,
        help='# images in batch',
    )
    parser.add_argument(
        '--lr',
        dest='lr',
        type=float,
        default=0.001,
        help='initial learning rate for adam',
    )
    parser.add_argument(
        '--use_gpu',
        dest='use_gpu',
        type=int,
        default=1,
        help='gpu flag, 1 for GPU and 0 for CPU',
    )
    parser.add_argument(
        '--phase',
        dest='phase',
        default='train',
        help='train or test',
    )
    parser.add_argument(
        '--checkpoint_dir',
        dest='ckpt_dir',
        default='./ckpt',
        help='checkpoints are saved here',
    )
    parser.add_argument(
        '--save_dir',
        dest='save_dir',
        default='./data/denoised',
        help='denoised sample are saved here',
    )

    args = parser.parse_args()

    return args


def denoiser_train(denoiser, lr, args):
    train_data_dir = os.path.join(args.data_dir, 'train')
    noisy_eval_files = glob('./data/test/noisy/*.png')
    noisy_eval_files = sorted(noisy_eval_files)
    eval_data_noisy = load_images(noisy_eval_files)
    eval_files = glob('./data/test/original/*.png')
    eval_files = sorted(eval_files)

    eval_data = load_images(eval_files)
    denoiser.train(
        train_data_dir,
        eval_data[0:20],
        eval_data_noisy[0:20],
        batch_size=args.batch_size,
        ckpt_dir=args.ckpt_dir,
        epoch=args.epoch,
        lr=lr,
    )


def denoiser_test(denoiser, args):
    noisy_eval_files = glob('./data/test/noisy/*.png')
    noisy_eval_files = sorted(noisy_eval_files)
    eval_files = glob('./data/test/original/*.png')
    eval_files = sorted(eval_files)
    denoiser.test(
        noisy_eval_files,
        eval_files,
        ckpt_dir=args.ckpt_dir,
        save_dir=args.save_dir,
    )


def denoiser_for_temp3_training(denoiser, args):
    noisy_eval_files = glob('../Temp3-CNN/data/train/noisy/*/*.png')
    noisy_eval_files = sorted(noisy_eval_files)
    eval_files = glob('../Temp3-CNN/data/train/original/*/*.png')
    eval_files = sorted(eval_files)
    denoiser.test(
        noisy_eval_files,
        eval_files,
        ckpt_dir=args.ckpt_dir,
        save_dir='../Temp3-CNN/data/train/denoised/',
    )


def main(_):
    args = parse_args()

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    lr = args.lr * np.ones([args.epoch])
    lr[3:] = lr[0] / 10.0
    if args.use_gpu:
        # Control the gpu memory setting per_process_gpu_memory_fraction
        print("GPU\n")
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = denoiser(sess)
            if args.phase == 'train':
                denoiser_train(model, lr, args)
            elif args.phase == 'test':
                denoiser_test(model, args)
            elif args.phase == 'test_temp':
                denoiser_for_temp3_training(model, args)
            else:
                print('[!] Unknown phase')
                exit(0)
    else:
        print("CPU\n")
        with tf.device('/cpu:0'):
            with tf.Session() as sess:
                model = denoiser(sess)
                if args.phase == 'train':
                    denoiser_train(model, lr, args)
                elif args.phase == 'test':
                    denoiser_test(model, args)
                elif args.phase == 'test_temp':
                    denoiser_for_temp3_training(model, args)
                else:
                    print('[!] Unknown phase')
                    exit(0)


if __name__ == '__main__':
    tf.app.run()
