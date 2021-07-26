# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:32:05 2019

@author: clausmichele
"""

import numpy as np
import cv2
from glob import glob
import os
import random
from tqdm import tqdm
from imgaug import augmenters as iaa

def gaussian_noise(sigma,image):
	gaussian = np.random.normal(0,sigma,image.shape)
	noisy_image = np.zeros(image.shape, np.float32)
	noisy_image = image + gaussian
	noisy_image = np.clip(noisy_image,0,255)
	noisy_image = noisy_image.astype(np.uint8)
	return noisy_image

def realistic_noise(Ag,Dg,image):
	CT1=1.25e-4
	CT2=1.11e-4
	Nsat=7480
	image = image/255.0
	M=np.sqrt( ((Ag*Dg)/(Nsat*image)+(Dg**2)*((Ag * CT1 + CT2)**2)))
	N = np.random.normal(0,1,image.shape)
	noisy_image = image + N*M
	cv2.normalize(noisy_image, noisy_image, 0, 1.0, cv2.NORM_MINMAX, dtype=-1)
	return noisy_image


def rand_patch(image, size_range=(32, 64)):
    im_width = image.shape[1]
    im_height = image.shape[0]

    patch_width = random.randint(size_range[0], size_range[1])
    patch_height = random.randint(size_range[0], size_range[1])
    patch_x = random.randint(0, im_width - patch_width)
    patch_y = random.randint(0, im_height - patch_height)

    return patch_x, patch_y, patch_width, patch_height


def apply_defects(image):
    seq = iaa.Sequential([
        iaa.imgcorruptlike.Spatter(severity=3),
    ])

    image_aug = image.copy()

    num_repeats = random.randint(10, 40)
    for _ in range(num_repeats):
        patch_x, patch_y, patch_width, patch_height = rand_patch(image_aug)
        patch_image = image_aug[
            patch_y:patch_y+patch_height,
            patch_x:patch_x+patch_width,
        ]
        patch_image_aug = seq(image=patch_image)

        alpha_mask = np.random.rand(patch_height, patch_width)
        alpha_mask = np.stack([alpha_mask] * 3, axis=2)
        image_aug[
            patch_y:patch_y+patch_height,
            patch_x:patch_x+patch_width,
        ] = \
            image_aug[
                patch_y:patch_y+patch_height,
                patch_x:patch_x+patch_width,
            ] * alpha_mask + patch_image_aug * (1 - alpha_mask)

    return image_aug


if __name__=="__main__":

	imgs_path = glob("./data/pristine_images/*.bmp")
	num_of_samples = len(imgs_path)
	imgs_path_train = imgs_path[:int(num_of_samples*0.7)]
	imgs_path_test = imgs_path[int(num_of_samples*0.7):]

	sigma_train = np.linspace(0,50,int(num_of_samples*0.7)+1)
	for i in tqdm(range(int(num_of_samples*0.7)),desc="[*] Creating original-noisy train set..."):
		img_path = imgs_path_train[i]
		img_file = os.path.basename(img_path).split('.bmp')[0]
		sigma = sigma_train[i]
		img_original = cv2.imread(img_path)
		img_noisy = gaussian_noise(sigma,img_original)

		cv2.imwrite("./data/train/noisy/"+img_file+".png",img_noisy)
		cv2.imwrite("./data/train/original/"+img_file+".png",img_original)

	for i in tqdm(range(int(num_of_samples*0.3)),desc="[*] Creating original-noisy test set..."):
		img_path = imgs_path_test[i]
		img_file = os.path.basename(img_path).split('.bmp')[0]
		sigma = np.random.randint(0,50)

		img_original = cv2.imread(img_path)
		img_noisy = gaussian_noise(sigma,img_original)

		cv2.imwrite("./data/test/noisy/"+img_file+".png",img_noisy)
		cv2.imwrite("./data/test/original/"+img_file+".png",img_original)

