#!/bin/bash

set -e

data_root=${1}

echo [*] Dataset preparation script. Run it once before training.

echo [*] Downloading the raw videos...
mkdir -p ${data_root}/raw
pushd ${data_root}/raw
# Download videos from TWCC.
twccli cp cos -bkt videos -okey mazu_full_before.mov -sync from-cos
twccli cp cos -bkt videos -okey mazu_full_after.mov -sync from-cos
popd
echo [*] Success to download the raw videos.

echo [*] Splitting the raw videos into trainining / validation /testing sets
mkdir -p ${data_root}/vid_train/before
mkdir -p ${data_root}/vid_train/after
mkdir -p ${data_root}/val/before
mkdir -p ${data_root}/val/after
mkdir -p ${data_root}/test/before
mkdir -p ${data_root}/test/after
# Split the videos.
# TODO (SuJiaKuan): Is there any elegant way to prevent the hard splitting?
ffmpeg -i mazu_full_before.mov -ss 00:00:03 -t 00:23:00 -async 1 -c copy -y ${data_root}/vid_train/before/mazu_full.mov
ffmpeg -i mazu_full_after.mov -ss 00:00:03 -t 00:23:00 -async 1 -c copy -y ${data_root}/vid_train/after/mazu_full.mov
ffmpeg -i mazu_full_before.mov -ss 00:23:05 -t 00:02:00 -async 1 -y ${data_root}/val/before/%08d.png
ffmpeg -i mazu_full_after.mov -ss 00:23:05 -t 00:02:00 -async 1 -y ${data_root}/val/after/%08d.png
ffmpeg -i mazu_full_before.mov -ss 00:25:07 -t 00:02:00 -async 1 -y ${data_root}/test/before/%08d.png
ffmpeg -i mazu_full_after.mov -ss 00:25:07 -t 00:02:00 -async 1 -y ${data_root}/test/after/%08d.png
echo [*] Success to split the videos.

echo [*] Generating the patches for training.
python3 generate_patches_spatialCNN.py  ${data_root}/vid_train ${data_root}/train --num_workers $(nproc --all)
echo [*] Success to enerate the patches.

echo [*] All preparation processes are done!
