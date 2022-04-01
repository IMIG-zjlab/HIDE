#!/usr/bin/env bash

sudo xmutil getpkgs
sudo dnf install packagegroup-kv260-smartcam.noarch
sudo xmutil listapps
sudo xmutil unloadapp
sudo xmutil loadapp kv260-smartcam
mkdir /mnt/usb
mount -t vfat /dev/sda1 /mnt/usb/
cd /mnt/usb
sudo smartcam --mipi -W 1920 -H 1080 --target file
ffmpeg -r 30 -i ./out.h264 -f image2 ./%03d.jpeg


