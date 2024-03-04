#!/bin/bash


python train.py --dataset cifar100 --arch wideresnet --batch-size 64 --rl 0.1 --rp 0 --ru 0.9 --lr 0.03 --seed 0 --out out/cifar-100/test

