#!/usr/bin/env bash

function runexp {

gpu=${1}
arch=${2}
alg=${3}
glr=${4}
iters=${5}
flags=${6}

flags_print="$(echo -e "${flags}" | tr -d '[:space:]')"
flags_print=${flags_print//--/_}

expname=gradinit-${arch}-cifar10-cutout-mixup-alg_${alg}-glr_${glr}-i_${iters}${flags_print}

cmd="
CUDA_VISIBLE_DEVICES=${gpu}
python train_cifar.py  --arch ${arch} --cutout --train-loss mixup
    --gradinit  --gradinit-alg ${alg} --gradinit-eta 0.1
    --gradinit-gamma 1 --gradinit-normalize-grad
    --gradinit-lr ${glr}  --gradinit-min-scale 0.01
    --gradinit-iters ${iters} --gradinit-grad-clip 1
    --expname ${expname} ${flags}
"

eval ${cmd}

}

# runexp  gpu    arch      alg    glr    iters      flags
runexp     0   wrn_28_10   sgd    3e-3    780       "--gradinit-bsize 64  --seed 4096"
