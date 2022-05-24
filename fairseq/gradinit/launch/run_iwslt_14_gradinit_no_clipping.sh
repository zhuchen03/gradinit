#!/usr/bin/env bash

function runexp {

gpu=${1}
glr=${2}
gbeta2=${3}
giters=${4}
smin=${5}
seed=${6}

mkdir logs

geta=1e-3
expname="iwslt14-gradinit-init-glr${glr}-geta${geta}-gbeta2_${gbeta2}-giters${giters}-smin${smin}-seed${seed}"
logname=logs/${expname}-fp16.log

smax=100
lnlr=-1
gclip=1e8

echo ${logname}


CUDA_VISIBLE_DEVICES=${gpu} \
python gradinit/train.py \
  data-bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --max-tokens 4096 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --min-lr 1e-9 \
    --lr 5e-4 --lr-scheduler linear --warmup-updates 1 --scheduler-max-update 100000 \
    --dropout 0.3 --weight-decay 0.0001 --attention-dropout 0.1 --relu-dropout 0.1 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric  --max-update 100000 \
    --user-dir gradinit/utils --use-gradinit --no-save --gradinit-lr ${glr} \
    --save-dir chks/${expname} --fp16 \
    --gradinit-eta ${geta} --gradinit-max-scale ${smax} --gradinit-ln-lr ${lnlr} \
    --gradinit-min-scale ${smin}  --gradinit-grad-clip ${gclip} \
    --gradinit-beta1 0.9 --gradinit-beta2 ${gbeta2} --no-progress-bar --gradinit-iters ${giters} --seed ${seed} \
#    > ${logname} 2>&1

}


# runexp gpu     glr     gbeta2    giters        smin    seed
runexp    0     2.5e-3    0.98      780          0.01    9017

