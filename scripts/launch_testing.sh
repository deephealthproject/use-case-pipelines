#!/bin/sh

EXECUTABLE="${HOME}/temp/use_case_pipeline/build/MS_SEGMENTATION_TRAINING"

onnx_file=$(ls -rt checkpoints | tail -1)

${EXECUTABLE}        -d ${HOME}/data/fli-iam/ms_segmentation.yml \
                     -b 6 --num_classes 1 --n_channels 1 \
                     --gpu 1,0 \
                     --checkpoint_dir checkpoints \
                     --checkpoint checkpoints/${onnx_file} \
                     --do_not_train --do_test

