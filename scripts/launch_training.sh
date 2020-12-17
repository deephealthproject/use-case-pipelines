#!/bin/sh


EXECUTABLE="${HOME}/temp/use_case_pipeline/build/MS_SEGMENTATION_TRAINING"

nohup ${EXECUTABLE} -d ${HOME}/data/fli-iam/ms_segmentation.yml \
                    -b 6 -e 500 --num_classes 1 --n_channels 1 \
                    --loss mse --learning_rate 1.0e-5 \
                    --gpu 1,0 \
                    --model UNetWithPaddingBN \
                    --checkpoint_dir checkpoints \
        >OUT 2>ERR &

