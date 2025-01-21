#!/bin/bash
file_path=images-0912-747-origin-071303-600
denoise_times=600
seed=1234

python main.py --exp ./runs/ --config church.yml --seed $i --sample -i $file_path --npy_name lsun_church --sample_step 1 --t $denoise_times --ni