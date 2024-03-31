#!/bin/bash

python Q21_image_optimization.py --prompt "a hamburger" --sds_guidance 0 --output_dir output_Q24
python Q21_image_optimization.py --prompt "a hamburger" --sds_guidance 1 --output_dir output_Q24
python Q21_image_optimization.py --prompt "a standing corgi dog" --sds_guidance 0 --output_dir output_Q24
python Q21_image_optimization.py --prompt "a standing corgi dog" --sds_guidance 1 --output_dir output_Q24
python Q21_image_optimization.py --prompt "a robot arm" --sds_guidance 0 --output_dir output_Q24
python Q21_image_optimization.py --prompt "a robot arm" --sds_guidance 1 --output_dir output_Q24
