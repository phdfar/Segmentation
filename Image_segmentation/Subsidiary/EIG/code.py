import os

os.system('python extract.py extract_features  --images_list "data/VOC2012/lists/images.txt"  --images_root "data/VOC2012/images" --output_dir "data/VOC2012/features/dino_vits16"  --model_name dino_vits16  --batch_size 1')