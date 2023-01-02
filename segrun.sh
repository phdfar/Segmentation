#PBS -N iplvisjob
#PBS -m abe
#PBS -M fa..@...
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -q CE-gpu
cd /home/kasaei2/FarnooshArefi/VIS/
source /share/apps/Anaconda/anaconda3.7/bin/activate iplvis
python -u Segmentation/mystem_kaggle/stemseg/training_m/main.py --save_interval 2000 --display_interval 50 --summary_interval 100 --model_dir some_dir_name --cfg youtube_vis_fake.yaml > output.txt
