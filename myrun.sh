#PBS -N iplvisjob
#PBS -m abe
#PBS -M far.arefi@sharif.edu
#PBS -l nodes=1:ppn=1:gpus=1
#PBS -q CE-gpu
cd /home/kasaei2/FarnooshArefi/VIS/
export LD_LIBRARY_PATH=/share/apps/cuda/cuda-10.1/lib64/:$LD_LIBRARY_PATH
export PATH=/share/apps/cuda/cuda-10.1/bin/:$PATH
source /share/apps/Anaconda/anaconda3.7/bin/activate iplvis
python -u iplvisgpu.py > output.txt