import os
from IPython.display import clear_output
from argparse import ArgumentParser

def init(args):
  if args.mode=='raftrgb':
      
    os.system('cp Segmentation/mystem_colab/stemseg/inference/stemseg/data/generic_video_dataset_parser_6C.py  Segmentation/mystem_colab/stemseg/inference/stemseg/data/generic_video_dataset_parser.py')
    os.system('cp Segmentation/mystem_kaggle/stemseg/training_m/stemseg/data/common_6C.py Segmentation/mystem_colab/stemseg/inference/stemseg/data/common.py')
    os.system('cp Segmentation/mystem_kaggle/stemseg/training_m/stemseg/structures/image_list_6C.py Segmentation/mystem_colab/stemseg/inference/stemseg/structures/image_list.py')
    os.system('cp /content/Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/model_builder_color.py Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/model_builder.py')
    os.system('cp /content/Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/inference_model_6C.py /content/Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/inference_model.py')
    


  if args.mode=='emcy2':
    os.system('cp Segmentation/mystem_kaggle/stemseg/training_m/stemseg/modeling/embedding_decoder_cyclebam2.py Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/embedding_decoder.py')
    
  if args.mode=='emcy4':
    os.system('cp Segmentation/mystem_kaggle/stemseg/training_m/stemseg/modeling/embedding_decoder_cyclebam4.py Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/embedding_decoder.py')
    
     
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str ,default='main', required=False)
    args = parser.parse_args()
    init(args)


clear_output()


