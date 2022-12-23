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

  if args.mode=='cbam':
    os.system('cp Segmentation/mystem_kaggle/stemseg/training_m/stemseg/modeling/embedding_decoder_cbam.py Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/embedding_decoder.py')

    
  if args.mode=='reset':
    os.system('rm -rf Segmentation')
    os.system('git clone https://github.com/phdfar/Segmentation')
    os.system('cp /content/Segmentation/mystem_kaggle/stemseg/training_m/stemseg/config/youtube_vis_fake.yaml /content/Segmentation/mystem_colab/stemseg/inference/stemseg/config/youtube_vis.yaml')
    os.system('rm -rf *.pth')
   
  if args.mode=='freez[cnt+myopt]':
    os.system('cp /content/Segmentation/mystem_kaggle/stemseg/training_m/stemseg/config/youtube_vis_plus.yaml /content/Segmentation/mystem_colab/stemseg/inference/stemseg/config/youtube_vis.yaml')
    os.system('cp Segmentation/mystem_kaggle/stemseg/training_m/stemseg/modeling/embedding_decoder_plus3.py Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/embedding_decoder.py')
    os.system('cp Segmentation/mystem_kaggle/stemseg/training_m/stemseg/modeling/semseg_decoder_plus.py Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/semseg_decoder.py')
    os.system('cp /content/Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/inference_model_plus3.py /content/Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/inference_model.py')
    os.system('cp Segmentation/mystem_kaggle/stemseg/training_m/stemseg/modeling/opt.py Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/opt.py')
    os.system('cp Segmentation/mystem_kaggle/stemseg/training_m/stemseg/modeling/embedding_utils.py Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/embedding_utils.py')
    
  if args.mode=='myopt':
    os.system('cp /content/Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/model_builder_color.py Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/model_builder.py')
    os.system('cp /content/Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/inference_model_myopt.py /content/Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/inference_model.py')
    os.system('cp Segmentation/mystem_kaggle/stemseg/training_m/stemseg/modeling/opt.py Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/opt.py')
    os.system('cp /content/Segmentation/mystem_kaggle/stemseg/training_m/stemseg/config/youtube_vis_fake.yaml /content/Segmentation/mystem_colab/stemseg/inference/stemseg/config/youtube_vis.yaml')

  if args.mode=='myoptwa':
    os.system('cp /content/Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/model_builder_color.py Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/model_builder.py')
    os.system('cp /content/Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/inference_model_myopt_wa.py /content/Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/inference_model.py')
    os.system('cp Segmentation/mystem_kaggle/stemseg/training_m/stemseg/modeling/opt.py Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/opt.py')
    os.system('cp /content/Segmentation/mystem_kaggle/stemseg/training_m/stemseg/config/youtube_vis_fake.yaml /content/Segmentation/mystem_colab/stemseg/inference/stemseg/config/youtube_vis.yaml')
  
  if args.mode=='myoptfm':
    os.system('cp /content/Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/model_builder_color.py Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/model_builder.py')
    os.system('cp /content/Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/inference_model_myopt_fm.py /content/Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/inference_model.py')
    os.system('cp Segmentation/mystem_kaggle/stemseg/training_m/stemseg/modeling/opt.py Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/opt.py')
    os.system('cp /content/Segmentation/mystem_kaggle/stemseg/training_m/stemseg/config/youtube_vis_fake.yaml /content/Segmentation/mystem_colab/stemseg/inference/stemseg/config/youtube_vis.yaml')

  if args.mode=='myoptwap':
    os.system('cp /content/Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/model_builder_color.py Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/model_builder.py')
    os.system('cp /content/Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/inference_model_myopt_wap.py /content/Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/inference_model.py')
    os.system('cp Segmentation/mystem_kaggle/stemseg/training_m/stemseg/modeling/opt.py Segmentation/mystem_colab/stemseg/inference/stemseg/modeling/opt.py')
    os.system('cp /content/Segmentation/mystem_kaggle/stemseg/training_m/stemseg/config/youtube_vis_fake.yaml /content/Segmentation/mystem_colab/stemseg/inference/stemseg/config/youtube_vis.yaml')
  
    
     
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str ,default='main', required=False)
    args = parser.parse_args()
    init(args)


clear_output()


