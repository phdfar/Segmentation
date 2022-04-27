
import random
import path

def start(args):

  val_samples = 1000
  random.Random(1337).shuffle(input_img_paths)
  random.Random(1337).shuffle(target_img_paths)
  train_input_img_paths = input_img_paths[:-val_samples]
  train_target_img_paths = target_img_paths[:-val_samples]
  val_input_img_paths = input_img_paths[-val_samples:]
  val_target_img_paths = target_img_paths[-val_samples:]

  # Instantiate data Sequences for each split
  train_gen = path.dataloader(args)
  val_gen = path.dataloader(args)

