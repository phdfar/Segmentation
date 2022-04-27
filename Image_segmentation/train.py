
import random
import path

def start(args):

  allframe_train,allframe_val,allframe_test = path.getinfo(args)
  random.Random(1337).shuffle(allframe_train)
  
  # Instantiate data Sequences for each split
  train_gen = path.dataloader(args,allframe_train)
  val_gen = path.dataloader(args,allframe_val)
  test_gen = path.dataloader(args,allframe_test)

