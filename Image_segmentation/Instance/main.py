from argparse import ArgumentParser
import path
import run
import numpy as np
def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

def main(args):
  if (args.classid)[0]==1000:
    args.classid = list(np.linspace(1,40,40).astype('int32'))
    print(args.classid )
  if args.input_imagesize==0:
      args.input_imagesize=args.imagesize
  run.start(args)
  return args 

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task', type=str ,default='binary_seg', required=False)
    parser.add_argument('--channel_input', type=int ,default=3, required=False)
    parser.add_argument('--branch_input', type=int ,default=1, required=False)
    parser.add_argument('--num_class', type=int ,default=1, required=False)
    parser.add_argument('--mode', type=str ,default='train', required=False)
    parser.add_argument('--restore', type=bool ,default=False, required=False)
    parser.add_argument('--network', type=str , required=True)
    parser.add_argument('--model_dir', type=str , required=True)
    parser.add_argument('--corrector', type=str,default='', required=False)
    parser.add_argument('--basepath', type=str , default='/content/', required=False)
    parser.add_argument('--baseinput', type=str , default='/content/', required=False)
    
    parser.add_argument('--upload', type=str ,default='local', required=False)
    parser.add_argument('--imagesize', type=tuple_type, required=True)
    parser.add_argument('--input_imagesize',default=(0), type=tuple_type, required=False)
    parser.add_argument('--epoch', type=int, default=15 , required=False)
    parser.add_argument('--batchsize', type=int, default=32, required=False)
    parser.add_argument('--config', type=int, default=0, required=False)

    parser.add_argument('--num_instance', type=int, default=1 , required=False)
    parser.add_argument('--unq_class', type=int, default=1, required=False)
    parser.add_argument('--loss', type=str, default='default', required=False)

    parser.add_argument('--colorspace', type=str ,default='rgb', required=False)
    parser.add_argument('--classid', type=tuple_type, default=(0) , required=False)


    args = parser.parse_args()

    main(args)
