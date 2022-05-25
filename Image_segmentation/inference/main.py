from argparse import ArgumentParser
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

  run.start(args)
  return args 

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--channel_input', type=int ,default=3, required=False)
    parser.add_argument('--mode', type=str ,default='only_vis', required=False)
    parser.add_argument('--model_dir', type=str , required=True)
    parser.add_argument('--basepath', type=str , default='/content/', required=False)
    parser.add_argument('--imagesize', type=tuple_type, required=True)
    parser.add_argument('--colorspace', type=str ,default='rgb', required=False)
    parser.add_argument('--classid', type=tuple_type, default=(1000) , required=False)


    args = parser.parse_args()

    main(args)
