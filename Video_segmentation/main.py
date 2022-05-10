from argparse import ArgumentParser
import path
import run
def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

def main(args):
  run.start(args)
  return args 

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--task', type=str ,default='binary_seg', required=False)
    parser.add_argument('--subseq_length', type=int ,default=4, required=False)
    parser.add_argument('--mode', type=str ,default='train', required=False)
    parser.add_argument('--restore', type=bool ,default=False, required=False)
    parser.add_argument('--network', type=str , required=True)
    parser.add_argument('--model_dir', type=str , required=True)
    parser.add_argument('--basepath', type=str , default='/content/', required=False)
    parser.add_argument('--upload', type=str ,default='local', required=False)
    parser.add_argument('--imagesize', type=tuple_type, required=True)

    parser.add_argument('--epoch', type=int, default=15 , required=False)
    parser.add_argument('--batchsize', type=int, default=32, required=False)

    parser.add_argument('--num_instance', type=int, default=1 , required=False)
    parser.add_argument('--unq_class', type=int, default=1, required=False)


    parser.add_argument('--classid', type=tuple_type, default=(0) , required=False)


    args = parser.parse_args()

    main(args)
