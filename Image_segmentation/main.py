from argparse import ArgumentParser
import path
import train
def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

def main(args):
  train.start(args)
  return args 

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--network', type=str , required=True)
    parser.add_argument('--model_dir', type=str , required=True)
    parser.add_argument('--trainpath', type=str , default='/content/train/', required=False)

    parser.add_argument('--imagesize', type=tuple_type, required=True)

    parser.add_argument('--epoch', type=int, default=15 , required=False)
    parser.add_argument('--batchsize', type=int, default=32, required=False)

    parser.add_argument('--num_instance', type=int, default=1 , required=False)
    parser.add_argument('--unq_class', type=int, default=1, required=False)


    parser.add_argument('--classid', type=tuple_type, default=(0) , required=False)


    args = parser.parse_args()

    main(args)