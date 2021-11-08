import argparse

from torch._C import StringType

global args
args = argparse.ArgumentParser()
#args.add_argument("--name",default='nano')
args.add_argument("--age",type=int, default=21)
args = args.parse_args()