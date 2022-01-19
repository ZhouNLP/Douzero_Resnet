import os
import torch
from douzero.dmc import parser, train

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    flags = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_devices
    train(flags)
