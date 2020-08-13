# encoding utf-8
import os
import torch
import logging
import argparse

from Train import Train
from Evalution import Evalution

from Model.DeepPTP import DeepPTP

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(1111)

parser = argparse.ArgumentParser(description='DeepPTP')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--in_channels_S', type=int, default=16)
parser.add_argument('--out_channels_S', type=int, default=32)
parser.add_argument('--kernel_size_S', type=int, default=3)
parser.add_argument('--num_inputs_T', type=int, default=32)
parser.add_argument('--num_channels_T', type=list, default=[64] * 5)
parser.add_argument('--num_outputs_T', type=int, default=4)
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--log_file', type=str, default='./logs/Run.log')
args = parser.parse_args()

if __name__ == '__main__':
    max_accuracy, max_precise, max_recall, max_f1_score = 0., 0., 0., 0.

    if os.path.exists(args.log_file):
        os.remove(args.log_file)
    logging.basicConfig(filename=args.log_file, level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log = logging.getLogger(__name__)

    model = DeepPTP(
        in_channels_S=args.in_channels_S,
        out_channels_S=args.out_channels_S,
        kernel_size_S=args.kernel_size_S,
        num_inputs_T=args.num_inputs_T,
        num_channels_T=args.num_channels_T,
        num_outputs_T=args.num_outputs_T
    )
    if torch.cuda.is_available():
        model.cuda()
    for epoch in range(args.epochs):
        model = Train(model=model, epoch=epoch, batchsize=args.batchsize, lr=args.lr)
        max_accuracy, max_precise, max_recall, max_f1_score = \
            Evalution(
                model=model,
                batchsize=args.batchsize,
                max_accuracy=max_accuracy,
                max_precise=max_precise,
                max_recall=max_recall,
                max_f1_score=max_f1_score,
                log=log,
            )
        torch.cuda.empty_cache()




