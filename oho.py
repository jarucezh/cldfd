import os
import math
import torch
import argparse
import torch.nn as nn
from tqdm import tqdm
from loss import ntxent
from utils import save_load
from torch.optim import Adam, SGD
from methods import student9 as student



def main(args):
    if not os.path.isdir(args.dir):
        os.makedirs(args.dir)

    stu = student.Student(dataset_name = args.target_dataset,
                          backbone_name = args.model,
                          args = args,
                          alpha = args.alpha)


    if args.optim == "adam":
        optimizer = Adam(params = stu.trainable_modules,
                         lr = args.lr, weight_decay = args.wd)

    elif args.optim == "sgd":
        optimizer = SGD(params = stu.trainable_modules,
                        lr = args.lr,
                        weight_decay = args.wd,momentum=0.9,
                        nesterov=False)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 500], gamma=0.1, last_epoch=-1)

    simclr_criterion = ntxent.NTXentLoss('cuda', args.bsize, temperature = args.temp, use_cosine_similarity = True)
    best_loss = math.inf
    kd_criterion = Fitnets()
    for epoch in tqdm(range(args.epochs)):
        stu.train_loop(epoch, simclr_criterion, optimizer)
        scheduler.step()
        if epoch % args.eval_freq == 0:
            loss_val = stu.validate(epoch, simclr_criterion, kd_criterion)
            if loss_val < best_loss:
                save_load.save(stu.student, stu.simclr_proj, os.path.join(args.dir, f"checkpoint_best.pkl"), epoch + 1, args)
                best_loss = loss_val

        if (epoch + 1)  % args.save_freq == 0 or epoch == 0:
            save_load.save(stu.student, stu.simclr_proj, os.path.join(args.dir, f"checkpoint_{epoch + 1}.pkl"), epoch + 1, args)


parser = argparse.ArgumentParser(description='CLD_FD')
parser.add_argument('--dir', type=str, default='tmp/tmp',
                    help='directory to save the checkpoints')
parser.add_argument('--ss_proj_dim', type=int, default=128,
                    help='directory to save the checkpoints')
parser.add_argument('--bsize', type=int, default=32,
                    help='batch_size for the training')
parser.add_argument('--epochs', type=int, default=600,
                    help='Number of training epochs')
parser.add_argument('--save_freq', type=int, default=50,
                    help='Frequency (in epoch) to save')
parser.add_argument('--eval_freq', type=int, default=2,
                    help='Frequency (in epoch) to evaluate on the val set')
parser.add_argument('--alpha', type=float, default = 2,
                    help='Weight of Knowledge Distillation Loss')
parser.add_argument('--coef', type=float, default=1,
                    help='The largest proportion of old students knowledge')
parser.add_argument('--seed', type=int, default=1,
                    help='Seed for randomness')
parser.add_argument('--wd', type=float, default=1e-4,
                    help='Weight decay for the model')
parser.add_argument('--num_workers', type=int, default=8,
                    help='Number of workers for dataloader')
parser.add_argument('--model', type=str, default='resnet10',
                    help='Backbone model')
parser.add_argument('--teacher_path', type=str, default = "../final/tmp/sl_1gpu/checkpoint_400.pkl",
                    help='path to the teacher model')
parser.add_argument('--teacher_path_version', type=int, default=1,
                    help='how to load the student')
parser.add_argument('--student_path', type = str,
                    help='path to the student model')
parser.add_argument('--student_path_version', type=int, default=1,
                    help='how to load the student')
parser.add_argument('--lr', type=float, default=0.1,
                    help='Learning rate')
parser.add_argument('--momentum', type=float, default=0.99,
                    help='Learning rate')
parser.add_argument('--temp', type=float, default=1,
                    help='Temperature of SIMCLR')
parser.add_argument('--ratio', type=float, default=0.9,
                    help='loading ratio of dataset')
parser.add_argument('--target_dataset', type=str, default = "EuroSAT", choices = ["EuroSAT", "ISIC", "ChestX", "CropDisease"],
                    help='the target domain dataset')
parser.add_argument('--target_subset_split', type=str, default = "splits/EuroSAT_20_unlabeled.csv",
                    help='path to the csv files that specify the unlabeled split for the target dataset')
parser.add_argument('--img_size', type=int, default=224,
                    help='Resolution of the input image')
parser.add_argument('--optim', type=str, default="sgd",
                    help='Optimizer to be used')

args = parser.parse_args()
print(args)
main(args)

