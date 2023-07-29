
import os
import copy
import time
import utils
import torch
import models
import data_loader
import torch.nn as nn
from itertools import cycle
import torch.nn.functional as F
from data_loader import dataset, transform


class Student(nn.Module):
    def __init__(self,
                 dataset_name,
                 backbone_name,
                 test_aug="test_test_test",
                 train_aug="strong_strong_strong",
                 args=None,
                 alpha=0.2,
                 ss_loss=True):

        super(Student, self).__init__()
        self.alpha = alpha
        self.ss_loss = ss_loss

        utils.fixseed(args.seed)
        if args is None:
            raise ValueError("No training parameters input!!!!!!")
        self.args = args
        self.dataset_name = dataset_name

        self.simclr_proj = models.Projector_SimCLR(self.feature_dim, args.ss_proj_dim)
        self.kd_proj0 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1),
                                      nn.BatchNorm2d(128),
                                      nn.ReLU(inplace=True))

        self.kd_proj1 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(inplace=True))

        self.kd_proj2 = nn.Sequential(nn.Conv2d(256, 512, 3, 2, 1),
                                         nn.BatchNorm2d(512),
                                         nn.ReLU(inplace = True))
        ###### Initialize the models
        self.load_teacher(backbone_name)
        self.load_student(backbone_name)


        if torch.cuda.is_available():
            self.to_device()

        ##### Initialize the dataloaders
        self.prepare_dataloader(train_aug=train_aug, test_aug=test_aug)
        self.trainable_modules = [{'params': self.student.parameters()},
                                  {'params': self.simclr_proj.parameters()},
                                  {'params': self.kd_proj0.parameters()},
                                  {'params': self.kd_proj1.parameters()},
                                  {'params': self.kd_proj2.parameters()}]
        self.old_student = None
        self.counter = 0

    def to_device(self):
        devices = list(range(torch.cuda.device_count()))
        self.teacher = nn.DataParallel(self.teacher, device_ids=devices).to("cuda:0")
        self.student = nn.DataParallel(self.student, device_ids=devices).to("cuda:0")
        self.kd_proj0 = nn.DataParallel(self.kd_proj0, device_ids=devices).to("cuda:0")
        self.kd_proj1 = nn.DataParallel(self.kd_proj1, device_ids=devices).to("cuda:0")
        self.kd_proj2 = nn.DataParallel(self.kd_proj2, device_ids=devices).to("cuda:0")
        self.simclr_proj = nn.DataParallel(self.simclr_proj, device_ids=devices).to("cuda:0")

    def prepare_dataloader(self, train_aug="strong_strong_strong", test_aug="test_test_test"):
        transforms_train = data_loader.TransformLoader(self.args.img_size).get_composed_transform(aug=train_aug)
        dataset_ins = getattr(dataset, self.dataset_name)(transforms_train, split=self.args.target_subset_split)

        transforms_test = data_loader.TransformLoader(self.args.img_size).get_composed_transform(aug=test_aug)
        dataset_copy = getattr(dataset, self.dataset_name)(transforms_test, split=self.args.target_subset_split)

        if self.args.ratio == 1.:
            self.trainloader = torch.utils.data.DataLoader(
                dataset_ins,
                batch_size=self.args.bsize,
                num_workers=self.args.num_workers,
                shuffle=True,
                drop_last=True)
            ind = torch.randperm(len(dataset_ins))
            self.validloader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(dataset_copy, ind[int(0.1 * len(ind)):]),
                batch_size=self.args.bsize,
                num_workers=self.args.num_workers,
                shuffle=False,
                drop_last=True)
        else:
            ind = torch.randperm(len(dataset_ins))
            self.trainloader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(dataset_ins, ind[: int(self.args.ratio * len(ind))]),
                batch_size=self.args.bsize,
                num_workers=self.args.num_workers,
                shuffle=True,
                drop_last=True)

            self.validloader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(dataset_copy, ind[int(self.args.ratio * len(ind)):]),
                batch_size=self.args.bsize,
                num_workers=self.args.num_workers,
                shuffle=False,
                drop_last=True)

    def load_teacher(self, backbone_name):
        if backbone_name == "resnet10":
            backbone = models.ResNet10()
            self.feature_dim = backbone.final_feat_dim
            if self.feature_dim != 512:
                raise ValueError("The feature dim is not 512, something wrong with it")
        else:
            raise ValueError("{} backbone is not supported , temporily".format(backbone_name))

        if self.args.teacher_path is None:
            raise ValueError("No teacher pretraining model path!")
        if self.args.teacher_path_version == 0:
            state = torch.load(self.args.teacher_path)["state"]
            state_keys = list(state)
            for key in state_keys:
                if "feature." in key:
                    newkey = key.replace("feature.", "")
                    state[newkey] = state.pop(key)

                else:
                    state.pop(key)
            state_dict = copy.deepcopy(state)
        elif self.args.teacher_path_version == 1:
            state_dict = torch.load(self.args.teacher_path)["model"]

        else:
            raise ValueError("Invalid teacher path version!!!")
        backbone.load_state_dict(state_dict)
        self.teacher = copy.deepcopy(backbone)
        for param in self.teacher.parameters():
            param.requires_grad = False

    def load_student(self, backbone_name):
        if backbone_name == "resnet10":
            backbone = models.ResNet10()
            self.feature_dim = backbone.final_feat_dim
            if self.feature_dim != 512:
                raise ValueError("feature dim is not 512, something wrong with it")
        else:
            raise ValueError("{} backbone is not supported , temporily".format(backbone_name))

        if self.args.student_path is None:
            self.student = copy.deepcopy(backbone)
            print("Train student from scratch")
            return None
        if self.args.student_path_version == 0:
            state = torch.load(self.args.student_path)["state"]
            state_keys = list(state)
            for key in state_keys:
                if "feature." in key:
                    newkey = key.replace("feature.", "")
                    state[newkey] = state.pop(key)

                else:
                    state.pop(key)
            state_dict = copy.deepcopy(state)
        elif self.args.student_path_version == 1:
            state_dict = torch.load(self.args.student_path)["model"]

        else:
            raise ValueError("Invalid student path version!!!")
        backbone.load_state_dict(state_dict)
        self.student = copy.deepcopy(backbone)


    def kd_group_loss(self, fs, ft, x3, epoch = 0):
        if epoch == 0:
            loss_kd0 = F.mse_loss(self.kd_proj0(fs[0]), ft[0].detach())
            loss_kd1 = F.mse_loss(self.kd_proj1(fs[1]), ft[1].detach())
            loss_kd2 = F.mse_loss(self.kd_proj2(fs[2]), ft[2].detach())
        else:
            momentum = self.args.coef * self.counter / (self.args.epochs * len(self.trainloader))
            self.old_student.eval()
            with torch.no_grad():
                f1_old_map, _ = self.old_student(x3, ret_layers=[5, 6, 7])
                ft0 = (1 - momentum)*ft[0].detach() + momentum * f1_old_map[0].detach()
                ft1 = (1 - momentum)*ft[1].detach() + momentum * f1_old_map[1].detach()
                ft2 = (1 - momentum)*ft[2].detach() + momentum * f1_old_map[2].detach()
            loss_kd0 = F.mse_loss(self.kd_proj0(fs[0]), ft0.detach())
            loss_kd1 = F.mse_loss(self.kd_proj1(fs[1]), ft1.detach())
            loss_kd2 = F.mse_loss(self.kd_proj2(fs[2]), ft2.detach())
            self.counter += 1
        return loss_kd0 + loss_kd1 + loss_kd2

    def train_loop(self, epoch, simclr_criterion=None, optimizer=None):
        self.switch_mode(module_list=[self.student, self.simclr_proj, self.kd_proj0, self.kd_proj1, self.kd_proj2], mode="train")
        self.switch_mode(module_list=[self.teacher], mode="eval")
        loss_avg, loss_sim_avg, loss_kd_avg = 0, 0, 0

        # print("======================= Begin to train =================")
        for i, ((x1, x2, x3), _) in enumerate(self.trainloader):
            x1 = x1.to("cuda:0")
            x2 = x2.to("cuda:0")
            x3 = x3.to("cuda:0")

            optimizer.zero_grad()

            f1_stu_map, f1_stu_final = self.student(x1, ret_layers = [4, 5, 6])
            f2_stu_map, f2_stu_final = self.student(x2, ret_layers = [4, 5, 6])

            z1_stu = self.simclr_proj(f1_stu_final)
            z2_stu = self.simclr_proj(f2_stu_final)
            loss_sim = simclr_criterion(z1_stu, z2_stu)

            with torch.no_grad():
                f1_tea_map, _ = self.teacher(x1, ret_layers = [5, 6, 7])
                f2_tea_map, _ = self.teacher(x2, ret_layers = [5, 6, 7])
            loss_kd = self.kd_group_loss(f1_stu_map, f1_tea_map, x3, epoch = epoch, kd_criterion=kd_criterion)
            #
            loss = loss_sim + self.alpha * loss_kd

            #loss = loss_sim
            loss.backward()
            optimizer.step()
            loss_avg += loss.item()
            loss_sim_avg += loss_sim.item()
            loss_kd_avg += (self.alpha * loss_kd.item())


            self.old_student = copy.deepcopy(self.student)
            self.old_student.eval()
            for param in self.old_student.parameters():
                param.requires_grad = False

        print("Total loss: {} | SimCLR loss: {} | KD loss: {}".format(loss_avg / i, loss_sim_avg / i, loss_kd_avg / i))


    def validate(self, epoch, simclr_criterion=None, kd_criterion = None):
        self.switch_mode([self.student, self.teacher, self.simclr_proj, self.kd_proj0, self.kd_proj1, self.kd_proj2],mode="eval")
        total_loss = 0
        with torch.no_grad():
            for i, ((x1, x2, x3), _) in enumerate(self.validloader):
                ### Step 1. Load data from source domain and target domain
                x1 = x1.to("cuda:0")
                x2 = x2.to("cuda:0")
                x3 = x3.to("cuda:0")

                ### Step 3. Calsulate student loss
                f1_stu_map, f1_stu_final = self.student(x1, ret_layers=[4, 5, 6])
                f2_stu_map, f2_stu_final = self.student(x2, ret_layers=[4, 5, 6])

                z1_stu = self.simclr_proj(f1_stu_final)
                z2_stu = self.simclr_proj(f2_stu_final)
                loss_sim = simclr_criterion(z1_stu, z2_stu)

                f1_tea_map, _ = self.teacher(x1, ret_layers=[5, 6, 7])
                f2_tea_map, _ = self.teacher(x2, ret_layers=[5, 6, 7])

                loss_kd = self.kd_group_loss(f1_stu_map, f1_tea_map, x3, epoch=epoch, kd_criterion = kd_criterion) + self.kd_group_loss(f2_stu_map, f2_tea_map, x3, epoch=epoch, kd_criterion = kd_criterion)
                loss = loss_sim + self.alpha * loss_kd

                total_loss += loss.item()
        return total_loss 

    def switch_mode(self, module_list=[], mode="train"):
        if mode == "train":
            for module in module_list:
                module.train()

        if mode == "test" or mode == "eval":
            for module in module_list:
                module.eval()

    def trainable_modules(self):
        modules = [{'params': self.student.parameters()},
                   {'params': self.simclr_proj.parameters()},
                   {'params': self.simclr_proj_teacher.parameters()},
                   {'params': self.clf.parameters()},
                   {'params': self.teacher.parameters()}]
        return modules
