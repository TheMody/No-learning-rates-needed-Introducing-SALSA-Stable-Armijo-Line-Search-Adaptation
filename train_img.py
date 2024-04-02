
from datasets import load_dataset
from image_classifier import Image_trainer
import torch
from torch.utils.data import DataLoader
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_img(args,config):
    max_epochs = int(config["DEFAULT"]["epochs"])

    batch_size = int(config["DEFAULT"]["batch_size"])

    dataset = config["DEFAULT"]["dataset"]
    train_data = None
    args.number_of_diff_lrs = int(config["DEFAULT"]["num_diff_opt"])
    if config["DEFAULT"]["optim"] == "sgd":
      lr = 1e-1
    else:
      lr = 1e-3
    args.opts = {"lr": lr, "opt": config["DEFAULT"]["optim"]}
    args.ds = dataset
    args.split_by = config["DEFAULT"]["split_by"]
    args.update_rule = config["DEFAULT"]["update_rule"]
    args.model = config["DEFAULT"]["model"]
    args.savepth = config["DEFAULT"]["directory"]
    args.combine = float(config["DEFAULT"]["combine"])
    args.c = float(config["DEFAULT"]["c"])
    args.beta = float(config["DEFAULT"]["beta"])
    args.hidden_dims = int(config["DEFAULT"]["num_hidden_dims"])

    if dataset == "imagenet":
      num_classes = 1000
    if dataset == "cifar100":
      num_classes = 100
    if dataset == "cifar10":
      num_classes = 10



    img_cls = Image_trainer(num_classes,batch_size,args)
    

    if dataset == "cifar10":
      from data import getCifar10
      train_dataloader,val_dataloader = getCifar10(batch_size)
    elif dataset == "cifar100":
      from data import getCifar100
      train_dataloader,val_dataloader = getCifar100(batch_size)
    elif dataset == "imagenet":
      from data import getImageNet
      train_dataloader,val_dataloader = getImageNet(batch_size)


    img_cls.fit(train_dataloader,max_epochs,val_dataloader)
