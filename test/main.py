import argparse
import os
import datetime as dt
import configparser
from logger import Logger
import shutil
import sys
from train import train
from train_img import train_img
import torch
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(config_file = None):
    if config_file == None:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
        parser.add_argument('--config_file', help='path_to_config_file', type=str, default="config.json")
        parser.add_argument('--train_type', help='cls or mlm', type=str, default="cls")
        # parser.add_argument('--optimizer', help='optimizer to use either adam or adamsls', type=str, default="adam")
        # parser.add_argument('--dataset', help='dataset to use either mrpc, s', type=str, default="adam")
        # parser.add_argument('--num_diff_opt', help='number of layers to optimize', type=int, default=1)

        args = parser.parse_args()
        config_file = args.config_file
    else:
        from argparse import Namespace
        args = Namespace()
        args.train_type = "cls"
    config = configparser.ConfigParser()
    config.sections()
    config.read(config_file)
    print("device is", device)
    if config["DEFAULT"]["directory"] == "default":
        config["DEFAULT"]["directory"] = "results/" + dt.datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
    
    
    os.makedirs(config["DEFAULT"]["directory"], exist_ok = True)
    print(config["DEFAULT"]["directory"] )
    
    for file in os.listdir(os.getcwd()):
        if ".py" in file or ".json" in file:
            shutil.copy2(file, config["DEFAULT"]["directory"] )
            
    sys.stdout = Logger(open(config["DEFAULT"]["directory"] +"/SysOut.txt","w"))
    if config["DEFAULT"]["type"]  == "NLP":
        train(args, config)
    if config["DEFAULT"]["type"]  == "img":
        train_img(args, config)
    
    
if __name__ == '__main__':
    main()