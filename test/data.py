

import torch
import tensorflow_datasets as tfds
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
smallsize = 500
rootdir = "/home/philipkenneweg/Documents/No-learning-rates-needed-Introducing-SALSA-Stable-Armijo-Line-Search-Adaptation/data"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
task_list = ["cola","colasmall","sst2", "sst2smallunbalanced","sst2small", "mrpcsmall", "mrpc", "qnli", "qnlismall", "mnli", "mnlismall"]#,"qnli"]
def load_data(name="sst2"):
    
    split = "train"
    if "small" in name:
        split = split + "[:" + str(smallsize) +"]"
#         X = X[:smallsize]
#         y = y[:smallsize]

#     if name not in task_list:
#         print("dataset not suported")
   
    if "sst2" in name:
        data = tfds.load('glue/sst2', split=split, shuffle_files=False)
        
        X = [str(e["sentence"].numpy()) for e in data]
        y = [int(e["label"]) for e in data]
    
        data = tfds.load('glue/sst2', split="validation", shuffle_files=False)
        X_val = [str(e["sentence"].numpy()) for e in data]
        y_val = [int(e["label"]) for e in data]
        
        data = tfds.load('glue/sst2', split="test", shuffle_files=False)
        X_test = [str(e["sentence"].numpy()) for e in data]
        y_test = [int(e["label"]) for e in data]
    elif "cola" in name:
        data = tfds.load('glue/cola', split=split, shuffle_files=False)
        
        X = [str(e["sentence"].numpy()) for e in data]
        y = [int(e["label"]) for e in data]
    
        data = tfds.load('glue/cola', split="validation", shuffle_files=False)
        X_val = [str(e["sentence"].numpy()) for e in data]
        y_val = [int(e["label"]) for e in data]
        
        #test data for cola was garbage
        X, X_test, y, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
    elif "mrpc" in name:
        data = tfds.load('glue/mrpc', split=split, shuffle_files=True)
        
        X = [str(e["sentence1"].numpy()) for e in data]
        X2 = [str(e["sentence2"].numpy()) for e in data]
        X = list(zip(X,X2))
        y = [int(e["label"]) for e in data]
    
        data = tfds.load('glue/mrpc', split="validation", shuffle_files=False)
        X_val = [str(e["sentence1"].numpy()) for e in data]
        X2_val = [str(e["sentence2"].numpy()) for e in data]
        X_val = list(zip(X_val,X2_val))
        y_val = [int(e["label"]) for e in data]

        
        data = tfds.load('glue/mrpc', split="test", shuffle_files=False)
        X_test = [str(e["sentence1"].numpy()) for e in data]
        X2_test = [str(e["sentence2"].numpy()) for e in data]
        X_test = list(zip(X_test,X2_test))
        y_test = [int(e["label"]) for e in data] #test labels are garbage
        
    elif "qnli" in name:
        data = tfds.load('glue/qnli', split=split, shuffle_files=False)
        
        X = [str(e["question"].numpy()) for e in data]
        X2 = [str(e["sentence"].numpy()) for e in data]
        X = list(zip(X,X2))
        y = [int(e["label"]) for e in data]
       # print(X)
    
        data = tfds.load('glue/qnli', split="validation", shuffle_files=False)
        X_val = [str(e["question"].numpy()) for e in data]
        X2_val = [str(e["sentence"].numpy()) for e in data]
        X_val = list(zip(X_val,X2_val))
        y_val = [int(e["label"]) for e in data]

        
        data = tfds.load('glue/qnli', split="test", shuffle_files=False)
        X_test = [str(e["question"].numpy()) for e in data]
        X2_test = [str(e["sentence"].numpy()) for e in data]
        X_test = list(zip(X_test,X2_test))
        y_test = [int(e["label"]) for e in data]
        
        
    elif "rte" in name:
        data = tfds.load('glue/rte', split=split, shuffle_files=False)
        
        X = [str(e["sentence1"].numpy()) for e in data]
        X2 = [str(e["sentence2"].numpy()) for e in data]
        X = list(zip(X,X2))
        y = [int(e["label"]) for e in data]
       # print(X)
    
        data = tfds.load('glue/rte', split="validation", shuffle_files=False)
        X_val = [str(e["sentence1"].numpy()) for e in data]
        X2_val = [str(e["sentence2"].numpy()) for e in data]
        X_val = list(zip(X_val,X2_val))
        y_val = [int(e["label"]) for e in data]

        
        data = tfds.load('glue/rte', split="test", shuffle_files=False)
        X_test = [str(e["sentence1"].numpy()) for e in data]
        X2_test = [str(e["sentence2"].numpy()) for e in data]
        X_test = list(zip(X_test,X2_test))
        y_test = [int(e["label"]) for e in data]
        
    elif "qqp" in name:
        data = tfds.load('glue/qqp', split=split, shuffle_files=False)
        
        X = [str(e["question1"].numpy()) for e in data]
        X2 = [str(e["question2"].numpy()) for e in data]
        X = list(zip(X,X2))
        y = [int(e["label"]) for e in data]
       # print(X)
    
        data = tfds.load('glue/qqp', split="validation[:10000]", shuffle_files=False)
        X_val = [str(e["question1"].numpy()) for e in data]
        X2_val = [str(e["question2"].numpy()) for e in data]
        X_val = list(zip(X_val,X2_val))
        y_val = [int(e["label"]) for e in data]

        
        data = tfds.load('glue/qqp', split="test[:1000]", shuffle_files=False)
        X_test = [str(e["question1"].numpy()) for e in data]
        X2_test = [str(e["question2"].numpy()) for e in data]
        X_test = list(zip(X_test,X2_test))
        y_test = [int(e["label"]) for e in data]
        
        

        
    elif "mnli" in name:
        data = tfds.load('glue/mnli', split=split, shuffle_files=False)

        X = [str(e["premise"].numpy()) for e in data]
        X2 = [str(e["hypothesis"].numpy())for e in data] 
        X = list(zip(X,X2))
        y = [int(e["label"]) for e in data]
    
        data = tfds.load('glue/mnli', split="validation_matched", shuffle_files=False)
        X_val = [str(e["premise"].numpy()) for e in data]
        X2_val = [str(e["hypothesis"].numpy())  for e in data]
        X_val = list(zip(X_val,X2_val))
        y_val = [int(e["label"]) for e in data]

        
        data = tfds.load('glue/mnli', split="test_matched", shuffle_files=False)
        X_test = [str(e["premise"].numpy()) for e in data]
        X2_test = [str(e["hypothesis"].numpy())  for e in data]
        X_test = list(zip(X_test,X2_test))
        y_test = [int(e["label"]) for e in data]
        
    

        
    if "unbalanced" in name:
        ids = []
        max_id = -1
        max_id_num = 10000000
        for a,unique in enumerate(np.unique(y)):
            id =[]
            for i,point in enumerate(y):
                if point == unique:
                    id.append(i)
            ids.append(id)
            if len(id)< max_id_num:
                max_id = a
        ids[max_id] = ids[max_id][:round(len(ids[max_id])*0.2)]
        new_ids = []
        for id in ids:
            for p in id:
                new_ids.append(p)
        np.random.shuffle(new_ids)
        y = np.asarray(y)
        X = np.asarray(X)
        y = list(y[new_ids])
        X = list(X[new_ids])
        

#     print("validation: ",X_val[0:2])
#     print(y_val[0:2])
#     print("train", X[0:2])
#     print(y[0:2])
#     print("test" , X_test[0:2])
#     print(y_test[0:2])
    return X,X_val, X_test, torch.LongTensor(y), torch.LongTensor(y_val), torch.LongTensor(y_test)

def load_wiki():
    data = tfds.load('wiki40b/en', split="train[:25000]", shuffle_files=False)
    
    X = [str(e["text"].numpy()) for e in data]
    return X

def load_wikiandbook(batch_size):
    from datasets import concatenate_datasets, load_dataset

    bookcorpus = load_dataset("bookcorpus", split="train")
    wiki = load_dataset("wikipedia", "20220301.en", split="train")
    wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])  # only keep the 'text' column

    assert bookcorpus.features.type == wiki.features.type
    raw_datasets = concatenate_datasets([bookcorpus, wiki]) #ds is 5.000.000 *16 examples big
    def batch_iterator(batch_size=batch_size):
        while True:
            for i in range(0, len(raw_datasets), batch_size):
                yield raw_datasets[i : i + batch_size]["text"]

    return batch_iterator()


from torch.utils.data import Dataset
import torchvision

def getCifar10(batch_size):
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=rootdir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR10(
        root=rootdir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

def getCifar100(batch_size):
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root=rootdir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR100(
        root=rootdir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

def getImageNet(batch_size):
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
        torchvision.transforms.RandomCrop((224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    transform_test = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True),
        torchvision.transforms.CenterCrop((224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    trainset = torchvision.datasets.ImageNet(
        root=rootdir, split = "train", transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True)

    testset = torchvision.datasets.ImageNet(
        root=rootdir, split= "val", transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True)
    return trainloader, testloader
    
class SimpleDataset(Dataset):
    def __init__(self, data, dataname, labelname, resize = True):
        self.resize = resize
        self.data = data
        self.dataname = dataname
        self.labelname = labelname
        self.transforms = torchvision.transforms.ToTensor()
        self.resize = torchvision.transforms.Resize((232), interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        self.center_crops = torchvision.transforms.CenterCrop((224,224))
        self.normalize = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch_x = self.data[idx][self.dataname]
        batch_x = self.transforms(batch_x)
        if self.resize:
            if batch_x.shape[0] == 1:
                batch_x = batch_x.squeeze()
                batch_x = torch.stack([batch_x,batch_x,batch_x], axis = 0)
            if batch_x.shape[0] == 4:
                batch_x = batch_x[:3,:,:]
            batch_x = self.resize(batch_x.to(device))
            batch_x = self.center_crops(batch_x)
            if torch.max(batch_x[:,:10,:10]) > 2: #check if image is float or integer
                batch_x = batch_x/255.0
            batch_x = self.normalize(batch_x)
        else:
            batch_x = batch_x.float().to(device)/255.0
      #  plt.imshow(batch_x.permute(1, 2, 0).cpu().numpy())
      #  plt.show()
        batch_y = torch.LongTensor([self.data[idx][self.labelname]]).squeeze().to(device)
      #  print(batch_y)
        return batch_x ,batch_y

from datasets import load_dataset
class SimpleDataset_electric(Dataset):
    def __init__(self, data):
        self.data = data 
        self.categorical = ["DOWN", "UP"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch_y = self.data[idx]["class"]
        batch_y = 1 if batch_y == self.categorical[0] else 0
        batch_x = self.data[idx]
        del batch_x["class"]
        batch_x = np.asarray(list(batch_x.values()))
        batch_x = torch.FloatTensor(batch_x).to(device)
        batch_y = torch.LongTensor([batch_y]).squeeze().to(device)
        return batch_x ,batch_y

class SimpleDataset_cover(Dataset):
    def __init__(self, data):
        self.data = data 
        self.categorical = [1, 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch_y = self.data[idx]["Y"]
        batch_y = 1 if batch_y == self.categorical[0] else 0
        batch_x = self.data[idx]
        del batch_x["Y"]
        batch_x = np.asarray(list(batch_x.values()))
        batch_x = torch.FloatTensor(batch_x).to(device)
        batch_y = torch.LongTensor([batch_y]).squeeze().to(device)
        return batch_x ,batch_y

class SimpleDataset_pol(Dataset):
    def __init__(self, data):
        self.data = data 
        self.categorical = ['P', 'N']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
       # print(self.data[idx])
        batch_y = self.data[idx]["binaryClass"]
        batch_y = 1 if batch_y == self.categorical[0] else 0
        batch_x = self.data[idx]
        del batch_x["binaryClass"]
        batch_x = np.asarray(list(batch_x.values()))
        batch_x = torch.FloatTensor(batch_x).to(device)
        batch_y = torch.LongTensor([batch_y]).squeeze().to(device)
      #  print(batch_x, batch_y)
        return batch_x ,batch_y
