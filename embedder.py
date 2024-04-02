import torch
import torch.nn as nn
import torch.optim as optim
import math
import time
import numpy as np
from transformers.utils import logging
from sls.adam_sls import AdamSLS
import wandb
from cosine_scheduler import CosineWarmupScheduler
logging.set_verbosity_error()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NLP_embedder(nn.Module):

    def __init__(self,  num_classes, batch_size, args):
        super(NLP_embedder, self).__init__()
        self.batch_size = batch_size
        self.padding = True
        self.num_classes = num_classes
        self.lasthiddenstate = 0
        self.args = args

        if args.model == "bert":
            from transformers import BertTokenizer, BertModel
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased')
            self.output_length = 768
        if args.model == "roberta":
            from transformers import RobertaTokenizer, RobertaModel
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.model = RobertaModel.from_pretrained('roberta-base')
            self.output_length = 768

        
        self.fc1 = nn.Linear(self.output_length,self.num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        

        if args.number_of_diff_lrs > 1:
            pparamalist = []
            for i in range(args.number_of_diff_lrs):
                paramlist = []
                optrangelower = math.ceil((12.0/(args.number_of_diff_lrs-2)) *(i-1))
                optrangeupper = math.ceil((12.0/(args.number_of_diff_lrs-2)) * (i))
                
                optrange = list(range(optrangelower,optrangeupper))
                if i == 0 or i == args.number_of_diff_lrs-1:
                    optrange =[]
                for name,param in self.named_parameters():
                    if "encoder.layer." in name:
                        included = False
                        for number in optrange:
                            if "." +str(number)+"." in name:
                                included = True
                        if included:
                            paramlist.append(param)
                            #  print("included", name , "in", i)
                    else:
                        if "embeddings." in name:
                            if i == 0:
                                paramlist.append(param)
                                #   print("included", name , "in", i)
                        else:
                            if i == args.number_of_diff_lrs-1 and not "pooler" in name:
                                paramlist.append(param)
                                #  print("included", name , "in", i)
                                #  print(name, param.requires_grad, param.grad)
                pparamalist.append(paramlist)
            if args.opts["opt"] == "adamsls":  
                self.optimizer = AdamSLS(pparamalist,strategy = args.update_rule , combine_threshold = args.combine, c = self.args.c)
            if args.opts["opt"] == "sgdsls":  
                self.optimizer = AdamSLS( pparamalist,strategy = args.update_rule, combine_threshold = args.combine, base_opt = "scalar",gv_option = "scalar" , c = self.args.c)
                #   self.optimizer.append(SgdSLS(pparamalist ))
        else:
            if args.opts["opt"] == "adam":    
                self.optimizer = optim.Adam(self.parameters(), lr=args.opts["lr"] )
            if args.opts["opt"] == "radam":    
                self.optimizer = optim.RAdam(self.parameters(), lr=args.opts["lr"] )
            if args.opts["opt"] == "sgd":    
                self.optimizer = optim.SGD(self.parameters(), lr=args.opts["lr"] )
            if args.opts["opt"] == "adamsls":    
                self.optimizer = AdamSLS( [[param for name,param in self.named_parameters() if not "pooler" in name]] , c = self.args.c, beta_s = self.args.beta, speed_up=self.args.speed_up)
            if args.opts["opt"] == "oladamsls":    
                self.optimizer = AdamSLS( [[param for name,param in self.named_parameters() if not "pooler" in name]] , c = 0.1, smooth = False)
            if args.opts["opt"] == "olsgdsls":    
                self.optimizer = AdamSLS( [[param for name,param in self.named_parameters() if not "pooler" in name]] , c = 0.1, base_opt = "scalar",gv_option = "scalar", smooth = False)
            if args.opts["opt"] == "sgdsls":    
                self.optimizer = AdamSLS( [[param for name,param in self.named_parameters() if not "pooler" in name]], base_opt = "scalar",gv_option = "scalar", c = self.args.c , beta_s = self.args.beta)


            
        
    def forward(self, x_in):
        x = self.model(**x_in).last_hidden_state
        x = x[:, self.lasthiddenstate]
        x = self.fc1(x)
        return x
    
    
     
    def fit(self, x, y, epochs=1, X_val= None,Y_val= None):
        wandb.init(project="suplementary"+self.args.ds, name = self.args.split_by + "_" + self.args.opts["opt"] + "_" + self.args.model +
            "_" + str(self.args.number_of_diff_lrs) +"_"+ self.args.savepth, entity="pkenneweg", 
            group = "suplementary"+str(self.args.speed_up)+ self.args.opts["opt"] + "_" + self.args.model +"_" + str(self.args.number_of_diff_lrs) + self.args.update_rule 
            + str(self.args.combine)+"bs"+ str(self.batch_size) +"c"+ str(self.args.c)+"beta"+ str(self.args.beta))
        #wandb.watch(self)
        
        self.mode = "cls"
        if (not "sls" in  self.args.opts["opt"]):
            self.scheduler= CosineWarmupScheduler(optimizer= self.optimizer, 
                                                warmup = math.ceil(len(x)*epochs *0.1 / self.batch_size) ,
                                                    max_iters = math.ceil(len(x)*epochs  / self.batch_size))
        


        accuracy = None
        accsteps = 0
        accloss = 0
        for e in range(epochs):
            start = time.time()
            for i in range(math.ceil(len(x) / self.batch_size)):
            
                startsteptime = time.time()
                ul = min((i+1) * self.batch_size, len(x))
                batch_x = x[i*self.batch_size: ul]
                batch_y = y[i*self.batch_size: ul]
                batch_x = self.tokenizer(batch_x, return_tensors="pt", padding=self.padding, max_length = 256, truncation = True)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                batch_y = batch_y.to(device)
                batch_x = batch_x.to(device)

                if "sls" in  self.args.opts["opt"]:
                    closure = lambda : self.criterion(self(batch_x), batch_y)
                    self.optimizer.zero_grad()
                    loss = self.optimizer.step(closure = closure)

                else:
                    self.optimizer.zero_grad()
                    y_pred = self(batch_x)
                    loss = self.criterion(y_pred, batch_y)    
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()       

                if (i*self.batch_size) % 32 == 0:
                    dict = {"loss": loss.item() , "time_per_step":time.time()-startsteptime}
                    if "sls" in  self.args.opts["opt"]:
                        dict["ls_freq"] = self.optimizer.state["LS_freq"]
                        for a,step_size in enumerate( self.optimizer.state['step_sizes']):
                            dict["step_size"+str(a)] = step_size
                            dict["loss_decrease"] = self.optimizer.state["loss_decrease"]
                    else:
                        dict["step_size"+str(0)] = self.scheduler.get_last_lr()[0]
                    wandb.log(dict)
                    accloss = accloss + loss.item()
                    accsteps += 1
                    if i % np.max((1,int((len(x)/self.batch_size)*0.1))) == 0:
                        print(i, accloss/ accsteps)
                        accsteps = 0
                        accloss = 0

            if X_val != None:
                with torch.no_grad():
                    accuracy = self.evaluate(X_val, Y_val).item()
                    print("accuracy after", e, "epochs:",accuracy, "time per epoch", time.time()-start)
                    wandb.log({"accuracy": accuracy})
            else:
                print("epoch",e,"time per epoch", time.time()-start)
                
                
        wandb.finish()
        return accuracy
        
    @torch.no_grad()
    def evaluate(self, X,Y):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Y = Y.to(device)
        y_pred = self.predict(X)
        accuracy = torch.sum(Y == y_pred)
        accuracy = accuracy/Y.shape[0]
        return accuracy
  
    
    def predict(self, x):
        resultx = None

        for i in range(math.ceil(len(x) / self.batch_size)):
            ul = min((i+1) * self.batch_size, len(x))
            batch_x = x[i*self.batch_size: ul]
            batch_x = self.tokenizer(batch_x, return_tensors="pt", padding=self.padding, max_length = 256, truncation = True)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            batch_x = batch_x.to(device)
            batch_x = self(batch_x)
            if resultx is None:
                resultx = batch_x.detach()
            else:
                resultx = torch.cat((resultx, batch_x.detach()))

     #   resultx = resultx.detach()
        return torch.argmax(resultx, dim = 1)

    
    

        
        

