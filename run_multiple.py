
from main import main

datasets = ["sst2", "mrpc", "qnli", "mnli"]
models = ["bert"]#, "roberta"]
optim = ["salsasls","adamsls","adam", "sgdsls", "sgd", "oladamsls", "olsgdsls"]
numexp = 5
batch_size = [32]
cs = [0.3]
betas = [0.99]
speed_up = [True]

def create_config(name, ds, model, opt,split = "layer", n_opt= 1, update_r = "cycle", i = 0, combine = 0, batch_size = 32, c = 0.1, cls = "transformer", beta = 0.99, onlygradsmooth = False, speed_up = False):
    with open(name, 'w') as f:
            f.write("[DEFAULT]\n")
            f.write("batch_size = "+ str(batch_size) +"\n")
            f.write("checkpoint = None\n")
            f.write("directory = results/"  + ds + opt + str(n_opt) + model + split + update_r +str(combine)+ str(i) + "\n")
            f.write("seed = 42\n")
            f.write("epochs = 5\n")
            f.write("dataset = " + ds + "\n")
            f.write("optim = " + opt + "\n")
            f.write("num_diff_opt =" + str(n_opt) + "\n")
            f.write("model = " + model + "\n")
            f.write("split_by = " + split + "\n")
            f.write("update_rule = " + update_r + "\n")
            f.write("combine = " + str(combine) + "\n")
            f.write("c = " + str(c) + "\n")
            f.write("cls = " + cls + "\n")
            f.write("type = " + "NLP" + "\n")
            f.write("beta = "+ str(beta)+ "\n")
            f.write("speed_up = "+ str(speed_up)+ "\n")
   # print("results/"  +ds + opt+ str(n_opt) + model + split )
    main(name)

for ds in datasets:
    for bs in batch_size:
        for model in models:
            for opt in optim:
                if "sls" in opt:
                        for beta in betas:
                            for s_up in speed_up:
                                    for c in cs:
                                        for i in range(numexp):
                                            create_config("config_gen.json", ds,  model , opt, i=i,batch_size = bs, c = c, beta = beta, speed_up= s_up)
                else:
                    for i in range(numexp):
                        create_config("config_gen.json", ds,  model , opt, i=i,batch_size = bs)


            
            
            
            
            
            
            
           
            
            


