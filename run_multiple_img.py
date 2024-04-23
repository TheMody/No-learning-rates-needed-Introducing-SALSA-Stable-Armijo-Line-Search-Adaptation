
from main import main

datasets = ["cifar10", "cifar100", "imagenet",]
models = [ "resNet34",  "resNet50"]
optim = [ "salsasls","adamsls","oladamsls","adam", "sgd", "sgdsls", "olsgdsls"]
numexp = 1
batch_size = [128]
cs = [0.3]
epochs = [1]
betas = [0.99]

def create_config(name, ds,  model, opt, update_r = "cycle",n_opt = 1, split = "layer", i = 0, combine = 0, batch_size = 32, c = 0.1, epochs = 5, cls = "cnn", n_hid = 1, beta = 0.99):
    with open(name, 'w') as f:
            f.write("[DEFAULT]\n")
            f.write("batch_size = "+ str(batch_size) +"\n")
            f.write("checkpoint = None\n")
            f.write("directory = results/"  + ds + opt + str(n_opt) + model + split + update_r +str(combine)+ str(i) + "\n")
            f.write("seed = 42\n")
            f.write("epochs = "+str(epochs)+"\n")
            f.write("dataset = " + ds + "\n")
            f.write("optim = " + opt + "\n")
            f.write("num_diff_opt =" + str(n_opt) + "\n")
            f.write("model = " + model + "\n")
            f.write("split_by = " + split + "\n")
            f.write("update_rule = " + update_r + "\n")
            f.write("combine = " + str(combine) + "\n")
            f.write("c = " + str(c) + "\n")
            f.write("cls = " + cls + "\n")
            f.write("type = " + "img" + "\n")
            f.write("num_hidden_dims = " + str(n_hid) + "\n")
            f.write("beta = " + str(beta) + "\n")

    main(name)

for ds in datasets:
    for model in models:
        for opt in optim:
            for e in epochs:
                for bs in batch_size:
                    if "sls" in opt:
                        for beta in betas:
                            for c in cs:
                                for i in range(numexp):
                                    create_config("config_gen.json", ds,  model = model , opt = opt, i=i, batch_size = bs, c = c, epochs = e, beta= beta)
                    else:
                        for i in range(numexp):
                            create_config("config_gen.json", ds, "layer", 1, model , opt,"cycle", i,  batch_size = bs, epochs = e)



            
            
            
            
            
            
            
           
            
            


