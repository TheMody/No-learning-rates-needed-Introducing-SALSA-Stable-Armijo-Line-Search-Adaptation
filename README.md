# No learning rates needed: Introducing SALSA - Stable ArmijoLine Search Adaptation

The Repository to the Paper Faster Convergence for No learning rates needed: Introducing SALSA - Stable ArmijoLine Search Adaptation

## Install

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3

for replication:
- `pip install transformers` for huggingface transformers <3 
- `pip install datasets` for huggingface datasets <3 
- `pip install tensorflow-datasets` for tensorflow datasets <3 
- `pip install wandb` for optional logging <3
- for easy replication use conda and environment.yml eg:
`$ conda env create -f environment.yml` and `$ conda activate sls3`



## Use in own projects

The custom optimizer is in \sls\SaLSA.py and the comparison version are in adam_sls.py
Example Usage:

```
from sls.SaLSA import SaLSA
self.optimizer = SaLSA( model.parameters())
```


The typical pytorch forward pass needs to be changed from :
``` 
optimizer.zero_grad()
y_pred = model(batch_x)
loss = criterion(y_pred, batch_y)    
loss.backward()
optimizer.step()
scheduler.step() 
```
to:
``` 
def closure(backwards = False):
    y_pred = self(batch_x)
    loss = self.criterion(y_pred, batch_y)
    if backwards:
        loss.backward()
    return loss
self.optimizer.zero_grad()
loss = self.optimizer.step(closure = closure)
```

This code change is necessary since, the optimizers needs to perform additional forward passes and thus needs to have the forward pass encapsulated in a function.
see embedder.py in the fit() method for more details


## Replicating Results
For replicating the main Results of the Paper run:

```
$ python run_multiple.py
$ python run_multiple_img.py
```


For replicating specific runs or trying out different hyperparameters use:

```
$ python main.py 
```

and change the config.json file appropriately


## Please cite:
No learning rates needed: Introducing SALSA - Stable ArmijoLine Search Adaptation
from 
Philip Kenneweg,
Tristan Kenneweg,
Fabian Fumagalli
Barbara Hammer
published in IJCNN 2024

