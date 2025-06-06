# SALSA - Optimizer

The official Repository to the Paper "No learning rates needed: Introducing SALSA - Stable Armijo Line Search Adaptation". With additional features.
If you have any question, remarks or issues with the SALSA-Optimizer please do not hesitate to contact us on Github.


## Install

```
pip install SaLSa-Optimizer
```
or clone the repo and use

```
pip install .
```
## Use:

Example Usage:

```
from salsa.SaLSA import SaLSA
self.optimizer = SaLSA(model.parameters())
```


The typical pytorch forward pass needs to be changed from :
``` 
x,y = load_input_data()
optimizer.zero_grad()
y_pred = model(x)
loss = criterion(y_pred, y)    
loss.backward()
optimizer.step()
scheduler.step() 
```
to:
``` 
x,y = load_input_data()
def closure(backwards = False):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    if backwards: loss.backward()
    return loss
optimizer.zero_grad()
loss = optimizer.step(closure = closure)
```
At the moment gradient scalers are not possible to be used simultaneously with SALSA.

This code change is necessary since, the optimizers needs to perform additional forward passes and thus needs to have the forward pass encapsulated in a function.
See embedder.py in the fit() method for exemplary usage.

## Disclaimer:

This optimizer was only tested and validated to perform well with MLPs, Transformers, and Convolutional Neural Networks (Mamba was also tested but needed a c value of 0.7). Your results may vary when you try it for other architectures and/or use cases on which it was not validated. When you encounter any issues, try tuning the c value of the optimizer and/or open an issue on Github.


## Replicating Results
Youtube Video Explaining the Concept:

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/EttProDnEDY/0.jpg)](https://www.youtube.com/watch?v=EttProDnEDY)

Dependencies:

- [pytorch](https://pytorch.org) <3
- [numpy](https://numpy.org/install/) <3


for replicating the results (not needed for using the optimizer):
- `pip install transformers` for huggingface transformers <3 
- `pip install datasets` for huggingface datasets <3 
- `pip install tensorflow-datasets` for tensorflow datasets <3 
- `pip install wandb` for optional logging <3
- for easy replication use conda and environment.yml eg:
`$ conda env create -f environment.yml` and `$ conda activate sls3`


The results of the Line Search Algorithm are:

![Loss Curve](figures/Table.png)

on average a 50\% reduction in final loss, while only needing about 3\% extra compute time on average.


For replicating the main Results of the Paper run:

```
$ python test/run_multiple.py
$ python test/run_multiple_img.py
```


For replicating specific runs or trying out different hyperparameters use:

```
$ python test/main.py 
```

and change the test/config.json file appropriately



## Older Versions of this Optimizer:
https://github.com/TheMody/Faster-Convergence-for-Transformer-Fine-tuning-with-Line-Search-Methods
https://github.com/TheMody/Improving-Line-Search-Methods-for-Large-Scale-Neural-Network-Training




## Please cite:
No learning rates needed: Introducing SALSA - Stable ArmijoLine Search Adaptation
from 
Philip Kenneweg,
Tristan Kenneweg,
Fabian Fumagalli
Barbara Hammer
published in IJCNN 2024 and on arvix



```
@INPROCEEDINGS{10650124,
  author={Kenneweg, Philip and Kenneweg, Tristan and Fumagalli, Fabian and Hammer, Barbara},
  booktitle={2024 International Joint Conference on Neural Networks (IJCNN)}, 
  title={No learning rates needed: Introducing SALSA - Stable Armijo Line Search Adaptation}, 
  year={2024},
  volume={},
  number={},
  pages={1-8},
  keywords={Training;Schedules;Codes;Search methods;Source coding;Computer architecture;Transformers},
  doi={10.1109/IJCNN60899.2024.10650124}}
```

