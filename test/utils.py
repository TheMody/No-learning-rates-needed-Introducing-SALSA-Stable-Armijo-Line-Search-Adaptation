import matplotlib.pyplot as plt

def show_img(dataloader):
  for i, data in enumerate(dataloader):
    inputs, labels = data
    plt.imshow(inputs[0].permute(1,2,0))
    plt.show()
    print(inputs[0])
    print(labels[0])
    if i == 2:
      break