# update 1.3

print("Loading Libraries ...")

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torch.nn.functional as F
from torch import max as tmax
from torch import save as tsave
from torch.utils.data import DataLoader
from Neural_Network import Net

# Loading Dataset
print("Loading MNIST Dataset ...")
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: transforms.functional.invert(x))])

train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
validation_dataset = MNIST(root='./data', train=False, download=True, transform=transform)


# Defining the dimension of input and output of the Neural Network
input_dim = validation_dataset[0][0].shape[1] ** 2 # 28 * 28 = 784
outputs = list()
for i in range(len(validation_dataset)):
    outputs.append(validation_dataset[i][1])

outputs = set(outputs)
output_dim = len(outputs) # 0 -> 9 = 10


# Defining Neural Network
print("Building Neural Network ...")
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(dataset=train_dataset, batch_size=1000, shuffle=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)

layers = [input_dim, 400, output_dim]
model = Net(layers)

optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9)

scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.02, total_iters=5)

# Start training
print("Training ...")

epochs = 40

for epoch in range(epochs):
    i = 0
    for x_train, y_train in train_loader:
        optimizer.zero_grad()
        z = model(x_train)
        loss = criterion(z, y_train)
        loss.backward()
        optimizer.step()
        i += 1
        print("  Training Epoch {} ----- {:.2f} %".format(epoch + 1, float(i / len(train_loader) * 100)), end="\r")


    i = 0
    correct = 0
    validation_loss = 0
    for x_test, y_test in validation_loader:
        z = model(x_test)
        _, label = tmax(z, 1)
        validation_loss += F.nll_loss(z, y_test, size_average=False).data.item()
        correct += (label == y_test).sum().item()
        i += 1
        print("  Validation Epoch {} ----- {:.2f} %".format(epoch + 1, float(i / len(validation_loader) * 100)), end="\r")

    validation_loss /= len(validation_dataset)
    accuracy = 100 * (correct / len(validation_dataset))

    if epoch >= 10:
        scheduler.step()

    print(f"Epoch {epoch + 1}/{epochs} -------------- Accuracy = {round(accuracy, 2)} %  Loss = {round(validation_loss, 4)}")
    tsave(model.state_dict(), f"model.pt")


# Saving results
print("Saving Model ...")
