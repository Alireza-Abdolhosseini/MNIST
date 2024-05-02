print("Loading Libraries ...")

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch import max as tmax
from torch import save as tsave
from torch.utils.data import DataLoader
from Neural_Network import Net

# Loading Dataset
print("Loading MNIST Dataset ...")
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
                                     transforms.Lambda(lambda x: transforms.functional.invert(x))])

train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform=transform)
validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform=transform)


# Defining the dimension of input and output of the Neural Network
input_dim = validation_dataset[0][0].shape[1] ** 2 # 28 * 28
outputs = list()
for i in range(len(validation_dataset)):
    outputs.append(validation_dataset[i][1])

outputs = set(outputs)
output_dim = len(outputs) # 0 -> 9 = 10


# Defining Neural Network
print("Building Neural Network ...")
criterion = nn.CrossEntropyLoss()

train_loader = DataLoader(dataset=train_dataset, batch_size=500, shuffle=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=2000, shuffle=False)

layers = [input_dim, 300, output_dim]
model = Net(layers)

optimizer = optim.SGD(model.parameters(), lr=0.5, momentum=0.9)

scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=30)

# Start training
print("Training ...")

epochs = 20
for epoch in range(epochs):
    for x_train, y_train in train_loader:
        optimizer.zero_grad()
        z = model(x_train)
        loss = criterion(z, y_train)
        loss.backward()
        optimizer.step()

    correct = 0
    for x_test, y_test in validation_loader:
        z = model(x_test)
        _, label = tmax(z, 1)
        correct += (label == y_test).sum().item()

    accuracy = 100 * (correct / len(validation_dataset))
    print(f"Epoch {epoch + 1}/{epochs} -------------- Accuracy = {round(accuracy, 2)} %")


# Saving results
print("Saving Model ...")
tsave(model.state_dict(), "model.pt")
