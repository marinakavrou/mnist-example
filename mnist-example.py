import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

class Net(nn.Module):
    # Constructor
    def __init__(self):
        # Define what each neuron does and other parameters
        super(Net, self).__init__()

        # Layer 1
        self.layer1_linear = nn.Linear(1, 10, bias=True)
        self.layer1_activation = nn.ReLU(inplace=True)

        # Layer 2
        self.layer2_linear = nn.Linear(10, 10, bias=True)
        self.layer2_activation = nn.ReLU(inplace=True)

    # Actually do the steps
    def forward(self, x):
        # Layer 1
        y1 = self.layer1_linear(x)
        y2 = self.layer1_activation(y1)

        # Layer 2
        y3 = self.layer2_linear(y2)
        y4 = self.layer2_activation(y3)

        y5 = y4.mean(1).mean(1)

        y6 = F.softmax(y5)

        return y6


# Network optimizer and loss
network = ConvNet()
network = network.to('cuda')
optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.5)

# Load the datasets


# Train dataset

train_dataset = torchvision.datasets.MNIST('/files/', train=True, download=True,
                                           transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                     torchvision.transforms.Normalize(
                                                                                         (0.1307,), (0.3081,))]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=False)

# Test dataset

test_dataset = torchvision.datasets.MNIST('/files/', train=False, download=True,
                                          transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                    torchvision.transforms.Normalize(
                                                                                        (0.1307,), (0.3081,))]))

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=True)

# Look into dataset

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

losses = []

# Train
for epoch in range(20):
    running_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        prediction = network(data.to('cuda'))
        loss = F.nll_loss(prediction, target.cuda())
        loss.backward()
        optimizer.step()

        if batch_idx == 0:
            print('Correct prediction: '+str(target[batch_idx]))
            print('Network prediction: '+str(torch.argmax(prediction[batch_idx])))

        running_loss += loss.item()

    losses.append(running_loss/len(train_loader))

    print(running_loss/len(train_loader))






