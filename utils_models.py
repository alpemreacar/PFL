from utils_libs import *

class client_model(nn.Module):
    def __init__(self, name, args=''):
        super(client_model, self).__init__()
        self.name = name
        self.proto = False
        if self.name == 'cifar10':
            self.n_cls = args if isinstance(args, int) else 10
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*5*5, 384) 
            self.fc2 = nn.Linear(384, 192)
            self.fc3 = nn.Linear(192, self.n_cls)
                
        if self.name == 'cifar100':
            self.n_cls = args if isinstance(args, int) else 100
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*5*5, 384) 
            self.fc2 = nn.Linear(384, 192)
            self.fc3 = nn.Linear(192, self.n_cls)
                
        if self.name == 'miniImageNet':
            self.n_cls = args if isinstance(args, int) else 100
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64 , kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64*7*7, 400) 
            self.fc2 = nn.Linear(400, 100)
            self.fc3 = nn.Linear(100, self.n_cls)
            

    def forward(self, x):

        if self.name == 'cifar10':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64*5*5)
            x = F.relu(self.fc1(x))
            x_= F.relu(self.fc2(x))
            x = self.fc3(x_)
        
        if self.name == 'cifar100':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64*5*5)
            x = F.relu(self.fc1(x))
            x_= F.relu(self.fc2(x))
            x = self.fc3(x_)
                
        if self.name == 'miniImageNet':
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 64*7*7)
            x = F.relu(self.fc1(x))
            x_= F.relu(self.fc2(x))
            x = self.fc3(x_)

        if self.proto:
            return x_
        else:
            return x