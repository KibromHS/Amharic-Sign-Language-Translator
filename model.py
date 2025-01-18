import torch.nn as nn

class ASLModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ASLModel, self).__init__()
        self.fc1 = nn.Linear(63, 128)  # 21 landmarks * 3 (x, y, z)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
