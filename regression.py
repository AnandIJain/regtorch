import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

# credit https://github.com/utkuozbulak/pytorch-custom-dataset-examples

fn = 'nba2'

headers = ['a_team', 'h_team', 'league', 'game_id',
                'a_pts', 'h_pts', 'secs', 'status', 'a_win', 'h_win', 'last_mod_to_start',
                'num_markets', 'a_odds_ml', 'h_odds_ml', 'a_hcap_tot', 'h_hcap_tot']

class Df(Dataset):
    def __init__(self, df):
        # self.to_tensor = torch.tensor()
        self.data_len = len(df.index)
        self.labels = df['a_odds_ml'].values
        # print([self.data_len, self.labels])
        self.data = df.drop(['a_odds_ml'], axis=1)
        self.data = self.data.values

    def __getitem__(self, index):
        line = self.data[index, :]
        print(len(line))
        line_tensor = torch.tensor(line)
        line_label = self.labels[index]
        print(line_label)
        # label_tensor = torch.tensor(line_label)
        # tup = (len(line_tensor), len(label_tensor))
        # print(tup)
        return (line_tensor, line_label)

    def __len__(self):
        return self.data_len


def read_csv(fn):  # read csv and scale data
    raw = pd.read_csv(fn + '.csv', usecols=headers)
    raw = raw.dropna()
    raw = pd.get_dummies(data=raw, columns=['a_team', 'h_team', 'league'])
    raw = raw.astype(np.float32)
    print(raw.shape)
    return raw.copy()

def train_test(df):
    test = df.sample(frac=0.5,random_state=42)
    train = df.drop(test.index)
    return train, test

def scale(data):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data.values)
    return scaled
 

tmp_df = read_csv(fn)
train, test = train_test(tmp_df)
# print(type(train))
train = Df(train)
test = Df(test)


num_cols = train.data.shape[1]

input_size = num_cols
hidden_size = 50
num_classes = 1
num_epochs = 5
batch_size = 1 
learning_rate = 0.01

train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
# data, label = enumerate(train)
# print(data)
# print('label')
# print(label)

test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

net = NeuralNet(input_size, hidden_size, num_classes)

lr = 1e-4
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

EPOCHS = 1
print(batch_size)
steps = 0
running_loss = 0
for i in range(EPOCHS):
    for j, (data, labels) in enumerate(train_loader):
        y_pred = net(data)
        print(y_pred)
        print(labels)
        loss = criterion(y_pred, labels) 
        print(loss)
        plt.scatter(steps, loss.item(), color='r', s=10, marker='o')
        running_loss += abs(loss)
        print(running_loss)
        print(running_loss / j)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        steps += 1


def test():
    with torch.no_grad():
        correct = 0
        total = 0
        for i in range(EPOCHS):
            for i, (data, labels) in enumerate(train_loader):
                outputs = net(data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

test()

# Save the model checkpoint
torch.save(net.state_dict(), 'model.ckpt')

plt.show()
