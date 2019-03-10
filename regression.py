import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

# credit https://github.com/utkuozbulak/pytorch-custom-dataset-examples

fn = 'nba2'

headers = ['a_team', 'h_team', 'sport', 'league', 'game_id', 'cur_time',
                'a_pts', 'h_pts', 'secs', 'status', 'a_win', 'h_win', 'last_mod_to_start',
                'num_markets', 'a_odds_ml', 'h_odds_ml', 'a_hcap_tot', 'h_hcap_tot']

class Df(Dataset):
    def __init__(self, np_df):
        # self.to_tensor = torch.tensor()
        self.data_len = len(np_df)
        self.data = np_df
        print(self.data_len)

    def __getitem__(self, index):
        line = self.data[index]
        line_tensor = torch.tensor(line)
        print(line_tensor.dtype)
        return line_tensor

    def __len__(self):
        return self.data_len


def read_csv(fn):  # read csv and scale data
    raw = pd.read_csv(fn + '.csv', usecols=headers)
    raw = raw.dropna()
    raw = pd.get_dummies(data=raw, columns=['a_team', 'h_team', 'league', 'sport'])
    print(raw.columns)
    # raw = raw.astype(np.float32)
    raw = raw.sort_values('cur_time', axis=0)
    return raw.copy()

def train_test(df):
    test = df.sample(frac=0.5,random_state=None)
    train = df.drop(test.index)
    return train, test

def scale(data):
    scaler = StandardScaler()
    print(data)
    vals = data.to_numpy()
    scaled = scaler.fit_transform(vals)
    print(scaled.shape)
    return scaled


class Net(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()

        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(output_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, input_size)

    def forward(self, x):
        x = F.relu(self.l1(x.float()))
        x = F.relu(self.l2(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x.double()

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# net = Net()
# print(net)

tmp_df = read_csv(fn)
train_df, test_df = train_test(tmp_df)

scaled_train = scale(train_df)
scaled_test = scale(test_df)

train = Df(scaled_train)
test = Df(scaled_test)



num_cols = tmp_df.shape[1]

input_size = num_cols
hidden_size = 50
output_size = 10
conv_size = 5
batch_size = 1 
learning_rate = 0.01

train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

net = Net(input_size, hidden_size, output_size)
print(net)
lr = 1e-4

calc_loss = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

EPOCHS = 1
steps = 0
running_loss = 0


def initial():

    prev_data = train.__getitem__(0)
    cur_data = train.__getitem__(1)

    print(prev_data.dtype)
    # print(cur_data)
    pred = net(prev_data)
    
    loss = calc_loss(pred, cur_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return cur_data

cur_data = initial()

for i in range(EPOCHS):
    for j, new_data in enumerate(train_loader):
        
        prev_data = cur_data
        cur_data = new_data
        pred = net(prev_data)
        loss = calc_loss(pred, cur_data) 
        print('{0:.16f}'.format(cur_data[0, 1]))
        print("loss: ", end='')
        print(loss)

        plt.scatter(steps, loss.item(), color='r', s=10, marker='o')
        
        running_loss += abs(loss)
        print(running_loss / j)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        steps += 1


torch.save(net.state_dict(), '5.ckpt')


def test(cur_data):
    with torch.no_grad():
        correct = 0
        total = 0
        for i in range(EPOCHS):
            for i, d in enumerate(train_loader):
                prev_data = cur_data
                cur_data = d
                pred = net(prev_data.double())
                loss = calc_loss(pred, cur_data)
                total += 1
                print('{0:.16f}'.format(cur_data[0, 1]))
                print("loss: ", end='')
                print(loss)

test(cur_data)


plt.show()
