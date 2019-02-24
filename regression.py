import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
cols = 0
headers = ['a_team', 'h_team', 'league', 'game_id',
                'a_pts', 'h_pts', 'secs', 'status', 'a_win', 'h_win', 'last_mod_to_start',
                'num_markets', 'a_odds_ml', 'h_odds_ml', 'a_hcap_tot', 'h_hcap_tot']


def read_csv(fn, dummie):  # read csv and scale data
    raw = pd.read_csv(fn + '.csv', usecols=headers)
    raw = raw.dropna()
    raw = pd.get_dummies(data=raw, columns=['a_team', 'h_team', 'league'])
    return raw.copy()

def scale(data):
    scaled = scaler.fit_transform(data.values)
    return scaled

df = read_csv('nba2', 1)
cols = df.columns
df = scale(df)

def features(df):
    y_tmp = df[:, :1]
    x_tmp = df[:, 1:]

    X = torch.FloatTensor(x_tmp)
    Y = torch.FloatTensor(y_tmp)
    return X, Y

X, Y = features(df)    

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(53, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

net = Net()

loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

def run(train_test, data, labels):  # train == 1, test == else
    for epoch in range(200):
        y_pred = net(data)
        loss = loss_func(y_pred, labels)

        if train_test == 1:  
            plt.scatter(epoch, loss.item(), color='r', s=10, marker='o')
        else:
            plt.scatter(epoch, loss.item(), color='b', s=10, marker='o')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

run(1, X, Y)

test = read_csv('data', 0)
test = test.reindex(columns=cols, fill_value=0)
test = scale(test)
X_test, Y_test = features(test)

run(0, X_test, Y_test)

plt.show()