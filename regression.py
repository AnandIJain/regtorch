import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

train_fn = 'nba2'
test_fn = 'college'
num_cols = 0

scaler = StandardScaler()
cols = 0

headers = ['a_team', 'h_team', 'league', 'game_id',
                'a_pts', 'h_pts', 'secs', 'status', 'a_win', 'h_win', 'last_mod_to_start',
                'num_markets', 'a_odds_ml', 'h_odds_ml', 'a_hcap_tot', 'h_hcap_tot']


def read_csv(fn, dummie):  # read csv and scale data
    raw = pd.read_csv(fn + '.csv', usecols=headers)
    raw = raw.dropna()
    raw = pd.get_dummies(data=raw, columns=['a_team', 'h_team', 'league'])
    raw = raw.astype(np.float32)
    print(raw.shape)
    return raw.copy()

def train_test(df):
    test = df.sample(frac=0.3,random_state=42)
    train = df.drop(test.index)
    return train, test


def scale(data):
    scaled = scaler.fit_transform(data.values)
    return scaled

df = read_csv(train_fn, 1)
train, test = train_test(df)
cols = df.columns
num_cols = len(cols)
# df = scale(df)
# print(df)

def get_batch(df, col='a_odds_ml', batch_size=1):
    batch = df.sample(batch_size)
    Y = batch[col]
    Y = Y.values
    Y = torch.tensor(Y, dtype=torch.float)
    X = batch.drop([col], axis=1)
    X = X.values
    X = torch.tensor(X, dtype=torch.float)
    return X, Y


def features(df):
    y_tmp = df[:, :1]
    x_tmp = df[:, 1:]

    X = torch.FloatTensor(x_tmp)
    Y = torch.FloatTensor(y_tmp)
    return X, Y


X, Y = get_batch(df)
num_teams = X.shape[1]
n_in, n_h, n_out, batch_size = num_teams, num_teams // 2, 128, 128

# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.l1 = torch.nn.Linear(n_in, n_h)
#         self.l2 = torch.nn.Linear(n_h, n_h)
#         self.l3 = torch.nn.Linear(n_h, n_out)


#     def forward(self, x):
#         print('f')
#         x = self.l1(x)
#         x = self.l2(x)
#         x = self.l3(x)
#         print(x)
#         return x


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

net = NeuralNet(n_in, n_h, n_out)

lr = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)


def step(data, labels, steps): 
    y_pred = net(data)
    print('.', end="")
    loss = criterion(y_pred, labels) 
    # print('loss: ')
    print(loss)
    steps += 1
    plt.scatter(steps, loss.item(), color='r', s=10, marker='o')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    


EPOCHS = 600
print(batch_size)
for i in range(EPOCHS):
    X, Y = get_batch(train, batch_size=batch_size)
    steps = 0
    step(X, Y, steps)

# Plot the graph
# predicted = net(torch.from_numpy()).detach().numpy()
# plt.plot(train, y_train, 'ro', label='Original data')
# plt.plot(x_train, predicted, label='Fitted line')
# plt.legend()
# plt.show()

# Save the model checkpoint
torch.save(net.state_dict(), 'model.ckpt')