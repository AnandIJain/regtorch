"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
"""
headers = [# 'a_team', 'h_team', 'sport', 'league', 
                'game_id', 'cur_time',
                'a_pts', 'h_pts', 'secs', 'status', 'a_win', 'h_win', 'last_mod_to_start',
                'num_markets', 'a_odds_ml', 'h_odds_ml', 'a_hcap_tot', 'h_hcap_tot']
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 64
LR = 0.005         # learning rate
N_TEST_IMG = 5


class Df(Dataset):
    def __init__(self, np_df, unscaled):
        # self.to_tensor = torch.tensor()
        self.data_len = len(np_df)
        # self.data_len = len(np_df.index)
        self.data = np_df
        # self.t_data = torch.tensor(self.data)
        self.unscaled_data = unscaled
        print(self.data_len)

    def __getitem__(self, index):
        # line = self.data.iloc[index]
        line = self.data[index]
        line_tensor = torch.tensor(line)
        # print(line_tensor.dtype)
        return line_tensor

    def __len__(self):
        return self.data_len


def read_csv(fn='nba2'):  # read csv and scale data
    raw = pd.read_csv(fn + '.csv')
    raw = raw.dropna()
    raw = pd.get_dummies(data=raw, columns=[ 'a_team', 'h_team', 'league', 'sport'])
    raw = raw.drop(['game_id', 'lms_date', 'lms_time'], axis=1)
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

tmp_df = read_csv()
train_df, test_df = train_test(tmp_df)

scaled_train = scale(train_df)
scaled_test = scale(test_df)

train = Df(scaled_train, train_df.values)
test = Df(scaled_test, test_df.values)


batch_size = 1
num_cols = tmp_df.shape[1]

input_size = num_cols
output_size = 10
learning_rate = 0.01

train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

# plot one example
# print(scaled_train.train_data.size())     # (60000, 28, 28)
# print(train_data.train_labels.size())   # (60000)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(num_cols, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, 10),   # compress to n features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, num_cols),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


autoencoder = AutoEncoder()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
loss_func = nn.MSELoss()

# initialize figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(10, 6))
plt.ion()   # continuously plot

# original data (first row) for viewing
view_data = torch.tensor(train.data).view(-1, num_cols).type(torch.FloatTensor)/255.
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (9, 10)), cmap='brg'); a[0][i].set_xticks(()); a[0][i].set_yticks(())

for epoch in range(EPOCH):
    for step, x in enumerate(train_loader):
        b_x = x.view(-1, num_cols).float()
        b_y = x.view(-1, num_cols).float()

        encoded, decoded = autoencoder(b_x)

        loss = loss_func(decoded, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy())

            # plotting decoded image (second row)
            _, decoded_data = autoencoder(view_data)
            for i in range(N_TEST_IMG):
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (9, 10)), cmap='brg')
                a[1][i].set_xticks(()); a[1][i].set_yticks(())
            plt.draw(); plt.pause(0.05)

plt.ioff()
plt.show()

# visualize in 3D plot
view_data = test.t_data[:200].view(-1, 91).type(torch.FloatTensor)/255.

encoded_data, _ = autoencoder(view_data)

fig = plt.figure(2); ax = Axes3D(fig)

X, Y, Z = encoded_data.data[:, 0].numpy(), encoded_data.data[:, 1].numpy(), encoded_data.data[:, 2].numpy()
values = train_data.train_labels[:200].numpy()

for x, y, z, s in zip(X, Y, Z, values):
    c = cm.rainbow(int(255*s/9)); ax.text(x, y, z, s, backgroundcolor=c)

ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
plt.show()