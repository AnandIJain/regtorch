import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

import helpers as h

# credit https://github.com/utkuozbulak/pytorch-custom-dataset-examples

class Net(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()

        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, hidden_size)
        self.l5 = nn.Linear(hidden_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, input_size)

    def forward(self, x):
        print(type(x))
        x = F.relu(self.l1(x.float()))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
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

if __name__ == "__main__":
    batch_size = 1
    df = h.get_df()
    num_cols = df.shape[1]
    print(num_cols)
    train_df, test_df = h.train_test(df)
    print(train_df)

    games = h.get_t_games(train_df)
    test_games = h.get_t_games(test_df)


    train = h.DfGame(games)
    test = h.DfGame(test_games)

    input_size = num_cols * train.game_len // 2
    hidden_size = 2048
    print(input_size)
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

    net = Net(input_size, hidden_size)
    print(net)

    lr = 1e-4

    calc_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    EPOCHS = 1
    steps = 0
    running_loss = 0

    for i in range(EPOCHS):
        for j, x in enumerate(train_loader):

            first_half = torch.reshape(x[0], (-1, 1))
            first_half = torch.squeeze(first_half)

            second_half = torch.reshape(x[1], (-1, 1))
            second_half = torch.squeeze(second_half)

            print(first_half.shape)
            pred_second_half = net(first_half)
            loss = calc_loss(second_half, pred_second_half) 
            if j % 10 == 1:
                print('pred', end='')
                with torch.no_grad():
                    # params = [pred[0, 3].item(), pred[0, 4].item(), pred[0, 6].item(), pred[0, 7].item(), pred[0, 10].item(), pred[0, 11].item()]
                    print(pred_second_half, end='\n\n')

                    print('actual for ^', end='')
                    # real = [cur_data[0, 3].item(), cur_data[0, 4].item(), cur_data[0, 6].item(), cur_data[0, 7].item(), cur_data[0, 10].item(), cur_data[0, 11].item()]
                    print(second_half)

                # print('{0:.16f}'.format(cur_data[0, 1]))
                print("loss: ", end='')
                print(loss)

            plt.scatter(steps, loss.item(), color='r', s=10, marker='o')
            
            running_loss += abs(loss)
            print(running_loss / j)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1
            j += 1
    torch.save(net.state_dict(), '6.ckpt')


# def test(cur_data):
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for i in range(EPOCHS):
#             for i, d in enumerate(train_loader):
#                 prev_data = cur_data
#                 cur_data = d
#                 pred = net(prev_data.double())
#                 loss = calc_loss(pred, cur_data)
#                 total += 1
#                 print('{0:.16f}'.format(cur_data[0, 1]))
#                 print("loss: ", end='')
#                 print(loss)

# test(cur_data)


# plt.show()
