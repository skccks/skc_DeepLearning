import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import h5py
import os
import torchvision.transforms as transforms
import time
import pandas as pd
from sklearn.model_selection import train_test_split
#pip install scikit-learn
import matplotlib.pyplot as plt
import torch.utils.data as Data

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X,y):
        super(MyDataset, self).__init__()
        self.X = X
        self.y = y

    def __getitem__(self, index):
        X, y = torch.Tensor(self.X[index]),torch.Tensor([self.y[index]])
        return X,y

    def __len__(self):
        return len(self.X)

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(21, 64)
        self.layer2 = nn.Linear(64,128)
        self.layer3 = nn.Linear(128, 256)
        self.layer4 = nn.Linear(256, 128)
        self.layer5 = nn.Linear(128, 1)

    def forward(self, x):
        y = self.layer1(x)
        y = F.relu(y)
        y = self.layer2(y)
        y = F.relu(y)
        y = self.layer3(y)
        y = F.relu(y)
        y = self.layer4(y)
        y = F.relu(y)
        y = self.layer5(y)
        return y

def val_plot(total_loss):
    x = range(len(total_loss))
    plt.plot(x,total_loss,label='Val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Val_loss')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Val_loss.png')

def true_pre_plot(true, pre):
    plt.figure()
    x = range(len(true))
    plt.plot(x, true, label='true')
    plt.plot(x, pre, label='pre')
    plt.xlabel('num')
    plt.ylabel('val')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('true_predict.png')


if __name__ == "__main__":
    EPOCH = 100  # train the training data n times, to save time, we just train 1 epoch
    LR = 0.001  # learning rate
    BATCH_SIZE=10

    data = pd.read_csv("train.csv")

    all_features = data.iloc[:, 0:-1]

    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
    # ??????????????????????????????????????????0????????????????????????0??????????????????
    all_features = all_features.fillna(0)
    # print(all_features)
    # print(all_features.shape)

    # dummy_na=True??????????????????????????????????????????????????????????????????
    all_features = pd.get_dummies(all_features, dummy_na=True)
    # print(all_features.shape)

    X = torch.tensor(all_features.values, dtype=torch.float)
    # print('train_features', train_features.shape)

    y = torch.tensor(data.label.values, dtype=torch.float).view(-1, 1)
    # print('train_labels:', train_labels)

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=23)
    # split train_val into training set and val set using sklearn.model_selection.traing_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1/9, random_state=23)

    print("Training instances   {}, Training features   {}".format(X_train.shape[0], X_train.shape[1]))
    print("Validation instances {}, Validation features {}".format(X_val.shape[0], X_val.shape[1]))
    print("Testing instances    {}, Testing features    {}".format(X_test.shape[0], X_test.shape[1]))

    train_data = MyDataset(X_train,y_train)
    val_data = MyDataset(X_val,y_val)
    test_data = MyDataset(X_test,y_test)

    train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = Data.DataLoader(dataset=val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

    time_start = time.time()
    model = DNN()
    print(model)  # net architecture
    # Loss and optimizer
    criterion = nn.MSELoss(reduction='mean')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)#LR=0.001
    val_MSE = []
    for epoch in range(EPOCH):  # loop over the dataset multiple times
        model.train()
        train_loss = 0.0
        for step, (data, label) in enumerate(train_loader):
            data=data.to(device)
            label=label.to(device)
            output = model(data)
            loss = criterion(output, label)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if step % 10 == 9:
                print('[%d,%5d] loss: %.3f' % (epoch + 1, (step + 1)*10, train_loss / 100))
                #Batch size=10,???????????????100?????????????????????loss
                train_loss = 0.0
        model.eval()
        val_loss = 0.
        with torch.no_grad():  # ???????????????????????????????????????
            for step, (data, label) in enumerate(val_loader):
                data = data.to(device)
                label = label.to(device)
                output = model(data)
                loss = criterion(output, label)
                val_loss += loss.item()
            val_MSE.append(val_loss/X_val.shape[0])
        model.train()
        if len(val_MSE) == 0 or val_MSE[-1] <= min(np.array(val_MSE)):
            # ??????????????????mse????????????????????????
            print("Best model on epoch: {}, val_mse: {:.4f}".format(epoch, val_MSE[-1]))
            torch.save(model.state_dict(), "Regression-best.th")
    val_plot(val_MSE)
    time_end = time.time()
    print('Training time:', time_end - time_start, 's')
    print('Train Finished')

    model = DNN()
    model.load_state_dict(torch.load('Regression-best.th'))
    model.to(device)
    predict = []
    true = []
    with torch.no_grad():
        test_loss,test_step=0,0
        for data, label in test_loader:
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            loss = criterion(output, label)
            test_loss += loss.item()
            batch_predict = output.squeeze(1).cpu().numpy().tolist()
            batch_true = label.squeeze(1).cpu().numpy().tolist()
            predict.extend(batch_predict)
            true.extend(batch_true)
        print(predict)
        print(true)
        true_pre_plot(true, predict)
        print("Mse of the best model on the test data is: {:.4f}".format(test_loss / X_test.shape[0]))
    print('Test Finished ')