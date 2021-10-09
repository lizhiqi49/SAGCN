#%%

# comparation of state-of-the-art methods (1)
# 2021.10.07

# ----------------------------------------------------------------

import numpy as np
import pandas as pd
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import StratifiedKFold

if torch.cuda.is_available():
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.cuda._initialized = True


parser = argparse.ArgumentParser(description="state_of_the_art_method_1")
parser.add_argument('-batch_size', default=2, type=int)
parser.add_argument('-lr', help="learning rate", type=float)
parser.add_argument('-epoch', type=int)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = args.batch_size
LR = args.lr
EPOCH = args.epoch
RESTART_INEFFECTIVE_TIMES = 5     # the times of ineffective training epoch to restart a training process
RESTART_CHANGE_LR = 0.2       # the change rate of learning rate when restarting training

print("learning rate: ", LR)
print("epoch: ", EPOCH)
print("batch size: ", BATCH_SIZE)
print("device: ", device)


class MyDataset(Dataset):

    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = self.Data[index].to(device)
        label = self.Label[index].to(device)
        return data, label


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channel)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != self.expansion*out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channel, self.expansion*out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion*out_channel)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1, bias=False)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64, stride=2),
            BasicBlock(64, 64, stride=1)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128, stride=1)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256, stride=1)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512, stride=1)
        )
        self.conv2 = nn.Conv1d(512, 512, kernel_size=125, padding=0, bias=False)
        self.linear1 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(), nn.Dropout(0.5)
            )
        self.linear2 = nn.Linear(128, 12)
    
    def forward(self, input):
        batch = input.shape[0]
        x = self.conv1(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv2(x)
        x = x.view(batch, -1)
        x = self.linear1(x)
        x = self.linear2(x)
        out = F.log_softmax(x, dim=1)
        return out


# load data
meth_matrix = pd.read_csv('../meth_matrix_maxstd_2k_sorted.csv', index_col=0)
random_sorted_sample_df = pd.read_csv('../sample_label_random.csv',index_col=0)
random_sorted_sample = random_sorted_sample_df.values[:,0]
y = random_sorted_sample_df.values[:,1].astype(int)
X = meth_matrix.loc[:,random_sorted_sample].values.T.astype(float)

y = torch.tensor(y, dtype=torch.long)
X = torch.tensor(X, dtype=torch.float).view(-1, 1, 2000)
print("data loaded")

kfold = StratifiedKFold(n_splits = 5)

def train():
    print("===============Start Training==================")

    output_dir = "./prediction_results/"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # === 5-fold cross-validation
    for fold, (train, test) in enumerate(kfold.split(y, y)):
        
        
        print("-------------fold: {}-----------------".format(str(fold)))

        train = torch.tensor(train, dtype=torch.long)
        test = torch.tensor(test, dtype=torch.long)

        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]
        dataset_train = MyDataset(X_train, y_train)
        dataset_test = MyDataset(X_test, y_test)
        train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True)

        
        if_train_effectively = False
        lr = LR
        while not (if_train_effectively and lr >= 0.00001):

            model = Model_1()
            print(model)
            model.to(device)

            lossf = nn.NLLLoss().to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,  milestones = [20,40], gamma = 0.2, last_epoch=-1)

            
            ineffective_times = 0
            for epoch in range(EPOCH):
                
                # training mode
                model.train()
                for X_train, y_train in train_loader:
                    
                    # forward
                    out = model(X_train) 
                    loss = lossf(out, y_train)
                    # backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                # test mode
                model.eval()
                with torch.no_grad():
                    train_accs = []
                    train_loss = 0.0
                    y_proba_train = torch.tensor([], dtype=torch.float).to(device)
                    y_real_train = torch.tensor([], dtype=torch.long).to(device)
                    for X_train, y_train in train_loader:
            
                        
                        out = model(X_train)
                        y_proba_train = torch.cat((y_proba_train, out), dim=0)
                        y_real_train = torch.cat((y_real_train, y_train), dim=0)
                        _, pred = torch.max(out,dim=1)
                        correct = (pred==y_train).sum().item()
                        acc = correct / len(y_train)
                        train_accs.append(acc)
                        train_loss += lossf(out, y_train).item()

                    train_acc = np.mean(train_accs)
                    
                    ind = torch.tensor([],dtype=torch.long)
                    test_accs = []
                    test_loss = 0.0
                    y_proba_test = torch.tensor([], dtype=torch.float).to(device)
                    y_real_test = torch.tensor([], dtype=torch.long).to(device)
                    for X_test, y_test in test_loader:
                        
                        out = model(X_test)
                        y_proba_test = torch.cat((y_proba_test, out), dim=0)
                        y_real_test = torch.cat((y_real_test, y_test), dim=0)
                        _, pred = torch.max(out,dim=1)
                        correct = (pred==y_test).sum().item()
                        acc = correct / len(y_test)
                        test_accs.append(acc)
                        test_loss += lossf(out, y_test).item()

                    test_acc = np.mean(test_accs)
                
                    # print
                    print('Epoch {:03d} train_loss: {:.4f} train_acc: {:.4f}  test_loss: {:.4f} test_acc: {:.4f} '.format(
                        epoch, train_loss, train_acc, test_loss, test_acc))

                if  epoch == 0:
                    orginal_acc = train_acc 
                elif epoch < EPOCH - 1:
                    if train_acc == orginal_acc:
                        ineffective_times += 1
                    else:
                        ineffective_times = 0
                else:
                    if_train_effectively = True

                if ineffective_times == RESTART_INEFFECTIVE_TIMES:
                    break
                
            if ineffective_times == RESTART_INEFFECTIVE_TIMES:
                lr = lr * RESTART_CHANGE_LR
                torch.cuda.empty_cache()
                print("\nRestart traning")
                print("Change learning rate to ", lr)
                continue
            #torch.save(model.state_dict(), output_dir+'/model_sagcn_{}sites.pkl'.format(str(num_select_sites)))
            y_proba_train = y_proba_train.cpu().detach().numpy()
            y_proba_train = np.exp(y_proba_train)
            y_real_train = y_real_train.cpu().detach().numpy()

            y_proba_test = y_proba_test.cpu().detach().numpy()
            y_proba_test = np.exp(y_proba_test)
            y_real_test = y_real_test.cpu().detach().numpy()

            """
            calibration_model = CalibrationModel()
            calibration_model.fit(y_proba_train, y_real_train)
            y_proba_test_calibrated = calibration_model.calibrate(y_proba_test)
            """
            result = np.hstack((y_proba_test, y_real_test.reshape((-1, 1))))
            
            #results = pd.DataFrame(np.hstack((y_proba, y_real.reshape((-1,1)))))
            #results.to_csv(output_dir + '/probas_sagcn_{}sites.csv'.format(str(num_select_sites)))
            if fold == 0:
                results = result
            else:
                results = np.vstack((results, result))

            torch.cuda.empty_cache()
        
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir + '/probas_method_1.csv')

train()
