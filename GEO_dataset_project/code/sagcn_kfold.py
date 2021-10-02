# %%

# The definition of SAGCN model
# train model

import os
import numpy as np
import pandas as pd
import argparse
import torch
from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn.glob.glob import global_max_pool
from torch_scatter import scatter_mean, scatter_max
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGPooling
import torch.optim as optim
from createDataset import MyInMemoryDataset
from sklearn.model_selection import StratifiedKFold
from calibration import CalibrationModel

parser = argparse.ArgumentParser(description="SAGCN")
parser.add_argument('-k', help="the pooling rate of the third SAGPooling layer", type=float)
parser.add_argument('-batch_size', default=2, type=int)
parser.add_argument('-lr', help="learning rate", type=float)
parser.add_argument('-epoch', type=int)
args = parser.parse_args()


if torch.cuda.is_available():
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.cuda._initialized = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = args.batch_size
LR = args.lr
EPOCH = args.epoch
k_3 = args.k
RESTART_INEFFECTIVE_TIMES = 8     # the times of ineffective training epoch to restart a training process
RESTART_CHANGE_LR = 0.2       # the change rate of learning rate when restarting training

print("learning rate: ", LR)
print("epoch: ", EPOCH)
print("batch size: ", BATCH_SIZE)
print("device: ", device)

# === load dataset
dataset = MyInMemoryDataset(root='./dataset')
kfold = StratifiedKFold(n_splits = 5)


class SAGCN(nn.Module):
    def __init__(self, sag_rate_3):
        """
        sag_rate_3: k_3, the pooling rate of the third SAGPooling
        """
        super(SAGCN, self).__init__()

        self.conv1 = GCNConv(40, 40)
        self.pool1 = SAGPooling(40, 0.5)
        self.conv2 = GCNConv(40, 40)
        self.pool2 = SAGPooling(40, 0.5)
        self.conv3 = GCNConv(40, 40)
        self.pool3 = SAGPooling(40, sag_rate_3)

        # self.lstm_hidden = self.init_lstm_hidden().to(device)
        #self.lstm = nn.LSTM(1, 1, num_layers=2, batch_first=True, bidirectional=True)    # 双向lstm
        num_select_sites = int(500 * sag_rate_3)
        self.cnnlayer = nn.Conv1d(1, 1, 5, padding=2)
        self.mlp = nn.Sequential(
            nn.Linear(2*num_select_sites, num_select_sites),
            nn.ReLU(),
            nn.Linear(num_select_sites, num_select_sites // 2),
            nn.ReLU(),
            nn.Linear(num_select_sites // 2, 91))

    def forward(self, data):
        indexs = torch.tensor(list(range(2000))*BATCH_SIZE)
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        # print(x.shape)
        batch_size = x.shape[0]
        x1 = self.conv1(x, edge_index, edge_attr)
        x1 = F.relu(x1)
        pool1, edge_index1, edge_attr1, batch1, perm1, score1 = self.pool1(
                x1, edge_index, edge_attr, batch)
        indexs = indexs[perm1]
        # global_pool1 = torch.cat([self.global_avg_pool(pool1, batch1), self.global_max_pool(pool1, batch1)],dim=1)

        x2 = self.conv2(pool1, edge_index1, edge_attr1)
        x2 = F.relu(x2)
        pool2, edge_index2 , edge_attr2, batch2, perm2, score2 = self.pool2(x2, edge_index1, edge_attr1, batch1)
        indexs = indexs[perm2]
        # global_pool2 = torch.cat([self.global_avg_pool(pool2, batch2), self.global_max_pool(pool2, batch2)],dim=1)

        x3 = self.conv3(pool2, edge_index2, edge_attr2)
        x3 = F.relu(x3)
        pool3, edge_index3 , edge_attr3, batch3, perm3, score3 = self.pool3(x3, edge_index2, edge_attr2, batch2)
        indexs = indexs[perm3]
        # global_pool3 = torch.cat([self.global_avg_pool(pool3, batch3), self.global_max_pool(pool3, batch3)],dim=1)

        global_max, _ = torch.max(pool3, dim=1)
        global_max = global_max.view(batch_size, -1)
        global_avg = torch.mean(pool3, dim=1)
        global_avg = global_avg.view(batch_size, -1)
        readout = torch.cat([global_avg, global_max], dim=1)


        #lstmout, _ = self.lstm(readout.view(BATCH_SIZE, len(readout[0]), -1))
        #lstmout = lstmout.view(BATCH_SIZE, len(readout[0]),-1)
        #lstmout = torch.mean(lstmout, dim=2).view(BATCH_SIZE,-1)

        out = self.cnnlayer(readout.view(batch_size, 1, len(readout[0]))).view(batch_size, len(readout[0]))

        logits = self.mlp(out)

        return F.log_softmax(logits, dim=1), indexs.view(batch_size,-1)


# === main training function
def train_sagcn(sag_rate_3, LR):
    print("===============[ k={:.1f} ]==================".format(sag_rate_3))
    num_select_sites = int(500*sag_rate_3)

    output_dir = "./results/sites_" + str(num_select_sites)
    #output_dir = "./sagcn_results"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # === 5-fold cross-validation
    for fold, (train, test) in enumerate(kfold.split(dataset.data.y, dataset.data.y)):
        
        
        print("-------------fold: {}-----------------".format(str(fold)))

        train = torch.tensor(train, dtype=torch.long)
        test = torch.tensor(test, dtype=torch.long)

        train_data = dataset.index_select(train)
        test_data = dataset.index_select(test)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

        
        if_train_effectively = False
        lr = LR
        while not (if_train_effectively and lr >= 0.00001):

            model = SAGCN(sag_rate_3)
            # model.load_state_dict(torch.load(model_dir+'/model_sagcn_{}sites.pkl'.format(str(num_select_sites))))
            #print(model)

            print(device)
            model.to(device)
            # lossf = nn.CrossEntropyLoss().to(device)
            lossf = nn.NLLLoss().to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,  milestones = [20,40], gamma = 0.5, last_epoch=-1)
            # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1/(epoch+1))
            
            ineffective_times = 0
            for epoch in range(EPOCH):
                
                # training mode
                model.train()
                for data in train_loader:
                    data.to(device)
                    
                    # forward + loss calculation
                    out,_ = model(data) 
                    loss = lossf(out, data.y)
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
                    for train_data in train_loader:
                        
                        train_data.to(device)
                        
                        out,_ = model(train_data)
                        y_proba_train = torch.cat((y_proba_train, out), dim=0)
                        y_real_train = torch.cat((y_real_train, train_data.y), dim=0)
                        _, pred = torch.max(out,dim=1)
                        correct = (pred==train_data.y).sum().item()
                        acc = correct / len(train_data.y)
                        train_accs.append(acc)
                        train_loss += lossf(out, train_data.y).item()

                    train_acc = np.mean(train_accs)
                    
                    ind = torch.tensor([],dtype=torch.long)
                    test_accs = []
                    test_loss = 0.0
                    y_proba_test = torch.tensor([], dtype=torch.float).to(device)
                    y_real_test = torch.tensor([], dtype=torch.long).to(device)
                    for test_data in test_loader:
                        
                        test_data.to(device)
                        
                        out,indexs = model(test_data)
                        y_proba_test = torch.cat((y_proba_test, out), dim=0)
                        y_real_test = torch.cat((y_real_test, test_data.y), dim=0)
                        ind = torch.cat([ind,indexs],dim=0)
                        _, pred = torch.max(out,dim=1)
                        correct = (pred==test_data.y).sum().item()
                        acc = correct / len(test_data.y)
                        test_accs.append(acc)
                        test_loss += lossf(out, test_data.y).item()

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

            calibration_model = CalibrationModel()
            calibration_model.fit(y_proba_train, y_real_train)
            y_proba_test_calibrated = calibration_model.calibrate(y_proba_test)

            result = np.hstack((y_proba_test, y_proba_test_calibrated))
            result = np.hstack((result, y_real_test.reshape((-1, 1))))
            
            perm = ind.cpu().detach().numpy()
            #results = pd.DataFrame(np.hstack((y_proba, y_real.reshape((-1,1)))))
            #results.to_csv(output_dir + '/probas_sagcn_{}sites.csv'.format(str(num_select_sites)))
            if fold == 0:
                results = result
                perms = perm
                
            else:
                results = np.vstack((results, result))
                perms = np.vstack((perms, perm))

            torch.cuda.empty_cache()
        
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir + '/probas_Mset_sagcn_{}sites.csv'.format(str(num_select_sites)))
    perms_df = pd.DataFrame(perms)
    perms_df.to_csv(output_dir + '/perms_Mset_{}sites.csv'.format(str(num_select_sites)))

    

if __name__ == '__main__':
    # sag_rate_3 = 0.2
    
    train_sagcn(k_3, LR)
    # train_sagcn(0.2, 0.001)
    
    

# %%
