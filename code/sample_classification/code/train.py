#%%

import argparse
import os
import time
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import StratifiedKFold
from model_gnn_sc import *
from early_stopping import EarlyStopping


def print_time():
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))


parser = argparse.ArgumentParser(description="state_of_the_art_method_1")
parser.add_argument('-batch_size', default=10, type=int)
parser.add_argument('-lr', help="learning rate", default=0.01, type=float)
parser.add_argument('-epoch', default=50, type=int)
args = parser.parse_args()

if torch.cuda.is_available():
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.cuda._initialized = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = args.batch_size
LR = args.lr
EPOCH = args.epoch

#BATCH_SIZE = 10
#LR = 0.01
#EPOCH = 10
RESTART_INEFFECTIVE_TIMES = 5     # the times of ineffective training epoch to restart a training process
RESTART_CHANGE_LR = 0.2       # the change rate of learning rate when restarting training

print("learning rate: ", LR)
print("epoch: ", EPOCH)
print("batch size: ", BATCH_SIZE)
print("device: ", device)


# load data
meth_matrix = pd.read_csv('./meth_matrix_probes_250.csv', index_col=0)
random_sorted_sample_df = pd.read_csv('./sample_label_random.csv',index_col=0)
random_sorted_sample = random_sorted_sample_df.values[:,0]
y = random_sorted_sample_df.values[:,1].astype(int)[:-1]
X = meth_matrix.loc[:,random_sorted_sample].values.T.astype(float)[:-1]

y = torch.tensor(y, dtype=torch.long)
X = torch.tensor(X, dtype=torch.float)
print_time()
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

            model = Model_2()
            #print(model)
            model.to(device)

            lossf = nn.NLLLoss().to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,  milestones = [15,20], gamma = 0.2, last_epoch=-1)
            save_model_name = 'checkpoint_gnn_sc.pt'
            early_stopping = EarlyStopping(patience=15, verbose=True, save_model_name=save_model_name)
            
            ineffective_times = 0
            for epoch in range(EPOCH):
                
                # training mode
                model.train()
                for x_sample_train, y_train, x_graph_train in train_loader:
                    
                    # forward
                    out = model(x_sample_train, x_graph_train) 
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
                    for x_sample_train, y_train, x_graph_train in train_loader:
                                    
                        out = model(x_sample_train, x_graph_train)
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
                    test_losses = []
                    y_proba_test = torch.tensor([], dtype=torch.float).to(device)
                    y_real_test = torch.tensor([], dtype=torch.long).to(device)
                    for x_sample_test, y_test, x_graph_test in test_loader:
                        
                        out = model(x_sample_test, x_graph_test)
                        y_proba_test = torch.cat((y_proba_test, out), dim=0)
                        y_real_test = torch.cat((y_real_test, y_test), dim=0)
                        _, pred = torch.max(out,dim=1)
                        correct = (pred==y_test).sum().item()
                        acc = correct / len(y_test)
                        test_accs.append(acc)
                        test_losses.append(lossf(out, y_test).item())

                    test_acc = np.mean(test_accs)
                    test_loss = np.mean(test_losses)
                
                    # print
                    print('Epoch {:03d} train_loss: {:.4f} train_acc: {:.4f}  test_loss: {:.4f} test_acc: {:.4f} '.format(
                        epoch, train_loss, train_acc, test_loss, test_acc))

                if  epoch == 0:
                    orginal_acc = int(train_acc*10000)
                else:
                    if int(train_acc*10000) == orginal_acc:
                        ineffective_times += 1
                    else:
                        ineffective_times = 0

                early_stopping(test_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                if ineffective_times == RESTART_INEFFECTIVE_TIMES:
                    break
                
            if ineffective_times < RESTART_INEFFECTIVE_TIMES:
                if_train_effectively = True
            
            else:
                lr = lr * RESTART_CHANGE_LR
                torch.cuda.empty_cache()
                print("\nRestart traning")
                print("Change learning rate to ", lr)
                continue
            #torch.save(model.state_dict(), output_dir+'/model_sagcn_{}sites.pkl'.format(str(num_select_sites)))
            model.load_state_dict(torch.load(save_model_name))
            model.eval()
            y_proba_test = torch.tensor([], dtype=torch.float).to(device)
            y_real_test = torch.tensor([], dtype=torch.long).to(device)
            for test_data in test_loader:
                        
                test_data.to(device)
                        
                out,indexs = model(test_data)
                y_proba_test = torch.cat((y_proba_test, out), dim=0)
                y_real_test = torch.cat((y_real_test, test_data.y), dim=0)
                ind = torch.cat([ind,indexs],dim=0)

            y_proba_test = y_proba_test.cpu().detach().numpy()
            y_proba_test = np.exp(y_proba_test)
            y_real_test = y_real_test.cpu().detach().numpy()
        
            result = np.hstack((y_proba_test, y_real_test.reshape((-1, 1))))
            
            #results = pd.DataFrame(np.hstack((y_proba, y_real.reshape((-1,1)))))
            #results.to_csv(output_dir + '/probas_sagcn_{}sites.csv'.format(str(num_select_sites)))
            if fold == 0:
                results = result
            else:
                results = np.vstack((results, result))

            torch.cuda.empty_cache()
        
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir + '/probas_gnn_sc.csv')

print_time()
train()
print_time()
