import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

def transfer_train_data():
    data = pd.read_csv('./Trainticket-processed-data-0.7907/train_data.csv')
    timestamps = data['timestamp'].unique()
    services = data['serviceName'].unique()
    data_total = []
    label_total = []


    for i in range(len(timestamps)):
        data_time = []
        label_time = []
        for s in range(len(services)):
            data_temp = data.loc[(data['timestamp'] == timestamps[i])]
            data_temp = data_temp[data_temp['serviceName'] == services[s]]
            data_temp.pop('timestamp')
            data_temp.pop('Unnamed: 0')
            data_temp.pop('fault_description')
            data_temp.pop('serviceName')
            label = 1 if np.sum(data_temp.pop('label')) > 0 else 0
            data_value = data_temp.values
            data_time.append(data_value.reshape(-1))
            label_time.append(label)
            # print()
        index = np.argwhere(label_time)[0][0]
        temp = np.zeros(len(services) + 1)
        label_total.append(0 if index == -1 else index + 1)
        # if index == -1:
        #     label_total.append(temp)
        # else:
        #     temp[index + 1]=1
        #     label_total.append(temp)

        data_total.append(np.array(data_time))


    try:
        data_total = np.array(data_total)
    except:

        data_total_new=[]
        max_size=0
        for data in data_total:
            max_size = max(len(arr) for arr in data) if max_size <max(len(arr) for arr in data) else max_size
            padded_data = [np.pad(arr, (0, max_size - len(arr)), mode='constant') for arr in data]
            data_total_new.append(padded_data)
        data_total=np.array(data_total_new)



    min_vals = np.min(data_total, axis=(0,1), keepdims=True)
    max_vals = np.max(data_total, axis=(0,1), keepdims=True)
    data_total = (data_total - min_vals) / (max_vals - min_vals)
    data_total = np.nan_to_num(data_total, nan=0)

    np.save('time-series-train-data.npy', data_total)
    np.save('time-series-train-label.npy', np.array(label_total))
    np.save('min_val.npy', np.array(min_vals))
    np.save('max_val.npy', np.array(max_vals))


def transfer_test_data(data):
    timestamps = data['timestamp'].unique()
    services = data['serviceName'].unique()
    data_total = []
    label_total = []
    max_size=0
    for i in range(len(timestamps)):
        data_time = []
        label_time = []
        for s in range(len(services)):
            data_temp = data.loc[(data['timestamp'] == timestamps[i])]
            data_temp = data_temp[data_temp['serviceName'] == services[s]]
            data_temp.pop('timestamp')
            data_temp.pop('Unnamed: 0')
            # data_temp.pop('fault_description')
            data_temp.pop('serviceName')
            label = 1 if np.sum(data_temp.pop('label')) > 0 else 0
            data_value = data_temp.values
            max_size = max_size if max_size > data_value.shape[1] else data_value.shape[1]
            data_time.append(data_value.reshape(-1))
            label_time.append(label)
            # print()
        index = np.argwhere(label_time)[0][0]
        temp = np.zeros(len(services) + 1)
        label_total.append(0 if index == -1 else index + 1)
        # if index == -1:
        #     label_total.append(temp)
        # else:
        #     temp[index + 1]=1
        #     label_total.append(temp)

        data_total.append(np.array(data_time))
    try:
        return np.array(data_total),np.array(label_total)
    except:
        data_total_new=[]
        for data in data_total:
            max_size = max(len(arr) for arr in data)
            padded_data = [np.pad(arr, (0, max_size - len(arr)), mode='constant') for arr in data]
            data_total_new.append(padded_data)
        #complementary data
        return np.array(data_total_new), np.array(label_total)







def test_data_dfl1(model):
    for root, dirs, files in os.walk("./Trainticket-processed-data-0.7907/raw_data/test/", topdown=False):
        for name in files:
            data = pd.read_csv(root+name)
            X,label=transfer_test_data(data)
            X=torch.Tensor(X)
            model.eval()
            with torch.no_grad():
                output=model(X)
            output=torch.argmax(output,dim=1)
            print(output)
            print(label)
            # with open('result.txt', 'a') as f:
            #     f.write(name + ':\n')
            #     f.close()

def test_data_dfl2(model):
    for root, dirs, files in os.walk("./Trainticket-processed-data-0.7907/raw_data/test/", topdown=False):
        for name in files:
            data = pd.read_csv(root+name)
            X,label=transfer_test_data(data)
            min_vals=np.load('min_val.npy')
            max_vals=np.load('max_val.npy')
            X=(X-min_vals)/(max_vals-min_vals)
            X = np.nan_to_num(X, nan=0)
            output=model.predict(X.reshape(X.shape[0],X.shape[1]*X.shape[2]))
            label=label[0]

            








import torch
import torch.nn as nn
import torch.optim as optim



class MLP_dfl1(nn.Module):
    def __init__(self,input_size,service_num,hidden_size,output_size,batch_size=25):
        super(MLP_dfl1, self).__init__()
        self.mlp_1=nn.ModuleDict()
        self.input_size=input_size
        self.service_num=service_num
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.batch_size=batch_size
        self.relu=nn.ReLU()
        self.BN=nn.BatchNorm1d(self.service_num*self.hidden_size)
        for i in range(service_num):
            self.mlp_1[str(i)]=nn.Linear(input_size,hidden_size)
        self.fc=nn.Linear(hidden_size*service_num,output_size)


    def forward(self,x):
        x_hidden=torch.zeros((x.shape[0],self.service_num,self.hidden_size))
        for i in range(self.service_num):
            x_hidden[:,i,:]=self.relu(self.mlp_1[str(i)](x[:,i,:]))
        x_hidden=x_hidden.reshape(x_hidden.shape[0], x_hidden.shape[1] * x_hidden.shape[2])
        x_hidden=self.BN(x_hidden)
        x_output=self.fc(x_hidden)
        return x_output



from torch.utils.data import DataLoader,TensorDataset
from sklearn.neural_network import MLPClassifier

def train_data(mode='dfl1'):
    if mode=='dfl2':
        X = np.load('./time-series-train-data.npy')
        label = np.load('time-series-train-label.npy')

        batch_size = 25
        input_size = X.shape[2]  # 输入特征的大小
        service_num = X.shape[1]
        hidden_size = 10  # 隐藏层大小
        output_size = service_num  #

        # model_dfl2
        model_dfl2 = MLPClassifier(hidden_layer_sizes=(200, 400, 200), max_iter=4000)
        model_dfl2.fit(X.reshape(X.shape[0], X.shape[1] * X.shape[2]), label)
        test_data_dfl2(model_dfl2)

    if mode=='dfl1':
        X = np.load('./time-series-train-data.npy')
        label = np.load('time-series-train-label.npy')
        batch_size = 25
        input_size = X.shape[2]  # 输入特征的大小
        service_num = X.shape[1]
        hidden_size = 10  # 隐藏层大小
        output_size = service_num  #

        X = torch.Tensor(X)
        label = torch.Tensor(label)
        dataset = TensorDataset(X, label)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        # create model
        model = MLP_dfl1(input_size, service_num, hidden_size, output_size, batch_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 100
        for epoch in range(num_epochs):
            for batch_x, batch_y in data_loader:
                model.train()
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.to(torch.long))
                loss.backward()
                optimizer.step()
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
        test_data_dfl1(model)





if __name__ == '__main__':
    transfer_train_data()
    train_data('dfl2')








