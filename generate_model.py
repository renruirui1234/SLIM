import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json
from mindxlib import utils
from mindxlib.ruleset import RuleSetImb
import shutil
import argparse
from sklearn.metrics import cohen_kappa_score



parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, choices=DATASETS.keys(), required=True, help='dataset name')
# parser.add_argument('--seed', type=int, required=True, help='random seed, 0<=seed<=4')
parser.add_argument('--mode', type=str, required=True, choices=['train', 'test'], help='train or test')
args = parser.parse_args()






#应该对每个service的单独建模





class SLIM():
    def __init__(self,mode='test',rule_len=2):
        self.mode=mode
        self.rule_len = rule_len
    def run(self):
        if self.mode=='train':
            self.train_model()
        else:
            self.test_model()


    def train_model(self):
        dict = {}
        for file in os.listdir("./train_data"):
            df = utils.DatasetLoader(file, basedir="./train_data").dataframe
            # Separate target variable
            df.pop('timestamp')
            df.pop('serviceName')

            y = df.pop('label')
            if np.sum(y) == 0:  # 没有正标签的故障类型不做训练
                continue
            # Binarize the features
            binarizer = utils.FeatureBinarizer(numThresh=99, negations=True, threshStr=True)
            # df = binarizer.fit_transform(df)
            binarizer = binarizer.fit(df)
            df = binarizer.transform(df)
            df.columns = [' '.join(col).strip() for col in df.columns.values]
            model = self.train(df, y)
            dict[file] = model.save_model()

        with open('./model/model.json', 'w') as f:
            json.dump(dict, f)

    def train(self,X: np.ndarray, y: np.ndarray):
        clf = RuleSetImb(
            max_num_rules=2,#设置规则数量
            factor_g=0.
        )
        clf.fit(X, y)

        return clf



    def load_rule_model(self,model_dict,service)->RuleSetImb:
        model = RuleSetImb(
            max_num_rules=2,
            factor_g=0.
        )
        model.load_model(model_dict[service][1], model_dict[service][0], model_dict[service][2], model_dict[service][3])
        return model







    def test(self,model: object, X: np.ndarray, y: np.ndarray):
        y_hat, rule_score = model.predict(X)
        return y_hat, rule_score


    def del_file(self,path: str):
        ls = os.listdir(path)
        for i in ls:
            c_path = os.path.join(path, i)
            if os.path.isdir(c_path):  # 如果是文件夹那么递归调用一下
                self.del_file(c_path)
                shutil.rmtree(c_path)
            else:  # 如果是一个文件那么直接删除
                os.remove(c_path)

    def transfer_test_data_unrule(self,filename: str, services: list) -> pd.Series:
        self.del_file('./test_data/')
        data = pd.read_csv(filename)
        data.pop('Unnamed: 0')
        data.pop('timestamp')
        serviceName = data.pop('serviceName')

        for service in services:
            data_service = data

            if not os.path.exists('./test_data/' + service):
                os.makedirs('./test_data/' + service, mode=0o777)
            dict_meta = {}
            dict_meta['label'] = 'label'
            dict_meta['positive'] = 'label == 1'
            with open('./test_data/' + service + '/meta.json', 'w') as f:
                json.dump(dict_meta, f)
            data_service.to_csv('./test_data/' + service + '/data.csv', index=False)

        return serviceName

    def binarizer_load(self) -> dict:#Binarizer Feature Module
        binarizers = {}
        for service in os.listdir("./train_data"):
            df_train = utils.TestDatasetLoader(service, basedir="./train_data/").dataframe
            if 'label' in df_train.columns:
                df_train.pop('label')
            if 'serviceName' in df_train.columns:
                df_train.pop('serviceName')
            if 'timestamp' in df_train.columns:
                df_train.pop('timestamp')

            binarizer = utils.FeatureBinarizer(numThresh=99, negations=True, threshStr=True)
            binarizer = binarizer.fit(df_train)
            binarizers[service] = binarizer
        return binarizers

    def evaluate_metric(self, acc_service: np.ndarray, name: str, rule_scores: dict, service: str,result: dict) -> np.ndarray:

        index_train = pd.read_csv('./train_data.csv')
        index_train = index_train['serviceName'].unique()
        index_service_transfer = {}
        for i in range(len(index_train)):
            index_service_transfer[index_train[i]] = i
        y_true = []
        y_pred = []


        with open('./result-fault.txt', 'a') as f:
            f.write(name + ':\n')
            f.close()
        # fault localization module--fault_type
        rule_scores_list = sorted(rule_scores.items(), key=lambda x: x[1], reverse=True)
        index = 1
        with open('./result-fault.txt', 'a') as f:
            for rule in rule_scores_list[:5]:
                f.write('top' + str(index) + ' :' + rule[0] + '-num:' + str(rule[1]) + '\n')
                index += 1
            f.write('\n')
        f.close()

        with open('./result-service.txt', 'a') as f:
            f.write(name + ':\n')
            f.close()
        # fault localization module--service
        index = 1
        with open('./result-service.txt', 'a') as f:
            for rule in result[:5]:
                if index == 1:
                    y_pred=index_service_transfer[rule[0]]
                    y_true=index_service_transfer[service]
                if rule[0] == service:
                    acc_service[index - 1:] += 1
                f.write('top' + str(index) + ' :' + rule[0] + '-num:' + str(rule[1]) + '\n')
                index += 1
            f.write('\n')

        f.close()
        print('finishing the diagnosis of ' + name)

        return acc_service,y_pred,y_true



    def test_model(self):
        binarizers = self.binarizer_load()
        rule_scores = {}
        total = 0
        acc_service = np.zeros(5)
        y_preds=[]
        y_trues=[]
        with open('./model/model.json', 'r', encoding='utf-8') as f:
            dict = json.load(f)
        for root, dirs, files in os.walk("./raw_data/test/", topdown=False):
            for name in files:
                serviceName = self.transfer_test_data_unrule(root + name, list(dict.keys()))

                service_dict = {}
                for service in os.listdir("./test_data/"):
                    # for file_name in files_list:
                    df = utils.TestDatasetLoader(service, basedir="./test_data/").dataframe
                    if len(df) == 0:
                        continue
                    # Separate target variable
                    y = df.pop('label')
                    df = binarizers[service].transform(df)
                    df.columns = [' '.join(col).strip() for col in df.columns.values]

                    model=self.load_rule_model(dict,service)#
                    y_hat, rule_score = self.test(model, df.to_numpy(), y.to_numpy())

                    df['y_hat'] = y_hat
                    df['serviceName'] = serviceName
                    try:
                        result += df.groupby('serviceName')['y_hat'].sum()
                    except:
                        result = df.groupby('serviceName')['y_hat'].sum()

                    rule_scores[service] = np.sum(y_hat)

                result = result.to_dict()
                result = sorted(result.items(), key=lambda x: x[1], reverse=True)
                service = name.split('-')[2]
                service = service.split('.')[0]
                total += 1
                acc_service,y_pred,y_true=self.evaluate_metric(acc_service, name, rule_scores, service,result)
                y_preds.append(y_pred)
                y_trues.append(y_true)


        print('acc:' + str(acc_service / total))
        print()
        kappa_value = cohen_kappa_score(y_trues, y_preds)
        print("kappa value is %f" % kappa_value)




if __name__ == '__main__':
    slim=SLIM(args.mode)
    slim.run()












