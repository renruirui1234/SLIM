import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import json


def transfer_data_unrule(filename: str):


    data = pd.read_csv(filename)

    if 'Unnamed: 0' in data.columns:
        data.pop('Unnamed: 0')

    fault_types = data['fault_description'].unique()
    for fault_type in fault_types:
        data_fault_type=data[data['fault_description'] == fault_type]
        data_fault_type.pop('fault_description')
        if not os.path.exists('./train_data/'+fault_type):
            os.makedirs('./train_data/' + fault_type,mode=0o777)

        dict_meta={}
        dict_meta['label']='label'
        dict_meta['positive']='label == 1'
        with open('./train_data/'+fault_type+'/meta.json', 'w') as f:
            json.dump(dict_meta, f)

        data_fault_type.to_csv('./train_data/'+fault_type+'/data.csv',index=False)




if __name__ == '__main__':
    transfer_data_unrule('./train_data.csv')






