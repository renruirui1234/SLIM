"""
Rule set model for classifying imbalanced data.
"""

import os
import sys
import tempfile
import subprocess as sp

import numpy as np
import pandas as pd
import re

from sklearn.base import BaseEstimator


# binpath = os.path.dirname(os.path.abspath(__file__)) + '/bin/f1rule-darwin-aarch64'
# binpath = os.path.dirname(os.path.abspath(__file__)) + '/bin/f1rule-linux-amd64'
binpath = os.path.dirname(os.path.abspath(__file__)) + '/bin/f1rule-windows-amd64'

class RuleSetImb(BaseEstimator):
    def __init__(
        self, max_num_rules: int=16, time_limit=60, factor_g = 0.0,
        local_search_iter = 0, beta_pos=1.0, beta_neg=1.0, 
        beta_diverse=0.1, beta_complex=0.1,parallelism=0, 
        warmcache=0, bestsubset=0, exactdepth=0, allowrandom=0,
        verbose=False,
    ):
        self.max_num_rules = max_num_rules
        self.factor_g = factor_g
        self.local_search_iter = local_search_iter
        self.time_limit = time_limit
        self.beta_pos = beta_pos
        self.beta_neg = beta_neg
        self.beta_diverse = beta_diverse
        self.beta_complex = beta_complex
        self.parallelism = parallelism
        self.warmcache = warmcache
        self.bestsubset = bestsubset
        self.exactdepth = exactdepth
        self.allowrandom = allowrandom
        self.verbose = verbose

    def fit(self, X, y):
        if type(X) == np.ndarray:
            data = np.concatenate((X, y[..., np.newaxis]), axis=-1)
            columns = ['f' + str(i) for i in range(X.shape[-1])]
            columns.append('label')
            df = pd.DataFrame(data, columns=columns)
        else:
            df = pd.concat([X, y], axis=1)


        tmp='./processed_data/data.csv'

        df.to_csv(tmp, index=False)

        stderr = sys.stderr.fileno() if self.verbose else sp.DEVNULL
        proc = sp.run([
            binpath,
            '-p', str(self.parallelism),
            '-d', tmp,
            '-l', 'label',
            '-o', 'h',
            '-k', str(self.max_num_rules),
            '-fac', str(self.factor_g),
            '-iter', str(self.local_search_iter),
            '-t', str(self.time_limit) + 's',
            '-c', str(self.warmcache),
            '-b', str(self.bestsubset),
            '-e', str(self.exactdepth),
            '-r', str(self.allowrandom),
            '-pos', str(self.beta_pos),
            '-neg', str(self.beta_neg),
            '-complex', str(self.beta_complex),
            '-diverse', str(self.beta_diverse),
        ], stdout=sp.PIPE, stderr=stderr, check=True)
        lines = proc.stdout.decode("utf-8").splitlines()

        self.itemsets = []
        self.rules = []
        self.rates = []
        self.pos_neg=[]

        for line in lines[:-1]:
            ret = line.split(" <=> ")

            items = re.findall(r'\[(.*?)\]', ret[0])
            rate=re.findall(r"\d+",items[1])
            pos_neg=items[1]
            rate=int(rate[0])/(int(rate[0])+int(rate[1]))
            items=items[0]
            if items.strip() != '':
                self.itemsets.append([int(field) for field in items.split(' ')])
            else:
                self.itemsets.append([])

            self.rules.append(ret[1])
            self.rates.append(rate)
            self.pos_neg.append(pos_neg)



    def save_model(self):
        return self.rules,self.itemsets,self.rates,self.pos_neg


    def load_model(self,itemsets,rules,rates,pos_neg):
        self.rules=rules
        self.itemsets=itemsets
        self.rates=np.array(rates)
        self.pos_neg=pos_neg



    def predict(self, X: np.ndarray):
        if len(self.rules) == 0:
            return np.zeros(X.shape[0], dtype=int)


        predictions=[]
        for itemset in self.itemsets:
            a=X[..., itemset]
            predictions.append(np.prod(X[..., itemset], axis=-1))


        predictions = [
            np.prod(X[..., itemset], axis=-1)
            for itemset in self.itemsets
        ]#选择被规则命中的所有X
        index = np.argwhere(np.sum(predictions, axis=1) > 0)
        try:
            index=np.argwhere(np.sum(predictions, axis=1) > 0)
            rule = np.max(self.rates[index[:,0]])
        except:
            rule=0
        return np.greater(np.sum(predictions, axis=0), 0).astype(int),rule
