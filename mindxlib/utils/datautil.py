from pathlib import Path
import json
from typing import Dict

import pandas as pd

class DatasetLoader(object):
    def __init__(self, name: str, basedir: str='datasets'):
        path = Path(basedir) / name
        with open(path / 'meta.json') as f:
            meta = json.load(f)
        df = pd.read_csv(path / 'data.csv')
        label = df.eval(meta['positive']).to_numpy().astype(int)
        df.drop(meta['label'], axis='columns', inplace=True)
        df['label'] = label
    
        self._name = name
        self._meta = meta
        self._df = df

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def meta(self) -> Dict:
        return self._meta
    
    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df


class TestDatasetLoader(object):
    def __init__(self, name: str, basedir: str = 'datasets'):
        path = Path(basedir) / name
        with open('./meta.json') as f:
            meta = json.load(f)
        df = pd.read_csv(path / 'data.csv')
        label = df.eval(meta['positive']).to_numpy().astype(int)
        df.drop(meta['label'], axis='columns', inplace=True)
        df['label'] = label
        self._name = name
        self._meta = meta
        self._df = df

    @property
    def name(self) -> str:
        return self._name

    @property
    def meta(self) -> Dict:
        return self._meta

    @property
    def dataframe(self) -> pd.DataFrame:
        return self._df




