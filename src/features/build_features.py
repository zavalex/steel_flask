import os
from pathlib import Path
import pandas as pd
import numpy as np


class Features:

    def __init__(self, path) -> None:
        self.path = path #path to data directory

    def load_raw_data(self):
        self.arc = pd.read_csv(os.path.join(self.path, 'raw\data_arc.csv'))
        self.bulk = pd.read_csv(self.path+r'\raw\data_bulk.csv')
        self.bulk_time = pd.read_csv(self.path+r'\raw\data_bulk_time.csv')
        self.gas = pd.read_csv(self.path+r'\raw\data_gas.csv')
        self.temp = pd.read_csv(self.path+r'\raw\data_temp.csv')
        self.wire = pd.read_csv(self.path+r'\raw\data_wire.csv')
        self.wire_time = pd.read_csv(self.path+r'\raw\data_wire_time.csv')
        dataframes = [self.arc, self.bulk, self.bulk_time, self.gas, self.temp, self.wire, self.wire_time]
        for d in dataframes:
            d.columns = d.columns.str.lower().str.replace(' ','_')
        self.arc.columns = ['key', 'heat_start', 'heat_end', 'active_power', 'reactive_power']
        self.gas.columns = ['key', 'gas']
        self.temp.columns = ['key', 'measure_time', 'temp']

    def build_features(self):
        self.load_raw_data()
        # preprocess data
        # Use correct datatypes
        self.arc['heat_start'] = pd.to_datetime(self.arc['heat_start'])
        self.arc['heat_end'] = pd.to_datetime(self.arc['heat_end'])
        self.temp['measure_time'] = pd.to_datetime(self.temp['measure_time'])
        
        # pass processing
        self.temp = self.temp.dropna()
        self.wire = self.wire.fillna(0)
        self.bulk = self.bulk.fillna(0)
        
        self.arc['reactive_power'][self.arc['reactive_power']<0] = np.mean(self.arc['reactive_power'])
        
        #create new features
        self.temp_with_features = self.temp.groupby('key').agg(['count', 'first', 'last']).reset_index()
        self.temp_with_features.columns = ['key', 'measure_count', 'first_measure_time', 
                'last_measure_time','temp_count', 'first_measure','last_measure']
        measure_count = self.temp.groupby('key')['measure_time'].count()
        self.temp_with_features = self.temp_with_features[~self.temp_with_features['key'].isin(measure_count[measure_count == 1].index)]
        self.temp_with_features['measure_time_diff'] = (self.temp_with_features['last_measure_time'] - self.temp_with_features['first_measure_time']).dt.total_seconds()
        self.temp_with_features.drop(['temp_count', 'first_measure_time','last_measure_time'], axis=1, inplace=True)
        
        self.arc['heat_time'] = (self.arc['heat_end'] - self.arc['heat_start']).dt.total_seconds()
        self.arc_with_features = self.arc.groupby('key').agg(['count', 'sum']).reset_index()
        self.arc_with_features.columns= ['key','1','active_power_sum','2','reactive_power_sum','heat_count','heat_time_sum']
        self.arc_with_features.drop(['1','2'], axis=1, inplace=True)
        self.arc_with_features['mean_power_ratio'] = self.arc_with_features['active_power_sum'] / self.arc_with_features['reactive_power_sum'] / self.arc_with_features['heat_count']

        #merge preprocessed datasets
        self.dataset = pd.merge(self.temp_with_features, self.arc_with_features, how='inner', on=['key'])
        self.dataset = pd.merge(self.dataset, self.bulk, how='inner', on=['key'])
        self.dataset = pd.merge(self.dataset, self.wire, how='inner', on=['key'])
        self.dataset = self.dataset[self.dataset['first_measure']!=self.dataset['last_measure']]
        
        #remove parameteres with high corr coefficient
        self.dataset = self.dataset.drop(['bulk_9', 'active_power_sum', 'reactive_power_sum','key','measure_count',
            'wire_5', 'bulk_8', 'bulk_2','bulk_13','bulk_5', 'wire_8', 'wire_9'],axis=1)
        DATASET_PATH = os.path.join(self.path, 'processed\dataset.csv')
        self.dataset.to_csv(index=False, path_or_buf=DATASET_PATH)

        return self.dataset