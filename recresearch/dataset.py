import csv
from datetime import datetime
from google_drive_downloader import GoogleDriveDownloader as gdd
import numpy as np
import os
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import LabelEncoder

import recresearch as rr

DATASETS_TABLE = pd.DataFrame(
    [[1,  'Anime Recommendations',     'E', 'S', '1GSz8EKsA3JlKfI-4qET0nUtmoyBGWNJl'],
     [2,  'BestBuy',                   'I', 'T', '1WZY5i6rRTBH4g8M5Qd0oSBVcWbis14Zq'],
     [3,  'Book-Crossing',             'E', 'S', '1mFC20Rauj-PRhYNm_jzzDGmKafobWdrq'],
     [4,  'CiaoDVD',                   'E', 'T', '1a_9fVVrelz-8XYs3tHZM8rnLtZRe6x8H'],
     [5,  'DeliciousBookmarks',        'I', 'T', '14geC9mUx1--xHkAUPtYLrMfZk4jc4ITW'],
     [6,  'Filmtrust',                 'E', 'S', '1V9Hd0DLhZzmA6c2bprmUlW6LWjfXC10p'],
     [7,  'Jester',                    'E', 'S', '1Yw28A-2l5Z-xB48puSWw4C_oP_DypNry'],
     [8,  'Last.FM - Listened',        'I', 'S', '1g3j9UP2a0gvB0fYJ9OzPAW1k1g59JobH'],
     [9,  'Last.FM - Tagged',          'I', 'S', '1bDHeh7L2TbBC_hJJCrb2TSohhK2Hl6ah'],
     [10, 'LibimSeTi',                 'E', 'S', '1AtmEnX415YAlUPmqjOK5pXGoJBSO3Jt3'],
     [11, 'MovieLens',                 'E', 'T', '1Tbi5EVs7BBZmnuKaFHDZelFgDuz-9YEP'],
     [12, 'NetflixPrize',              'E', 'T', '1gpoUoSFQTTAIUtdYCVLRuv0SI6vAZifr'],
     [13, 'RetailRocket-All',          'I', 'T', '12oHsCzjrlNbRe_pvTOCxkWVFlfO9WjhH'],
     [14, 'RetailRocket-Transactions', 'I', 'T', '12EwJisOE-6-xvYXe-YN_qX1FBfFWEiZv']], 
    columns=[rr.DATASET_ID, rr.DATASET_NAME, rr.DATASET_TYPE, rr.DATASET_TEMPORAL_BEHAVIOUR, rr.DATASET_GDRIVE]
)

""" DATASETS_TABLE = pd.DataFrame(
    [[4,  'CiaoDVD',                   'E', 'T', '1a_9fVVrelz-8XYs3tHZM8rnLtZRe6x8H'],
     [5,  'DeliciousBookmarks',        'I', 'T', '14geC9mUx1--xHkAUPtYLrMfZk4jc4ITW'],
     [13, 'RetailRocket-All',          'I', 'T', '12oHsCzjrlNbRe_pvTOCxkWVFlfO9WjhH'],
     [14, 'RetailRocket-Transactions', 'I', 'T', '12EwJisOE-6-xvYXe-YN_qX1FBfFWEiZv']], 
    columns=[rr.DATASET_ID, rr.DATASET_NAME, rr.DATASET_TYPE, rr.DATASET_TEMPORAL_BEHAVIOUR, rr.DATASET_GDRIVE]
) """

class SparseRepr(object):
    def __init__(self, df=None, users=None, items=None):
        if df is not None and users is None or items is None:
            users = df[rr.COLUMN_USER_ID].unique()
            items = df[rr.COLUMN_ITEM_ID].unique()
        if users is not None and items is not None:
            self._create_encoders(users, items)
        else:
            raise Exception('Error: wrong parameters for class SparseRepr.')

    def _create_encoders(self, users, items):
        self._user_encoder = LabelEncoder()
        self._item_encoder = LabelEncoder()
        self._user_encoder.fit(users)
        self._item_encoder.fit(items)

    def get_matrix(self, users, items, interactions=None):
        # Captura os indices de linha e coluna
        users_coo, items_coo = self._user_encoder.transform(users), self._item_encoder.transform(items)
        # Constroi o vetor de conteudo da matrix
        data = interactions if interactions is not None else np.ones(len(users_coo))
        # Constroi a matriz
        sparse_matrix = sparse.coo_matrix((data, (users_coo, items_coo)), shape=(len(self._user_encoder.classes_), len(self._item_encoder.classes_))).tocsr()
        return sparse_matrix

    def get_n_users(self):
        return len(self._user_encoder.classes_)

    def get_n_items(self):
        return len(self._item_encoder.classes_)

    def get_idx_of_user(self, user):
        return self._user_encoder.transform(user) if type(user) in [list, np.ndarray] else self._user_encoder.transform([user])[0]

    def get_idx_of_item(self, item):
        return self._item_encoder.transform(item) if type(item)in [list, np.ndarray] else self._item_encoder.transform([item])[0]

    def get_user_of_idx(self, idx):
        return self._user_encoder.inverse_transform(idx) if type(idx) in [list, np.ndarray, pd.Series] else self._user_encoder.inverse_transform([idx])[0]

    def get_item_of_idx(self, idx):
        return self._item_encoder.inverse_transform(idx) if type(idx) in [list, np.ndarray, pd.Series] else self._item_encoder.inverse_transform([idx])[0]


class Dataset(object):

    def __init__(self, name, path, ds_type='I', temporal_behaviour='S'):
        self.name = name
        self.df = self._format_df(
            pd.read_csv(path, delimiter=rr.DELIMITER, encoding=rr.ENCODING, quoting=rr.QUOTING, quotechar=rr.QUOTECHAR),
            ds_type,
            temporal_behaviour
        ).drop_duplicates(subset=[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID], keep='last')
        self.ds_type = ds_type        
        self.sparse_repr = None

    def _format_df(self, df, ds_type, temporal_behaviour):
        # Arruma as colunas
        columns = [rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID]
        if ds_type == 'E':
            columns += [rr.COLUMN_INTERACTION]
        if temporal_behaviour == 'T':
            columns += [rr.COLUMN_DATETIME]
        # Formata bases especificas
        if self.name == 'Anime Recommendations':
            df = df[df[rr.COLUMN_INTERACTION]!=-1] if ds_type == 'E' else df[df[rr.COLUMN_INTERACTION]==-1]
        elif self.name == 'Book-Crossing':
            df = df[df[rr.COLUMN_INTERACTION]!=0] if ds_type == 'E' else df[df[rr.COLUMN_INTERACTION]==0]
        # Normaliza a interação
        if ds_type == 'E':
            self.max_rating = max(df[rr.COLUMN_INTERACTION])                        
            self.min_rating = min(df[rr.COLUMN_INTERACTION])
            df[rr.COLUMN_INTERACTION] = ((df[rr.COLUMN_INTERACTION]-self.min_rating)/(self.max_rating-self.min_rating)) * (rr.RATING_SCALE_HI - rr.RATING_SCALE_LO) + rr.RATING_SCALE_LO        
        # Limpa entradas nulas
        df = df[columns].dropna()
        # Limpa entradas duplicatas
        df = df.drop_duplicates(subset=[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID])
        # Trunca o tempo e gera timestamp
        if temporal_behaviour == 'T':
            df[rr.COLUMN_DATETIME] = pd.to_datetime(df[rr.COLUMN_DATETIME])
            df[rr.COLUMN_TIMESTAMP] = df[rr.COLUMN_DATETIME].values.astype(np.int64) // 10 ** 9
        return df        
    
    def get_name(self):
        return self.name

    def get_dataframe(self):
        return self.df

    def get_n_users(self):
        return len(self.df[rr.COLUMN_USER_ID].unique())

    def get_n_items(self):
        return len(self.df[rr.COLUMN_ITEM_ID].unique())

    def get_n_interactions(self):
        return len(self.df)

    def get_min_max_rating(self):
        return (self.min_rating, self.max_rating) if self.ds_type == 'E' else None


# Faz o download dos datasets
def download_datasets(ds_dir='datasets', datasets=None, verbose=True):
    downloaded_datasets = list()
    d = 0
    for _, row in DATASETS_TABLE.iterrows():
        cur_dataset = row[rr.DATASET_NAME]
        if datasets is None or cur_dataset in datasets:
            d += 1
            if verbose:
                print('Fazendo download de {}... ({}/{})'.format(cur_dataset, d, len(datasets) if datasets is not None else len(DATASETS_TABLE)))
            gdd.download_file_from_google_drive(file_id=row[rr.DATASET_GDRIVE], dest_path='./{}/{}.zip'.format(ds_dir, cur_dataset), unzip=True)
            os.remove('./{}/{}.zip'.format(ds_dir, cur_dataset))
            downloaded_datasets.append(cur_dataset)
    if verbose:
        print('Downloads concluidos!')
    return downloaded_datasets


# Recupera um conjunto de datasets, retornando-os um de cada vez
def get_datasets(ds_dir='datasets', datasets=None, ds_type='I', temporal_behaviour='S'):
    # Se nao for informado datasets, recupera de acordo com os parametros
    if datasets is None:
        datasets = DATASETS_TABLE[DATASETS_TABLE[rr.DATASET_TYPE]=='E'] if ds_type == 'E' else DATASETS_TABLE
        datasets = datasets[datasets[rr.DATASET_TEMPORAL_BEHAVIOUR]=='T'] if temporal_behaviour == 'T' else datasets        
    
    # Recupera o nome dos datasets
    if type(datasets) == pd.DataFrame:
        datasets = datasets[rr.DATASET_NAME].values

    # Percorre os datasets
    for ds_name in datasets:
        ds_path = os.path.join(ds_dir, ds_name, rr.FILE_INTERACTIONS)
        if not os.path.exists(ds_path):
            raise Exception("File '{}' does not exist".format(ds_path))
        yield Dataset(ds_name, ds_path, ds_type, temporal_behaviour)


# Recupera um unico dataset
def get_dataset(dataset, ds_dir='datasets', ds_type='I', temporal_behaviour='S'):
    ds_path = os.path.join(ds_dir, dataset, rr.FILE_INTERACTIONS)
    if not os.path.exists(ds_path):
        raise Exception("File '{}' does not exist".format(ds_path))
    return Dataset(dataset, ds_path, ds_type, temporal_behaviour)