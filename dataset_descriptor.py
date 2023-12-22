import csv
import os
import pandas as pd
from scipy.stats import iqr

import recresearch as rr
from recresearch.dataset import get_datasets
from recresearch.logger import dataframe_to_latex
from recresearch.utils import cut_by_min_number_of_interactions

DESCRIPTOR_BASE_FILE_NAME = 'descriptor'
MIN_INTERACTIONS_CUT = [1, 2]

descriptors = dict()
for min_interactions in MIN_INTERACTIONS_CUT:
    descriptor_name = '{}_{}_'.format(DESCRIPTOR_BASE_FILE_NAME , min_interactions)
    for table_type in ['general', 'user', 'item']:
        descriptors[descriptor_name+table_type] = list()

# Percorre todos os datasets
for dataset in get_datasets(ds_dir='datasets', ds_type='I'):
    
    # Gera descricoes diferentes removendo usuarios e itens de acordo com numero de interacoes
    for min_interactions in MIN_INTERACTIONS_CUT:
        
        # Gera o nome do arquivo
        descriptor_name = '{}_{}_'.format(DESCRIPTOR_BASE_FILE_NAME , min_interactions)

        # Recupera o nome e o dataframe
        ds_name = dataset.get_name()
        df = dataset.get_dataframe()
        
        # Corta o dataframe
        df = cut_by_min_number_of_interactions(df, min_interactions=min_interactions)

        # Dados gerais
        n_users = dataset.get_n_users()
        n_items = dataset.get_n_items()    
        n_interactions = dataset.get_n_interactions()
        esparsity = (1-(n_interactions/(n_users*n_items)))*100
        descriptors[descriptor_name+'general'].append([ds_name, n_users, n_items, n_interactions, esparsity])

        # Interacoes por usuario
        interactions_by_user = df.groupby(rr.COLUMN_USER_ID).size()
        interactions_user_median = interactions_by_user.median()
        interactions_user_q1 = interactions_by_user.quantile(0.25)
        interactions_user_q3 = interactions_by_user.quantile(0.75)
        interactions_user_min = interactions_by_user.min()
        interactions_user_max = interactions_by_user.max()
        descriptors[descriptor_name+'user'].append([ds_name, interactions_user_min, interactions_user_q1, interactions_user_median, interactions_user_q3, interactions_user_max])
        
        # Interacoes por item
        interactions_by_item = df.groupby(rr.COLUMN_ITEM_ID).size()
        interactions_item_median = interactions_by_item.median()
        interactions_item_q1 = interactions_by_item.quantile(0.25)
        interactions_item_q3 = interactions_by_item.quantile(0.75)
        interactions_item_min = interactions_by_item.min()
        interactions_item_max = interactions_by_item.max()
        descriptors[descriptor_name+'item'].append([ds_name, interactions_item_min, interactions_item_q1, interactions_item_median, interactions_item_q3, interactions_item_max])
        
# Loga resultados
for min_interactions in MIN_INTERACTIONS_CUT:
    descriptor_name = '{}_{}_'.format(DESCRIPTOR_BASE_FILE_NAME , min_interactions)
    dataframe_to_latex(descriptor_name+'general.tex', pd.DataFrame(descriptors[descriptor_name+'general'], columns=['Dataset', 'Usuarios', 'Itens', 'Interacoes', 'Esparsidade']))
    dataframe_to_latex(descriptor_name+'user.tex', pd.DataFrame(descriptors[descriptor_name+'user'], columns=['Dataset', 'Min', 'Q1', 'Mediana', 'Q3', 'Max']))
    dataframe_to_latex(descriptor_name+'item.tex', pd.DataFrame(descriptors[descriptor_name+'item'], columns=['Dataset', 'Min', 'Q1', 'Mediana', 'Q3', 'Max']))

print('Tabela de datasets criada!')
