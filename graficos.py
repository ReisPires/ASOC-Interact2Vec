from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

import recresearch as rr
from recresearch.dataset import get_datasets
from recresearch.utils.model_selection import recsys_train_test_split
from recresearch.dataset import get_dataset
from recresearch.utils.preprocessing import recsys_sampling, cut_by_minimal_interactions, remove_cold_start

def dif_media(df):
    return (np.diff(df)/np.timedelta64(1,'D')).mean()

n_bins = 50

# Regras do split
train_size = 0.8
val_size = 0.1
test_size = 0.1

#dataset = get_dataset('CiaoDVD', temporal_behaviour='T')
#dataset = get_dataset('DeliciousBookmarks', temporal_behaviour='T')
#dataset = get_dataset('RetailRocket-All', temporal_behaviour='T')
dataset = get_dataset('RetailRocket-Transactions', temporal_behaviour='T')
dataset_name = dataset.get_name()
df = dataset.get_dataframe()

# Separa o treino, validacao e teste
df_train, df_val, df_test = recsys_train_test_split(df, train_size, val_size, test_size, temporal_behaviour='T')
# Realiza undersampling para otimizacao de parametros
df_train_gs = recsys_sampling(df_train, dataset_name, rr.SAMPLING_RATE_HYPERPARAMETERIZATION)
# Remove usuarios com uma unica interacao
df_train_gs = cut_by_minimal_interactions(df_train_gs, min_interactions=2)
# Remove cold start da validacao
df_val_gs = remove_cold_start(df_train_gs, df_val)

df_train = df_train_gs
df_val = df_val_gs

#==============================TREINO==============================
#Base de cor: Azul
#TONS: 'deepskyblue' 'lightskyblue' 'lightblue' 'skyblue'
#==========Diferença entre a primeira e a última interação de cada usuário==========
first_last_user = df_train.groupby(rr.COLUMN_USER_ID).agg({rr.COLUMN_DATETIME :['min','max']}).values
diff = np.diff(first_last_user)/np.timedelta64(1,'D')
plt.hist(diff, bins=n_bins, color='deepskyblue')
plt.title(dataset_name+': Base de TREINO')
plt.suptitle('Diferença entre a primeira e a última interação de cada usuário')
# ate 35,5 se mantem
plt.show()
#==========Quantidade de itens consumidos por cada usuário==========
item_user = df_train.groupby(rr.COLUMN_USER_ID)[rr.COLUMN_ITEM_ID].size().values
plt.hist(item_user, bins=n_bins, color='deepskyblue')
plt.title(dataset_name+': Base de TREINO')
plt.suptitle('Quantidade de itens consumidos por cada usuário')
plt.show()
#==========Quantidade de usuários que interegiram com cada item==========
item_user = df_train.groupby(rr.COLUMN_ITEM_ID)[rr.COLUMN_USER_ID].size().values
plt.hist(item_user, bins=n_bins, color='deepskyblue')
plt.title(dataset_name+': Base de TREINO')
plt.suptitle('Quantidade de usuários que interegiram com cada item')
plt.show()
#==========Diferença média de dias entre interações do mesmo usuário==========
df_order = df_train.sort_values(by=rr.COLUMN_DATETIME)
diff_mean = df_order.groupby(rr.COLUMN_USER_ID).agg({rr.COLUMN_DATETIME: [dif_media]}).values
plt.hist(diff_mean, bins=n_bins, color='deepskyblue')
plt.title(dataset_name+': Base de TREINO')
plt.suptitle('Diferença média de dias entre interações do mesmo usuário')
plt.show()

#VALIDAÇÃO
#Base de cor: Vermelho
#TONS: 'coral' 'lightcoral' 'red' 'darkred'
#==========Diferença entre a primeira e a última interação de cada usuário==========
first_last_user = df_val.groupby(rr.COLUMN_USER_ID).agg({rr.COLUMN_DATETIME :['min','max']}).values
diff = np.diff(first_last_user)/np.timedelta64(1,'D')
plt.hist(diff, bins=n_bins, color='coral')
plt.title(dataset_name+': Base de VALIDAÇÃO')
plt.suptitle('Diferença entre a primeira e a última interação de cada usuário')
# ate 35,5 se mantem
plt.show()
#==========Quantidade de itens consumidos por cada usuário==========
item_user = df_val.groupby(rr.COLUMN_USER_ID)[rr.COLUMN_ITEM_ID].size().values
plt.hist(item_user, bins=n_bins, color='coral')
plt.title(dataset_name+': Base de VALIDAÇÃO')
plt.suptitle('Quantidade de itens consumidos por cada usuário')
plt.show()
#==========Quantidade de usuários que interegiram com cada item==========
item_user = df_val.groupby(rr.COLUMN_ITEM_ID)[rr.COLUMN_USER_ID].size().values
plt.hist(item_user, bins=n_bins, color='coral')
plt.title(dataset_name+': Base de VALIDAÇÃO')
plt.suptitle('Quantidade de usuários que interegiram com cada item')
plt.show()
#==========Diferença média de dias entre interações do mesmo usuário==========
df_order = df_val.sort_values(by=rr.COLUMN_DATETIME)
diff_mean = df_order.groupby(rr.COLUMN_USER_ID).agg({rr.COLUMN_DATETIME: [dif_media]}).values
plt.hist(diff_mean, bins=n_bins, color='coral')
plt.title(dataset_name+': Base de VALIDAÇÃO')
plt.suptitle('Diferença média de dias entre interações do mesmo usuário')
plt.show()

#TESTE
#Base de cor: Verde
#TONS: 'green' 'darkgreen' 'limegreen' 'lime'
#==========Diferença entre a primeira e a última interação de cada usuário==========
first_last_user =df_test.groupby(rr.COLUMN_USER_ID).agg({rr.COLUMN_DATETIME :['min','max']}).values
diff = np.diff(first_last_user)/np.timedelta64(1,'D')
plt.hist(diff, bins=n_bins, color='green')
plt.title(dataset_name+': Base de TESTE')
plt.suptitle('Diferença entre a primeira e a última interação de cada usuário')
plt.show()
#==========Quantidade de itens consumidos por cada usuário==========
item_user = df_test.groupby(rr.COLUMN_USER_ID)[rr.COLUMN_ITEM_ID].size().values
plt.hist(item_user, bins=n_bins, color='green')
plt.title(dataset_name+': Base de TESTE')
plt.suptitle('Quantidade de itens consumidos por cada usuário')
plt.show()
#==========Quantidade de usuários que interegiram com cada item==========
item_user = df_test.groupby(rr.COLUMN_ITEM_ID)[rr.COLUMN_USER_ID].size().values
plt.hist(item_user, bins=n_bins, color='green')
plt.title(dataset_name+': Base de TESTE')
plt.suptitle('Quantidade de usuários que interegiram com cada item')
plt.show()
#==========Diferença média de dias entre interações do mesmo usuário==========
df_order = df_test.sort_values(by=rr.COLUMN_DATETIME)
diff_mean = df_order.groupby(rr.COLUMN_USER_ID).agg({rr.COLUMN_DATETIME: [dif_media]}).values
plt.hist(diff_mean, bins=n_bins, color='green')
plt.title(dataset_name+': Base de TESTE')
plt.suptitle('Diferença média de dias entre interações do mesmo usuário')
plt.show()