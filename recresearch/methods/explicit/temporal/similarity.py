import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import LabelEncoder

import recresearch as rr
from recresearch.dataset import SparseRepr

class RNN_Temp(object):
	def __init__(self, threshold=0.5, alpha=1):
		self.threshold = threshold
		self.alpha = alpha
		self.r_scale = rr.RATING_SCALE_HI-rr.RATING_SCALE_LO+1

	def fit(self, df):		
		# Media e Qt de avaliacoes
		df = df.merge(df.groupby(rr.COLUMN_USER_ID).agg({rr.COLUMN_INTERACTION: ['mean', 'size']})[rr.COLUMN_INTERACTION], 'left', left_on=rr.COLUMN_USER_ID, right_index=True)
		# Diferenca
		df['diff'] = df[rr.COLUMN_INTERACTION]-df['mean']
		# Diferenca ao quadrado
		df['mean_sqr'] = df['diff'].pow(2)
		# Soma da diferenca ao quadrado
		df = df.merge(df.groupby(rr.COLUMN_USER_ID)['mean_sqr'].sum().to_frame('mean_sqr_sum'),'left', left_on=rr.COLUMN_USER_ID, right_index=True)
		# Preferencia
		df['preference'] = ((df[rr.COLUMN_INTERACTION]-df['mean'])/np.sqrt(df['mean_sqr_sum']/(df['size']-1))).fillna(0)

		# Representacoes esparsas
		users = df[rr.COLUMN_USER_ID].values
		items = df[rr.COLUMN_ITEM_ID].values
		self.sparse_repr = SparseRepr(df)
		self.preferences_matrix = self.sparse_repr.get_matrix(users, items, df['preference'].values)
		self.ratings_matrix = self.sparse_repr.get_matrix(users, items, df[rr.COLUMN_INTERACTION].values)
		self.timestamp_matrix = self.sparse_repr.get_matrix(users, items, df[rr.COLUMN_TIMESTAMP].values)		

		# Limpa colunas auxiliares
		df = df.drop(columns=['mean', 'size', 'diff', 'mean_sqr', 'mean_sqr_sum', 'preference'])

	def predict(self, df):
		# Inicializa
		estimative = pd.Series((rr.RATING_SCALE_HI-rr.RATING_SCALE_LO)/2, index=pd.MultiIndex.from_product([[], []], names=[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID]))

		# Itera os items
		items = df[rr.COLUMN_ITEM_ID].unique()
		for i, item in enumerate(items, start=1):
			# Inicializa
			item_idx = self.sparse_repr.get_idx_of_item(item)
			if item_idx is None:
				continue
			users = df[df[rr.COLUMN_ITEM_ID]==item][rr.COLUMN_USER_ID].unique()
			item_estimative = pd.Series(self.ratings_matrix[:, item_idx].data.mean(), index=pd.MultiIndex.from_product([users, [item]], names=[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID]))

			# Calcula as distancias entre itens
			interacted_users = self.preferences_matrix[:, item_idx].nonzero()[0]
			item_neighbors = np.setdiff1d(self.preferences_matrix[interacted_users, :].nonzero()[1], [item_idx])
			if len(item_neighbors) == 0:
				continue
			item_preferences = np.tile(self.preferences_matrix[interacted_users, item_idx].toarray(), len(item_neighbors))
			neighbors_preferences = self.preferences_matrix[interacted_users, :][:, item_neighbors].toarray()
			distances = np.power(np.e, -np.abs(np.subtract(
				item_preferences, 
				neighbors_preferences, 
				out=np.zeros(item_preferences.shape), 
				where=neighbors_preferences!=0
			)).sum(axis=0))

			# Filtra aqueles que tem a distancia acima do threshold
			item_neighbors = item_neighbors[distances>=self.threshold]
			distances = distances[distances>=self.threshold]
			if len(item_neighbors) == 0:
				estimative = pd.concat([estimative, item_estimative])
				continue			
			# Calcula as infos relacionadas aos usuarios
			users_idx = self.sparse_repr.get_idx_of_user(users)
			recent_items = np.argmax(self.timestamp_matrix[users_idx, :][:, item_neighbors], axis=1).A1
			user_ratings = self.ratings_matrix[users_idx, :][:, item_neighbors]
			recent_ratings = user_ratings[(np.arange(len(users)), recent_items)]
			weights = np.power(1-np.abs(user_ratings-recent_ratings.T)/self.r_scale, self.alpha)

			# Gera a predicao pra cada usuario
			for u, user in enumerate(users):
				print('%06d/%06d - %06d/%06d' % (i, len(items), u, len(users)), end='\r', flush=True)
				rated_items = user_ratings[u].nonzero()
				rated_weights = weights[rated_items].A1
				rated_distances = distances[rated_items[1]]
				rated_user_ratings = user_ratings[u].data
				summed_weights = np.sum(rated_distances*rated_weights)
				if summed_weights > 0:
					item_estimative.loc[user, item] = np.sum((rated_user_ratings*rated_distances*rated_weights))/summed_weights
    
			# Adiciona a series de resultados
			estimative = pd.concat([estimative, item_estimative])
		return estimative

#KNN e RNN dependendo da entrada do threshold
class RecencyTemporalRaw(object):
	def __init__(self, threshold=None, alpha=1):
		self.threshold = threshold
		self.alpha = alpha
		self.r_scale = rr.RATING_SCALE_HI-rr.RATING_SCALE_LO+1

	def fit(self, df):		
		# Media e Qt de avaliacoes
		df = df.merge(df.groupby(rr.COLUMN_USER_ID).agg({rr.COLUMN_INTERACTION: ['mean', 'size']})[rr.COLUMN_INTERACTION], 'left', left_on=rr.COLUMN_USER_ID, right_index=True)
		# Diferenca
		df['diff'] = df[rr.COLUMN_INTERACTION]-df['mean']
		# Diferenca ao quadrado
		df['mean_sqr'] = df['diff'].pow(2)
		# Soma da diferenca ao quadrado
		df = df.merge(df.groupby(rr.COLUMN_USER_ID)['mean_sqr'].sum().to_frame('mean_sqr_sum'),'left', left_on=rr.COLUMN_USER_ID, right_index=True)
		# Preferencia
		df['preference'] = ((df[rr.COLUMN_INTERACTION]-df['mean'])/np.sqrt(df['mean_sqr_sum']/(df['size']-1))).fillna(0)

		# Representacoes esparsas
		users = df[rr.COLUMN_USER_ID].values
		items = df[rr.COLUMN_ITEM_ID].values
		self.sparse_repr = SparseRepr(df)
		self.preferences_matrix = self.sparse_repr.get_matrix(users, items, df['preference'].values)
		self.ratings_matrix = self.sparse_repr.get_matrix(users, items, df[rr.COLUMN_INTERACTION].values)
		self.timestamp_matrix = self.sparse_repr.get_matrix(users, items, df[rr.COLUMN_TIMESTAMP].values)		

		# Limpa colunas auxiliares
		df = df.drop(columns=['mean', 'size', 'diff', 'mean_sqr', 'mean_sqr_sum', 'preference'])

	def predict(self, df):
		# Inicializa
		estimative = pd.Series((rr.RATING_SCALE_HI-rr.RATING_SCALE_LO)/2, index=pd.MultiIndex.from_product([[], []], names=[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID]))

		# Itera os items
		items = df[rr.COLUMN_ITEM_ID].unique()
		for i, item in enumerate(items, start=1):
			# Inicializa
			item_idx = self.sparse_repr.get_idx_of_item(item)
			if item_idx is None:
				continue
			users = df[df[rr.COLUMN_ITEM_ID]==item][rr.COLUMN_USER_ID].unique()
			item_estimative = pd.Series(self.ratings_matrix[:, item_idx].data.mean(), index=pd.MultiIndex.from_product([users, [item]], names=[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID]))

			# Calcula as distancias entre itens
			interacted_users = self.preferences_matrix[:, item_idx].nonzero()[0]
			item_neighbors = np.setdiff1d(self.preferences_matrix[interacted_users, :].nonzero()[1], [item_idx])
			if len(item_neighbors) == 0:
				continue
			item_preferences = np.tile(self.preferences_matrix[interacted_users, item_idx].toarray(), len(item_neighbors))
			neighbors_preferences = self.preferences_matrix[interacted_users, :][:, item_neighbors].toarray()
			distances = np.power(np.e, -np.abs(np.subtract(
				item_preferences, 
				neighbors_preferences, 
				out=np.zeros(item_preferences.shape), 
				where=neighbors_preferences!=0
			)).sum(axis=0))

			#filtra os itens e a distancia, acima do threshold ou os k mais proximos
			#caso threshold
			if(self.threshold < 1):
				item_neighbors = item_neighbors[distances>=self.threshold]
				distances = distances[distances>=self.threshold]
			#caso k vizinhos
			else:
				item_neighbors = item_neighbors[np.argsort(distances)][:self.threshold]
				distances = np.sort(distances)[:self.threshold]
			
			
			if len(item_neighbors) == 0:
				estimative = pd.concat([estimative, item_estimative])
				continue
			# Calcula as infos relacionadas aos usuarios
			users_idx = self.sparse_repr.get_idx_of_user(users)
			recent_items = np.argmax(self.timestamp_matrix[users_idx, :][:, item_neighbors], axis=1).A1
			user_ratings = self.ratings_matrix[users_idx, :][:, item_neighbors]
			recent_ratings = user_ratings[(np.arange(len(users)), recent_items)]
			weights = np.power(1-np.abs(user_ratings-recent_ratings.T)/self.r_scale, self.alpha)

			# Gera a predicao pra cada usuario
			for u, user in enumerate(users):
				print('%06d/%06d - %06d/%06d' % (i, len(items), u, len(users)), end='\r', flush=True)
				rated_items = user_ratings[u].nonzero()
				rated_weights = weights[rated_items].A1
				rated_distances = distances[rated_items[1]]
				rated_user_ratings = user_ratings[u].data
				summed_weights = np.sum(rated_distances*rated_weights)
				if summed_weights > 0:
					item_estimative.loc[user, item] = np.sum((rated_user_ratings*rated_distances*rated_weights))/summed_weights
    
			# Adiciona a series de resultados
			estimative = pd.concat([estimative, item_estimative])
		return estimative