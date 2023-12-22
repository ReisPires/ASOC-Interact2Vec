import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import LabelEncoder
import turicreate as tc

import recresearch as rr
from recresearch.dataset import SparseRepr

class KNNRaw(object):
    def __init__(self, item_based=True, k=20, min_k=1):
        self.item_based = item_based
        self.k = k
        self.min_k = min_k    

    def fit(self, df):
        self.sparse_repr = SparseRepr(df)
        self.ratings_matrix = self.sparse_repr.get_matrix(
            df[rr.COLUMN_USER_ID].values, 
            df[rr.COLUMN_ITEM_ID].values, 
            df[rr.COLUMN_INTERACTION].values
        )
        if self.item_based:
            self.avg_rating = df.groupby(rr.COLUMN_ITEM_ID)[rr.COLUMN_INTERACTION].mean()
        else:
            self.avg_rating = df.groupby(rr.COLUMN_USER_ID)[rr.COLUMN_INTERACTION].mean()

    def predict(self, df):
        # Calcula a media dos itens como baseline
        if self.item_based:
            estimative = pd.Series(self.avg_rating.loc[df[rr.COLUMN_ITEM_ID]].values, index=pd.MultiIndex.from_frame(df[[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID]]))
        else:
            estimative = pd.Series(self.avg_rating.loc[df[rr.COLUMN_USER_ID]].values, index=pd.MultiIndex.from_frame(df[[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID]]))

        # Recupera indice de usuarios e itens alvo
        target_users = df[rr.COLUMN_USER_ID].values
        target_users_idx = self.sparse_repr.get_idx_of_user(target_users)
        target_items = df[rr.COLUMN_ITEM_ID].values
        target_items_idx = self.sparse_repr.get_idx_of_item(target_items)

        # Recupera os vetores dos itens alvo
        target_emb_matrix_encoder = LabelEncoder()
        if self.item_based:
            target_emb_matrix_encoder.fit(target_items_idx)
            target_arrays = self.ratings_matrix.T[target_emb_matrix_encoder.classes_, :]
        else:
            target_emb_matrix_encoder.fit(target_users_idx)
            target_arrays = self.ratings_matrix[target_emb_matrix_encoder.classes_, :]
        
        # Calcula as distancias
        if self.item_based:
            dists = cosine_distances(target_arrays, self.ratings_matrix.T)
        else:
            dists = cosine_distances(target_arrays, self.ratings_matrix)
        np.fill_diagonal(dists, np.inf)

        # Encontra os k vizinhos mais próximos e as distancias
        self.k = min(self.k, len(dists)-1)
        neighbors = np.argsort(dists, axis=1)[:, :self.k]
        neighbors_dists = np.sort(dists, axis=1)[:, :self.k]

        # Recupera as notas dos itens vizinhos e faz a predicao
        if self.item_based:
            neighbors_rating = self.ratings_matrix[
                np.repeat(target_users_idx.reshape(-1,1), self.k, axis=1), 
                neighbors[target_emb_matrix_encoder.transform(target_items_idx), :]
            ]
            weights = neighbors_dists[target_emb_matrix_encoder.transform(target_items_idx)]
        else:
            neighbors_rating = self.ratings_matrix[
                neighbors[target_emb_matrix_encoder.transform(target_users_idx), :],
                np.repeat(target_items_idx.reshape(-1,1), self.k, axis=1)
            ]
            weights = neighbors_dists[target_emb_matrix_encoder.transform(target_users_idx)]
        ratings_sums = neighbors_rating.multiply(weights).sum(axis=1).A1
        ratings_weights = (neighbors_rating!=0).multiply(weights).sum(axis=1).A1
        ratings_counts = np.diff(neighbors_rating.indptr)
        valid = np.where(ratings_counts>=self.min_k)[0]
        estimative.iloc[valid] = ratings_sums[valid] / ratings_weights[valid]
        return estimative


class RNN_Static(object):
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
			user_ratings = self.ratings_matrix[users_idx, :][:, item_neighbors]

			# Gera a predicao pra cada usuario
			for u, user in enumerate(users):
				print('%06d/%06d - %06d/%06d' % (i, len(items), u, len(users)), end='\r', flush=True)
				rated_items = user_ratings[u].nonzero()
				rated_distances = distances[rated_items[1]]
				rated_user_ratings = user_ratings[u].data
				summed_weights = np.sum(rated_distances)
				if summed_weights > 0:
					item_estimative.loc[user, item] = np.sum((rated_user_ratings*rated_distances))/summed_weights
    
			# Adiciona a series de resultados
			estimative = pd.concat([estimative, item_estimative])
		return estimative

#KNN e RNN dependendo da entrada do threshold
class RecencyStaticRaw(object):
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
			user_ratings = self.ratings_matrix[users_idx, :][:, item_neighbors]

			# Gera a predicao pra cada usuario
			for u, user in enumerate(users):
				print('%06d/%06d - %06d/%06d' % (i, len(items), u, len(users)), end='\r', flush=True)
				rated_items = user_ratings[u].nonzero()
				rated_distances = distances[rated_items[1]]
				rated_user_ratings = user_ratings[u].data
				summed_weights = np.sum(rated_distances)
				if summed_weights > 0:
					item_estimative.loc[user, item] = np.sum((rated_user_ratings*rated_distances))/summed_weights
    
			# Adiciona a series de resultados
			estimative = pd.concat([estimative, item_estimative])
		return estimative