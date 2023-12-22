#from fastFM import als, sgd, mcmc
import numpy as np
import os
import pandas as pd
import pywFM
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
#from spotlight.factorization.explicit import ExplicitFactorizationModel
#from spotlight.interactions import Interactions
import surprise
import turicreate as tc

import recresearch as rr
from recresearch.dataset import SparseRepr


class SVDSurprise(object):
	def __init__(self, n_factors, regularization=0.02, learning_rate=0.005, n_epochs=20):
		self.model = surprise.SVD(n_factors=n_factors, lr_all=learning_rate, reg_all=regularization, n_epochs=n_epochs)

	def fit(self, df):
		reader = surprise.Reader()
		data = surprise.Dataset.load_from_df(df[[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID, rr.COLUMN_INTERACTION]], reader)
		self.model.fit(data.build_full_trainset())
		return 

	def predict(self, df):
		predictions = self.model.test(df[[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID, rr.COLUMN_INTERACTION]].values)
		estimative = pd.Series({(p.uid, p.iid): p.est for p in predictions})
		estimative.index.set_names([rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID], level=[0,1], inplace=True)
		return estimative


class SVDSpotlight(object):
	def __init__(self, n_factors, regularization=0.0, learning_rate=0.01, n_epochs=10):
		self.model = ExplicitFactorizationModel(embedding_dim=n_factors, l2=regularization, learning_rate=learning_rate, batch_size=2**15, use_cuda=True, n_iter=n_epochs)

	def fit(self, df):
		# Transforma em matriz esparsa apra capturar indices
		self.sparse_repr = SparseRepr(df)
		sparse_matrix = self.sparse_repr.get_matrix(
			df[rr.COLUMN_USER_ID].values,
			df[rr.COLUMN_ITEM_ID].values,
			df[rr.COLUMN_INTERACTION].values
		).tocoo()

		# Gera as interacoes
		interactions = Interactions(user_ids=sparse_matrix.row, item_ids=sparse_matrix.col, ratings=sparse_matrix.data)

		# Treina o modelo
		self.model.fit(interactions)
		self.sparse_matrix = sparse_matrix.tocsr()

	def predict(self, df):
		# Iniciaiza
		estimative = pd.Series(
			(rr.RATING_SCALE_HI-rr.RATING_SCALE_LO)/2, 
			index=pd.MultiIndex.from_product([[], []],names=[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID])
		)
		# Itera os usuarios
		users = df[rr.COLUMN_USER_ID].unique()
		for user in users:
			user_idx = self.sparse_repr.get_idx_of_user(user)
			
			# Captura os items
			items = df[df[rr.COLUMN_USER_ID]==user][rr.COLUMN_ITEM_ID].unique()	
			items_idx = self.sparse_repr.get_idx_of_item(items)
			# Realiza a predicao
			try:
				user_estimative = pd.Series(
					np.clip(self.model.predict(user_idx, items_idx), rr.RATING_SCALE_LO, rr.RATING_SCALE_HI),
					index=pd.MultiIndex.from_product([[user], items], names=[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID])
				)
			except:
				user_estimative = pd.Series(
					self.sparse_matrix[user_idx, :].data.mean(),
					index=pd.MultiIndex.from_product([[user], items], names=[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID])
				)
			estimative = pd.concat([estimative, user_estimative])
		return estimative


class SVDppSurprise(object):
	def __init__(self, n_factors, regularization=0.02, learning_rate=0.007, n_epochs=20):
		self.model = surprise.SVDpp(n_factors=n_factors, lr_all=learning_rate, reg_all=regularization, n_epochs=n_epochs)

	def fit(self, df):
		reader = surprise.Reader()
		data = surprise.Dataset.load_from_df(df[[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID, rr.COLUMN_INTERACTION]], reader)
		self.model.fit(data.build_full_trainset())
		return 

	def predict(self, df):
		predictions = self.model.test(df[[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID, rr.COLUMN_INTERACTION]].values)
		estimative = pd.Series({(p.uid, p.iid): p.est for p in predictions})
		estimative.index.set_names([rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID], level=[0,1], inplace=True)
		return estimative


class FMLibFM(object):

	TEMP_PATH = 'libfm-temp'

	def __init__(self, n_factors, regularization=0, learning_rate=0.1, solver='mcmc', n_epochs=100):
		os.makedirs(self.TEMP_PATH, exist_ok=True)
		if solver == 'mcmc':
			self.model = pywFM.FM(task='regression', num_iter=n_epochs, temp_path=self.TEMP_PATH, learning_method='mcmc')
		elif solver == 'als':
			self.model = pywFM.FM(task='regression', num_iter=n_epochs, temp_path=self.TEMP_PATH, learning_method='als', r2_regularization=regularization)
		elif solver == 'sgd':
			self.model = pywFM.FM(task='regression', num_iter=n_epochs, temp_path=self.TEMP_PATH, learning_method='sgd', r2_regularization=regularization, learn_rate=learning_rate)
		else:
			raise Exception('Error: Problems on parameters of FMLibFM model')

	def fit(self, df):
		self.user_encoder = OneHotEncoder(categories='auto', handle_unknown='ignore')
		self.item_encoder = OneHotEncoder(categories='auto', handle_unknown='ignore')
		self.X_train = sparse.hstack([
			self.user_encoder.fit_transform(df[rr.COLUMN_USER_ID].values.reshape(-1,1)),
			self.item_encoder.fit_transform(df[rr.COLUMN_ITEM_ID].values.reshape(-1,1))
		])
		self.y_train = df[rr.COLUMN_INTERACTION].values

	def predict(self, df):
		X_test = sparse.hstack([
			self.user_encoder.transform(df[rr.COLUMN_USER_ID].values.reshape(-1,1)),
			self.item_encoder.transform(df[rr.COLUMN_ITEM_ID].values.reshape(-1,1))
		])
		pred = np.clip(self.model.run(self.X_train, self.y_train, X_test, np.zeros(X_test.shape[0])).predictions, rr.RATING_SCALE_LO, rr.RATING_SCALE_HI)
		estimative = pd.Series(pred, index=df.set_index([rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID]).index)
		return estimative


class FMFastFM(object):
	def __init__(self, n_factors, regularization=0, learning_rate=0.1, solver='mcmc', n_epochs=100):
		if solver == 'mcmc':
			self.model = mcmc.FMRegression(n_iter=n_epochs, rank=n_factors)
		elif solver == 'als':
			self.model = als.FMRegression(n_iter=n_epochs, rank=n_factors, l2_reg=regularization)
		elif solver == 'sgd':
			self.model = sgd.FMRegression(n_iter=n_epochs, rank=n_factors, l2_reg=regularization, step_size=learning_rate)
		else:
			raise Exception('Error: Problems on parameters of FMFastFM model')
	
	def fit(self, df):
		self.user_encoder = OneHotEncoder(categories='auto', handle_unknown='ignore')
		self.item_encoder = OneHotEncoder(categories='auto', handle_unknown='ignore')
		X_train = sparse.hstack([
			self.user_encoder.fit_transform(df[rr.COLUMN_USER_ID].values.reshape(-1,1)),
			self.item_encoder.fit_transform(df[rr.COLUMN_ITEM_ID].values.reshape(-1,1))
		])
		y_train = df[rr.COLUMN_INTERACTION].values
		self.model.fit(X_train, y_train)

	def predict(self, df):
		X_test = sparse.hstack([
			self.user_encoder.transform(df[rr.COLUMN_USER_ID].values.reshape(-1,1)),
			self.item_encoder.transform(df[rr.COLUMN_ITEM_ID].values.reshape(-1,1))
		])
		pred = np.clip(self.model.predict(X_test), rr.RATING_SCALE_LO, rr.RATING_SCALE_HI)
		estimative = pd.Series(pred, index=df.set_index([rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID]).index)
		return estimative


class FMStaticTuricreate(object):
	def __init__(self, n_factors, regularization=1e-08, learning_rate=0, n_epochs=50):
		self.n_factors = n_factors
		self.learning_rate = learning_rate
		self.regularization = regularization
		self.n_epochs = n_epochs

	def fit(self, df):				
		sframe = tc.SFrame(df[[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID, rr.COLUMN_INTERACTION]])
		self.model = tc.recommender.factorization_recommender.create(
			observation_data=sframe,
			user_id=rr.COLUMN_USER_ID,
			item_id=rr.COLUMN_ITEM_ID,
			target=rr.COLUMN_INTERACTION,
			num_factors=self.n_factors,
			regularization=self.regularization,
			sgd_step_size=self.learning_rate,
			side_data_factorization=False,
			max_iterations=self.n_epochs
		)

	def predict(self, df):
		sframe = tc.SFrame(df[[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID, rr.COLUMN_INTERACTION]])
		pred = self.model.predict(sframe)
		estimative = df.set_index([rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID])
		estimative[rr.COLUMN_INTERACTION] = pred
		estimative = estimative[rr.COLUMN_INTERACTION]
		return estimative