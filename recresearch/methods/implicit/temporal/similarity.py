import turicreate as tc

import recresearch as rr

class ISTemporalTuricreate(object):
	def __init__(self, k=20):
		self.k = k

	def fit(self, df):
		sframe = tc.SFrame(df[[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID, rr.COLUMN_TIMESTAMP]])
		self.model = tc.recommender.item_similarity_recommender.create(
			observation_data=sframe,
			user_id=rr.COLUMN_USER_ID,
			item_id=rr.COLUMN_ITEM_ID,
			similarity_type='cosine',
			only_top_k=self.k,
			target_memory_usage=rr.MEM_SIZE_LIMIT,
			target=rr.COLUMN_TIMESTAMP
		)

	def predict(self, df, top_n=10):
		recommendations = self.model.recommend(
			users=df[rr.COLUMN_USER_ID].unique(),
			k=top_n,
			exclude_known=True
		).to_dataframe().drop(columns=['score'])
		return recommendations