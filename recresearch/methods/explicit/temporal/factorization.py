import pandas as pd
import turicreate as tc

import recresearch as rr

class FMTemporalTuricreate(object):
    def __init__(self, n_factors, regularization=1e-08, learning_rate=0, n_epochs=50):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs

    def fit(self, df):
        sframe = tc.SFrame(df[[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID, rr.COLUMN_INTERACTION, rr.COLUMN_TIMESTAMP]])
        self.model = tc.recommender.factorization_recommender.create(
            observation_data=sframe,
            user_id=rr.COLUMN_USER_ID,
            item_id=rr.COLUMN_ITEM_ID,
            target=rr.COLUMN_INTERACTION,
            num_factors=self.n_factors,
            regularization=self.regularization,
            sgd_step_size=self.learning_rate,
            side_data_factorization=True,
            max_iterations=self.n_epochs
        )

    def predict(self, df):
        sframe = tc.SFrame(df[[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID, rr.COLUMN_INTERACTION, rr.COLUMN_TIMESTAMP]])
        pred = self.model.predict(sframe)
        estimative = pd.Series({(row[rr.COLUMN_USER_ID], row[rr.COLUMN_ITEM_ID]): pred[i] for i, (_, row) in enumerate(df.iterrows())})
        estimative.index.levels[0].name = rr.COLUMN_USER_ID
        estimative.index.levels[1].name = rr.COLUMN_ITEM_ID
        return estimative