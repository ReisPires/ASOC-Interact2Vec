import numpy as np
import pandas as pd
import recresearch as rr

class Baseline(object):
    def __init__(self, item_based=True):
        self.item_based = item_based

    def fit(self, df):
        if self.item_based:
            self.means = df.groupby(rr.COLUMN_ITEM_ID)[rr.COLUMN_INTERACTION].mean()
        else:
            self.means = df.groupby(rr.COLUMN_USER_ID)[rr.COLUMN_INTERACTION].mean()        
  
    def predict(self, df):
        estimative = pd.merge(
            df[[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID]], 
            self.means.to_frame(rr.COLUMN_INTERACTION), 
            how='left', 
            left_on=rr.COLUMN_ITEM_ID if self.item_based else rr.COLUMN_USER_ID,
            right_index=True
        )
        estimative = estimative.set_index([rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID])[rr.COLUMN_INTERACTION]
        estimative.index.levels[0].name = rr.COLUMN_USER_ID
        estimative.index.levels[1].name = rr.COLUMN_ITEM_ID
        return estimative
