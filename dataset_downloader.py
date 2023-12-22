import numpy as np
import os

import recresearch as rr
from recresearch.dataset import download_datasets

DATASET_DIR = 'datasets'

datasets = download_datasets(ds_dir=DATASET_DIR)

for dataset in datasets:
    dir_path = os.path.join(DATASET_DIR, dataset)
    explicit_interactions = os.path.join(dir_path, 'interactions_explicit.csv')
    implicit_interactions = os.path.join(dir_path, 'interactions_implicit.csv')
    recurrent_interactions = os.path.join(dir_path, 'interactions_recurrent.csv')
    final_interactions = os.path.join(dir_path, rr.FILE_INTERACTIONS)
    
    if dataset in ['Book-Crossing', 'Anime Recommendations']:
        if os.path.exists(explicit_interactions):
            os.remove(explicit_interactions)
        if os.path.exists(implicit_interactions):
            os.rename(implicit_interactions, final_interactions)
    
    elif os.path.exists(explicit_interactions):        
        os.rename(explicit_interactions, final_interactions)
        os.remove(implicit_interactions)
    
    elif os.path.exists(implicit_interactions):
        os.rename(implicit_interactions, final_interactions)
    
    if 'interactions_recurrent.csv' in os.listdir(dir_path):
        os.remove(recurrent_interactions)
