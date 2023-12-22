import numpy as np
import os
import pickle 
from sklearn.preprocessing import LabelEncoder

import recresearch as rr
from recresearch.dataset import get_datasets
from recresearch.evaluation.metrics import content_based_ndcg_score, autotagging_score
from recresearch.logger import BasicLogger

DATASETS = ['Anime Recommendations', 'BestBuy', 'DeliciousBookmarks', 'Last.FM - Listened', 'MovieLens-cat', 'MovieLens-tag']
MODELS = ['SVD', 'BPR', 'Item2Vec', 'User2Vec']
EMBEDDING_DIR = 'embeddings/full'

results_log = BasicLogger('content_eval_cbndcg.results')

for dataset in DATASETS:
    print('Lendo matrizes de conteúdo')
    content_matrix = pickle.load(open('./content_matrices/{}_matrix.pkl'.format(dataset), 'rb'))
    content_encoder = pickle.load(open('./content_matrices/{}_encoder.pkl'.format(dataset), 'rb'))

    print('Limpando entradas zeradas')
    valid_items = content_encoder.inverse_transform(np.unique(content_matrix.nonzero()[0]))    
    valid_encoder = LabelEncoder()
    valid_encoder.fit(valid_items)
    content_matrix = content_matrix[content_encoder.transform(valid_encoder.classes_)]

    for model in MODELS:
        print('Lendo Embeddings')
        if dataset in ['MovieLens-cat', 'MovieLens-tag']:            
            embeddings = pickle.load(open(os.path.join(EMBEDDING_DIR, '{}_{}_item_embeddings.pkl'.format('MovieLens', model)), 'rb'))
            sparse_repr = pickle.load(open(os.path.join(EMBEDDING_DIR, '{}_{}_sparse_repr.pkl'.format('MovieLens', model)), 'rb'))
        else:
            embeddings = pickle.load(open(os.path.join(EMBEDDING_DIR, '{}_{}_item_embeddings.pkl'.format(dataset, model)), 'rb'))
            sparse_repr = pickle.load(open(os.path.join(EMBEDDING_DIR, '{}_{}_sparse_repr.pkl'.format(dataset, model)), 'rb'))
        
        common_items = list(set(valid_encoder.classes_) & set(sparse_repr._item_encoder.classes_))
            
        content_sort = valid_encoder.transform(common_items)        
        embeddings_sort = sparse_repr._item_encoder.transform(common_items)        

        #print('Calculando CB-NDCG')        
        cbndcg = content_based_ndcg_score(content_matrix[content_sort], embeddings[embeddings_sort], [10, 25, 50, 100])
        for top_n, cbndcg_val in cbndcg.items():
           results_log.log('{};{};{};{}'.format(dataset, model, top_n, cbndcg_val))

        print('Realizando auto-tagging')
        precision, recall = autotagging_score(content_matrix[content_sort], embeddings[embeddings_sort], list(range(1, 100, 4)))
        os.makedirs('autotagging', exist_ok=True)
        pickle.dump(precision, open(os.path.join('autotagging', '{}_{}_precision.pkl'.format(dataset, model)), 'wb'))
        pickle.dump(recall, open(os.path.join('autotagging', '{}_{}_recall.pkl'.format(dataset, model)), 'wb'))