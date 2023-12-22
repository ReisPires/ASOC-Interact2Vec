from gensim.models import Doc2Vec
from multiprocessing import cpu_count
import numpy as np
import os
import pickle

import recresearch as rr
from recresearch.dataset import SparseRepr

class User2Vec(object):
    
    def generate_embeddings(self, df, embeddings_dir, embeddings_filename, embedding_dim=100, n_epochs=5, negative_sampling=5, negative_exponent=0.75, subsampling_p=1e-3, regularization_lambda=None, batch_size=2**14):
        # Verifica se embeddings ja foram criadas previamente
        sparse_repr_filepath = os.path.join(embeddings_dir, rr.FILE_SPARSE_REPR.format(embeddings_filename))
        user_embeddings_filepath = os.path.join(embeddings_dir, rr.FILE_USER_EMBEDDINGS.format(embeddings_filename))
        item_embeddings_filepath = os.path.join(embeddings_dir, rr.FILE_ITEM_EMBEDDINGS.format(embeddings_filename))
        if os.path.exists(sparse_repr_filepath) and os.path.exists(user_embeddings_filepath) and os.path.exists(item_embeddings_filepath):
            print('Embeddings já criadas...')
            return

        sparse_repr = SparseRepr(df)
        sparse_matrix = sparse_repr.get_matrix(df[rr.COLUMN_USER_ID], df[rr.COLUMN_ITEM_ID])
        
        print('Gerando arquivo de interações...')        
        fid = 0
        interactions_file = 'user2vec_interactions_{}.temp'.format(fid)
        while os.path.exists(interactions_file):
            fid += 1
            interactions_file = 'user2vec_interactions_{}.temp'.format(fid)
        with open(interactions_file, 'w') as f:
            for user in range(sparse_matrix.shape[0]):
                f.write(' '.join(sparse_matrix[user].nonzero()[1].astype(str)) + '\n')
        
        print('Gerando embeddings do User2Vec...')
        model = Doc2Vec(
            corpus_file=interactions_file,
            dm=1,
            vector_size=embedding_dim,
            window=sparse_matrix.sum(axis=1).max()*10000,
            min_count=1,
            max_vocab_size=None,
            sample=0 if subsampling_p is None else subsampling_p,
            workers=cpu_count(),
            epochs=n_epochs,
            hs=0,
            negative=negative_sampling,
            ns_exponent=negative_exponent,
            dm_mean=1,
            dm_concat=0,
            trim_rule=None,
            seed=rr.RANDOM_SEED
        )
        items_embeddings = model.wv.vectors[np.argsort(np.fromiter(model.wv.index2word, dtype=np.int32, count=len(model.wv.index2word)))]        
        users_embeddings = model.docvecs.vectors_docs
        os.remove(interactions_file)

        os.makedirs(embeddings_dir, exist_ok=True)
        pickle.dump(sparse_repr, open(sparse_repr_filepath, 'wb'))
        pickle.dump(users_embeddings, open(user_embeddings_filepath, 'wb'))
        pickle.dump(items_embeddings, open(item_embeddings_filepath, 'wb'))
        print('Embeddings do User2Vec geradas!')