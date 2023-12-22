import implicit
import os
import pickle

from recresearch.dataset import SparseRepr
import recresearch as rr

class ALSEmbeddingsImplicit(object):
    def generate_embeddings(self, df, embeddings_dir, embeddings_filename, n_factors=100, n_epochs=20, regularization=0.01, learning_rate=0.025, verbose=True):
        # Verifica se embeddings ja foram criadas previamente
        sparse_repr_filepath = os.path.join(embeddings_dir, rr.FILE_SPARSE_REPR.format(embeddings_filename))
        user_embeddings_filepath = os.path.join(embeddings_dir, rr.FILE_USER_EMBEDDINGS.format(embeddings_filename))
        item_embeddings_filepath = os.path.join(embeddings_dir, rr.FILE_ITEM_EMBEDDINGS.format(embeddings_filename))
        if os.path.exists(sparse_repr_filepath) and os.path.exists(user_embeddings_filepath) and os.path.exists(item_embeddings_filepath):
            print('Embeddings já criadas...')
            return

        model = implicit.als.AlternatingLeastSquares(
            factors=n_factors,
            regularization=regularization,
            use_gpu=True,
            iterations=n_epochs
        )   
        sparse_repr = SparseRepr(df)
        sparse_matrix = sparse_repr.get_matrix(df[rr.COLUMN_USER_ID].values, df[rr.COLUMN_ITEM_ID].values)
        model.fit(sparse_matrix.T)

        items_embeddings = model.item_factors
        users_embeddings = model.user_factors
        
        os.makedirs(embeddings_dir, exist_ok=True)
        pickle.dump(sparse_repr, open(sparse_repr_filepath, 'wb'))
        pickle.dump(users_embeddings, open(user_embeddings_filepath, 'wb'))
        pickle.dump(items_embeddings, open(item_embeddings_filepath, 'wb'))
        return sparse_repr, items_embeddings, users_embeddings


class BPREmbeddingsImplicit(object):
    def generate_embeddings(self, df, embeddings_dir, embeddings_filename, n_factors=100, n_epochs=100, regularization=0.01, learning_rate=0.01, verbose=True):
        # Verifica se embeddings ja foram criadas previamente
        sparse_repr_filepath = os.path.join(embeddings_dir, rr.FILE_SPARSE_REPR.format(embeddings_filename))
        user_embeddings_filepath = os.path.join(embeddings_dir, rr.FILE_USER_EMBEDDINGS.format(embeddings_filename))
        item_embeddings_filepath = os.path.join(embeddings_dir, rr.FILE_ITEM_EMBEDDINGS.format(embeddings_filename))
        if os.path.exists(sparse_repr_filepath) and os.path.exists(user_embeddings_filepath) and os.path.exists(item_embeddings_filepath):
            print('Embeddings já criadas...')
            return

        model = implicit.bpr.BayesianPersonalizedRanking(
            factors=n_factors,
            learning_rate=learning_rate,
            regularization=regularization,
            use_gpu=True,
            iterations=n_epochs
        )   
        sparse_repr = SparseRepr(df)
        sparse_matrix = sparse_repr.get_matrix(df[rr.COLUMN_USER_ID].values, df[rr.COLUMN_ITEM_ID].values)
        model.fit(sparse_matrix.T)

        items_embeddings = model.item_factors
        users_embeddings = model.user_factors
        
        os.makedirs(embeddings_dir, exist_ok=True)
        pickle.dump(sparse_repr, open(sparse_repr_filepath, 'wb'))
        pickle.dump(users_embeddings, open(user_embeddings_filepath, 'wb'))
        pickle.dump(items_embeddings, open(item_embeddings_filepath, 'wb'))
        return sparse_repr, items_embeddings, users_embeddings