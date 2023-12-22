import numpy as np
import os
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.preprocessing import LabelEncoder
import turicreate as tc

import recresearch as rr

from recresearch.dataset import SparseRepr
from recresearch.evaluation.metrics import ndcg_score
from recresearch.utils.model_selection import recsys_train_test_split
from recresearch.utils.preprocessing import cut_by_minimal_interactions


def filter_embeddings(df, sparse_repr, item_embeddings, user_embeddings=None):
    # Recupera universo conhecido
    known_users = df[rr.COLUMN_USER_ID].unique()
    known_items = df[rr.COLUMN_ITEM_ID].unique()
    # Verifica se ha necessidade de filtrar embeddings
    if len(known_items) == len(item_embeddings) and (user_embeddings is None or len(known_users) == len(user_embeddings)):
        new_sparse_repr = sparse_repr
        new_item_embeddings = item_embeddings
        new_user_embeddings = user_embeddings
    else:
        # Recupera indices antigos
        old_items_idx = sparse_repr.get_idx_of_item(known_items)
        old_users_idx = sparse_repr.get_idx_of_user(known_users)
        # Gera indices novos
        new_sparse_repr = SparseRepr(users=known_users, items=known_items)
        new_items_idx = new_sparse_repr.get_idx_of_item(known_items)
        new_users_idx = new_sparse_repr.get_idx_of_user(known_users)
        # Reorganiza embeddings
        new_item_embeddings = item_embeddings[old_items_idx[np.argsort(new_items_idx)], :]
        if user_embeddings is not None:
            new_user_embeddings = user_embeddings[old_users_idx[np.argsort(new_users_idx)], :]
    # Retorna embeddings filtradas
    if user_embeddings is None:
        return new_sparse_repr, new_item_embeddings
    else:
        return new_sparse_repr, new_item_embeddings, new_user_embeddings



class ExplicitKNN(object):
    def __init__(self, embeddings_dir, embeddings_filename, item_based=True, k=20, min_k=1):
        self.item_based = item_based
        self.k = k
        self.min_k = min_k
        self.sparse_repr = pickle.load(open(os.path.join(embeddings_dir, rr.FILE_SPARSE_REPR.format(embeddings_filename)), 'rb'))
        if item_based:
            self.embeddings = pickle.load(open(os.path.join(embeddings_dir, rr.FILE_ITEM_EMBEDDINGS.format(embeddings_filename)), 'rb'))
        else:
            self.embeddings = pickle.load(open(os.path.join(embeddings_dir, rr.FILE_USER_EMBEDDINGS.format(embeddings_filename)), 'rb'))

    def fit(self, df):        
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
        # Calcula a media como baseline
        if self.item_based:
            estimative = pd.Series(self.avg_rating.loc[df[rr.COLUMN_ITEM_ID]].values, index=pd.MultiIndex.from_frame(df[[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID]]))
        else:
            estimative = pd.Series(self.avg_rating.loc[df[rr.COLUMN_USER_ID]].values, index=pd.MultiIndex.from_frame(df[[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID]]))
        
        # Recupera indice de usuarios e itens alvo
        target_users = df[rr.COLUMN_USER_ID].values
        target_users_idx = self.sparse_repr.get_idx_of_user(target_users)
        target_items = df[rr.COLUMN_ITEM_ID].values
        target_items_idx = self.sparse_repr.get_idx_of_item(target_items)

        # Recupera as embeddings dos alvos
        target_emb_matrix_encoder = LabelEncoder()
        if self.item_based:
            target_emb_matrix_encoder.fit(target_items_idx)
        else:
            target_emb_matrix_encoder.fit(target_users_idx)
        target_embeddings = self.embeddings[target_emb_matrix_encoder.classes_, :]

        # Calcula as distancias
        dists = cosine_distances(target_embeddings, self.embeddings)
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



class ImplicitKNN(object):
    def __init__(self, embeddings_dir, embeddings_filename, k=64):        
        self.k = k
        self.sparse_repr = pickle.load(open(os.path.join(embeddings_dir, rr.FILE_SPARSE_REPR.format(embeddings_filename)), 'rb'))        
        self.embeddings = pickle.load(open(os.path.join(embeddings_dir, rr.FILE_ITEM_EMBEDDINGS.format(embeddings_filename)), 'rb'))

    def fit(self, df):
        n_items = self.embeddings.shape[0]
        items_per_batch = int(rr.MEM_SIZE_LIMIT / (8 * n_items))
        nearest_neighbors = np.empty((n_items, self.k))
        nearest_sims = np.empty((n_items, self.k))
        for i in range(0, n_items, items_per_batch):
            batch_sims = cosine_similarity(self.embeddings[i:i+items_per_batch], self.embeddings)
            np.fill_diagonal(batch_sims[:, i:i+items_per_batch], -np.inf)
            nearest_neighbors[i:i+items_per_batch] = np.flip(np.argsort(batch_sims, axis=1), axis=1)[:, :self.k]
            nearest_sims[i:i+items_per_batch] = np.flip(np.sort(batch_sims, axis=1), axis=1)[:, :self.k]
        sim_table = tc.SFrame({
            'id_item': self.sparse_repr.get_item_of_idx(np.repeat(np.arange(n_items), self.k).astype(int)),
            'similar': self.sparse_repr.get_item_of_idx(nearest_neighbors.flatten().astype(int)),
            'score': nearest_sims.flatten()
        })
        sframe = tc.SFrame(df[[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID]])
        self.model = tc.recommender.item_similarity_recommender.create(
            observation_data=sframe,
            user_id=rr.COLUMN_USER_ID,
            item_id=rr.COLUMN_ITEM_ID,
            similarity_type='cosine',
            only_top_k=self.k,
            nearest_items=sim_table,
            target_memory_usage=rr.MEM_SIZE_LIMIT
        )

    def predict(self, df, top_n=10):
        recommendations = self.model.recommend(
            users=df[rr.COLUMN_USER_ID].unique(),
            k=top_n,
            exclude_known=True
        ).to_dataframe().drop(columns=['score'])
        return recommendations


class UserItemSimilarity(object):
    def __init__(self, embeddings_dir, embeddings_filename, similarity_metric='cosine'):        
        self.sparse_repr = pickle.load(open(os.path.join(embeddings_dir, rr.FILE_SPARSE_REPR.format(embeddings_filename)), 'rb'))
        self.item_embeddings = pickle.load(open(os.path.join(embeddings_dir, rr.FILE_ITEM_EMBEDDINGS.format(embeddings_filename)), 'rb'))
        self.user_embeddings = pickle.load(open(os.path.join(embeddings_dir, rr.FILE_USER_EMBEDDINGS.format(embeddings_filename)), 'rb'))
        self.similarity_metric = similarity_metric

    def fit(self, df):
        self.df_train = df.copy()
        known_items = set(self.sparse_repr._item_encoder.classes_)
        self.df_train = self.df_train[self.df_train[rr.COLUMN_ITEM_ID].isin(known_items)]

    def predict(self, df, top_n=10):
        target_users = sorted(df[rr.COLUMN_USER_ID].unique())
        top_n_items = np.empty((len(target_users), top_n), dtype=np.int32)
        users_per_batch = int(rr.MEM_SIZE_LIMIT / (8 * self.sparse_repr.get_n_items()))
        for u in range(0, len(target_users), users_per_batch):
            batch_users = target_users[u:u+users_per_batch]
            batch_encoder = LabelEncoder()
            batch_encoder.fit(batch_users)
            batch_users = batch_encoder.inverse_transform(np.arange(len(batch_users)))
            users_idx = self.sparse_repr.get_idx_of_user(batch_users)
            if self.similarity_metric == 'cosine':
                batch_sims = cosine_similarity(self.user_embeddings[users_idx], self.item_embeddings)
            elif self.similarity_metric == 'dot':
                batch_sims = np.dot(self.user_embeddings[users_idx], self.item_embeddings.T)
            known_interactions = self.df_train[self.df_train[rr.COLUMN_USER_ID].isin(batch_users)]
            batch_sims[
                batch_encoder.transform(known_interactions[rr.COLUMN_USER_ID]), 
                self.sparse_repr.get_idx_of_item(known_interactions[rr.COLUMN_ITEM_ID].values)
            ] = -np.inf
            top_n_items[u:u+users_per_batch] = np.flip(np.argsort(batch_sims, axis=1), axis=1)[:, :top_n]
        recommendations = pd.DataFrame([], columns=[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID, rr.COLUMN_RANK])
        recommendations[rr.COLUMN_USER_ID] = np.repeat(target_users, top_n)
        recommendations[rr.COLUMN_ITEM_ID] = self.sparse_repr.get_item_of_idx(top_n_items.flatten())
        recommendations = recommendations.sort_values(rr.COLUMN_USER_ID, ascending=True)
        recommendations[rr.COLUMN_RANK] = np.concatenate(recommendations.groupby(rr.COLUMN_USER_ID).size().sort_index(ascending=True).apply(lambda x:np.arange(1, x+1)).values)
        return recommendations


class UserItemConcatenationSimilarity(object):
    def __init__(self, embeddings_dir, embeddings_filename):        
        self.sparse_repr = pickle.load(open(os.path.join(embeddings_dir, rr.FILE_SPARSE_REPR.format(embeddings_filename)), 'rb'))
        self.item_embeddings = pickle.load(open(os.path.join(embeddings_dir, rr.FILE_ITEM_EMBEDDINGS.format(embeddings_filename)), 'rb'))
        self.user_embeddings = pickle.load(open(os.path.join(embeddings_dir, rr.FILE_USER_EMBEDDINGS.format(embeddings_filename)), 'rb'))

    def fit(self, df):
        self.df_train = df.copy()
        known_items = set(self.sparse_repr._item_encoder.classes_)
        self.df_train = self.df_train[self.df_train[rr.COLUMN_ITEM_ID].isin(known_items)]

    def predict(self, df, top_n=10):
        target_users = sorted(df[rr.COLUMN_USER_ID].unique())
        recommendations = pd.DataFrame()
        for tu, target_user in enumerate(target_users, start=1):
            print('UICS - {}/{}'.format(tu, len(target_users)), end='\r', flush=True)
            target_user_idx = self.sparse_repr.get_idx_of_user(target_user)
            user_items = self.df_train[self.df_train[rr.COLUMN_USER_ID]==target_user][rr.COLUMN_ITEM_ID].values
            user_interactions_embeddings = np.hstack([
                np.tile(self.user_embeddings[target_user_idx], (len(user_items), 1)),
                self.item_embeddings[self.sparse_repr.get_idx_of_item(user_items)]
            ])
            neighbor_interactions = self.df_train[
                (self.df_train[rr.COLUMN_USER_ID]!=target_user)
                &(~self.df_train[rr.COLUMN_ITEM_ID].isin(user_items))
            ].copy()
            neighbor_interactions_embeddings = np.hstack([
                self.user_embeddings[self.sparse_repr.get_idx_of_user(neighbor_interactions[rr.COLUMN_USER_ID].values)],
                self.item_embeddings[self.sparse_repr.get_idx_of_item(neighbor_interactions[rr.COLUMN_ITEM_ID].values)]
            ])
            neighbor_interactions['sim'] = cosine_similarity(
                user_interactions_embeddings, neighbor_interactions_embeddings
            ).mean(axis=0)    
            recommendations = pd.concat([
                recommendations,
                pd.DataFrame([
                    np.repeat(target_user, top_n),
                    neighbor_interactions.groupby(rr.COLUMN_ITEM_ID)['sim'].mean().nlargest(top_n).index.values,
                    np.arange(1, top_n+1, dtype=int)
                ], index=[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID, rr.COLUMN_RANK]).transpose().astype(int)
            ])
        return recommendations


class ImplicitKNNUserConcatenation(object):
    def __init__(self, embeddings_dir, embeddings_filename, k=64, user_mean=True, user_top_n=None):
        self.k = k
        self.user_mean = user_mean
        self.user_top_n = user_top_n        
        self.sparse_repr = pickle.load(open(os.path.join(embeddings_dir, rr.FILE_SPARSE_REPR.format(embeddings_filename)), 'rb'))
        self.item_embeddings = pickle.load(open(os.path.join(embeddings_dir, rr.FILE_ITEM_EMBEDDINGS.format(embeddings_filename)), 'rb'))
        self.user_embeddings = pickle.load(open(os.path.join(embeddings_dir, rr.FILE_USER_EMBEDDINGS.format(embeddings_filename)), 'rb'))

    def fit(self, df):
        # Recupera os ids
        users_idx = self.sparse_repr.get_idx_of_user(df[rr.COLUMN_USER_ID].values)
        items_idx = self.sparse_repr.get_idx_of_item(df[rr.COLUMN_ITEM_ID].values)        
        
        # Calcula similaridade entre usuarios e items
        df['sims'] = np.sum(self.user_embeddings[users_idx]*self.item_embeddings[items_idx], axis=1)/(np.sqrt(np.sum(np.power(self.user_embeddings[users_idx], 2), axis=1))*np.sqrt(np.sum(np.power(self.item_embeddings[items_idx],2), axis=1)))
        df['sims'] = df['sims'].fillna(0)
        df = df.sort_values('sims', ascending=False)

        # Captura top n usuarios mais proximos por item
        if self.user_top_n is not None:
            df = df.groupby(rr.COLUMN_ITEM_ID).head(self.user_top_n)
            users_idx = self.sparse_repr.get_idx_of_user(df[rr.COLUMN_USER_ID].values)
            items_idx = self.sparse_repr.get_idx_of_item(df[rr.COLUMN_ITEM_ID].values)

        # Gera concatenacao
        embedding_dim = self.item_embeddings.shape[1]
        concat_emb_size = embedding_dim if (self.user_mean or self.user_top_n is None) else (df.groupby(rr.COLUMN_ITEM_ID).size().max() * embedding_dim)
        self.item_embeddings = np.hstack([self.item_embeddings, np.zeros((len(self.item_embeddings), concat_emb_size))])
        if self.user_mean or self.user_top_n is None:
            user_emb_reduced = pd.DataFrame(self.user_embeddings[users_idx]).groupby(items_idx).mean()
            self.item_embeddings[user_emb_reduced.index, embedding_dim:] = user_emb_reduced.values
        else:
            df['user_embeddings'] = self.user_embeddings[users_idx].tolist()
            user_emb_reduced = df.groupby(rr.COLUMN_ITEM_ID)['user_embeddings'].apply(lambda x: np.pad(np.concatenate(x.values), pad_width=(0, concat_emb_size-embedding_dim*len(x)), mode='wrap'))
            self.item_embeddings[self.sparse_repr.get_idx_of_item(user_emb_reduced.index.values), embedding_dim:] = np.stack(user_emb_reduced.values)

        # Calcula as similaridades
        n_items = self.item_embeddings.shape[0]
        items_per_batch = int(rr.MEM_SIZE_LIMIT / (8 * n_items))
        nearest_neighbors = np.empty((n_items, self.k))
        nearest_sims = np.empty((n_items, self.k))
        for i in range(0, n_items, items_per_batch):
            batch_sims = cosine_similarity(self.item_embeddings[i:i+items_per_batch], self.item_embeddings)
            np.fill_diagonal(batch_sims[:, i:i+items_per_batch], -np.inf)
            nearest_neighbors[i:i+items_per_batch] = np.flip(np.argsort(batch_sims, axis=1), axis=1)[:, :self.k]
            nearest_sims[i:i+items_per_batch] = np.flip(np.sort(batch_sims, axis=1), axis=1)[:, :self.k]
        sim_table = tc.SFrame({
            'id_item': self.sparse_repr.get_item_of_idx(np.repeat(np.arange(n_items), self.k).astype(int)),
            'similar': self.sparse_repr.get_item_of_idx(nearest_neighbors.flatten().astype(int)),
            'score': nearest_sims.flatten()
        })
        sframe = tc.SFrame(df[[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID]])
        self.model = tc.recommender.item_similarity_recommender.create(
            observation_data=sframe,
            user_id=rr.COLUMN_USER_ID,
            item_id=rr.COLUMN_ITEM_ID,
            similarity_type='cosine',
            only_top_k=self.k,
            nearest_items=sim_table,
            target_memory_usage=rr.MEM_SIZE_LIMIT
        )

    def predict(self, df, top_n=10):
        recommendations = self.model.recommend(
            users=df[rr.COLUMN_USER_ID].unique(),
            k=top_n,
            exclude_known=True
        ).to_dataframe().drop(columns=['score'])
        return recommendations


class UserItemWeightedSimilarity(object):
    def __init__(self, embeddings_dir, embeddings_filename, k=64, user_item_weights=None, similarity_metric='cosine'):
        self.k = k
        self.user_item_weights = user_item_weights
        self.sparse_repr = pickle.load(open(os.path.join(embeddings_dir, rr.FILE_SPARSE_REPR.format(embeddings_filename)), 'rb'))
        self.item_embeddings = pickle.load(open(os.path.join(embeddings_dir, rr.FILE_ITEM_EMBEDDINGS.format(embeddings_filename)), 'rb'))
        self.user_embeddings = pickle.load(open(os.path.join(embeddings_dir, rr.FILE_USER_EMBEDDINGS.format(embeddings_filename)), 'rb'))
        self.similarity_metric = similarity_metric

    def fit(self, df):
        # Similaridade entre itens
        n_items = self.item_embeddings.shape[0]
        items_per_batch = int(rr.MEM_SIZE_LIMIT / (8 * n_items))
        self.item_item_sim = pd.DataFrame()
        for i in range(0, n_items, items_per_batch):
            batch_items = self.sparse_repr.get_item_of_idx(np.arange(i, min(i+items_per_batch, n_items)))
            if self.similarity_metric == 'cosine':
                batch_sims = cosine_similarity(self.item_embeddings[i:i+items_per_batch], self.item_embeddings)
            elif self.similarity_metric == 'dot':
                batch_sims = np.dot(self.item_embeddings[i:i+items_per_batch], self.item_embeddings.T)
            np.fill_diagonal(batch_sims[:, i:i+items_per_batch], -np.inf)
            self.item_item_sim = pd.concat([
                self.item_item_sim,    
                pd.DataFrame(
                    np.column_stack([
                        np.repeat(batch_items, self.k),
                        self.sparse_repr.get_item_of_idx(np.flip(np.argsort(batch_sims, axis=1), axis=1)[:, :self.k].flatten()),
                        np.flip(np.sort(batch_sims, axis=1), axis=1)[:, :self.k].flatten(),
                    ]),
                    columns=[rr.COLUMN_ITEM_ID, 'neighbor', 'sim']            
                )
            ])
        self.df_train = df.copy()

    def predict(self, df, top_n=10):
        target_users = sorted(df[rr.COLUMN_USER_ID].unique())
        user_item_sim = pd.DataFrame()
        users_per_batch = int(rr.MEM_SIZE_LIMIT / (8 * self.sparse_repr.get_n_items()))
        for u in range(0, len(target_users), users_per_batch):
            batch_users = target_users[u:u+users_per_batch]
            batch_encoder = LabelEncoder()
            batch_encoder.fit(batch_users)
            batch_users = batch_encoder.inverse_transform(np.arange(len(batch_users)))
            users_idx = self.sparse_repr.get_idx_of_user(batch_users)
            if self.similarity_metric == 'cosine':
                batch_sims = cosine_similarity(self.user_embeddings[users_idx], self.item_embeddings)
            elif self.similarity_metric == 'dot':
                batch_sims = np.dot(self.user_embeddings[users_idx], self.item_embeddings.T)            
            known_interactions = self.df_train[self.df_train[rr.COLUMN_USER_ID].isin(batch_users)]
            batch_sims[
                batch_encoder.transform(known_interactions[rr.COLUMN_USER_ID]), 
                self.sparse_repr.get_idx_of_item(known_interactions[rr.COLUMN_ITEM_ID].values)
            ] = -np.inf
            user_item_sim = pd.concat([
                user_item_sim,    
                pd.DataFrame(
                    np.column_stack([
                        np.repeat(batch_users, self.k),
                        self.sparse_repr.get_item_of_idx(np.flip(np.argsort(batch_sims, axis=1), axis=1)[:, :self.k].flatten()),
                        np.flip(np.sort(batch_sims, axis=1), axis=1)[:, :self.k].flatten(),
                    ]),
                    columns=[rr.COLUMN_USER_ID, 'neighbor', 'sim']
                )
            ])
        user_item_sim = user_item_sim.set_index([rr.COLUMN_USER_ID, 'neighbor'])['sim']

        item_based_neighborhood = pd.merge(self.df_train[self.df_train[rr.COLUMN_USER_ID].isin(target_users)], self.item_item_sim, on=rr.COLUMN_ITEM_ID, how='inner')        
        item_based_neighborhood_qt = item_based_neighborhood.groupby([rr.COLUMN_USER_ID, 'neighbor']).size()
        if self.user_item_weights is None:
            item_based_neighborhood_sim = item_based_neighborhood.groupby([rr.COLUMN_USER_ID, 'neighbor'])['sim'].sum()
            final_sim = item_based_neighborhood_sim.add(user_item_sim, fill_value=0).divide(item_based_neighborhood_qt.add(pd.Series(1, index=user_item_sim.index), fill_value=0)).to_frame('sim').reset_index()
        else:
            user_weight, item_weight = self.user_item_weights
            item_based_neighborhood_sim = item_based_neighborhood.groupby([rr.COLUMN_USER_ID, 'neighbor'])['sim'].mean().multiply(item_weight)
            final_sim = item_based_neighborhood_sim.add(user_item_sim.multiply(user_weight), fill_value=0).divide(item_weight+user_weight).to_frame('sim').reset_index()        
        del item_based_neighborhood
        del item_based_neighborhood_sim
        del item_based_neighborhood_qt

        final_sim = final_sim.merge(
            self.df_train, 
            how='left', 
            left_on=[rr.COLUMN_USER_ID, 'neighbor'], 
            right_on=[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID]
        )
        final_sim = final_sim[final_sim[rr.COLUMN_ITEM_ID].isna()].drop(columns=[rr.COLUMN_ITEM_ID])
        recommendations = final_sim.sort_values('sim', ascending=False).groupby(rr.COLUMN_USER_ID).head(top_n).sort_values(['id_user', 'sim'], ascending=[True, False])
        del final_sim
        
        recommendations[rr.COLUMN_RANK] = np.concatenate(recommendations.groupby(rr.COLUMN_USER_ID).size().sort_index(ascending=True).apply(lambda x:np.arange(1, x+1)).values)
        recommendations = recommendations.rename(columns={'neighbor': rr.COLUMN_ITEM_ID})[[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID, rr.COLUMN_RANK]].reset_index(drop=True)
        return recommendations


class Comitee(object):
    def __init__(self, embeddings_dir, embeddings_filename, k=64, use_rank=True, use_ndcg=False, num_votes=10):
        self.embeddings_dir = embeddings_dir
        self.embeddings_filename = embeddings_filename
        self.k = k
        self.use_rank = use_rank
        self.use_ndcg = use_ndcg
        self.num_votes = num_votes

    def fit(self, df):
        self.comitee = list()        
        model = ImplicitKNN(self.embeddings_dir, self.embeddings_filename, self.k)
        self.comitee.append(model)
        model = UserItemSimilarity(self.embeddings_dir, self.embeddings_filename)
        self.comitee.append(model)
        for user_mean in [False, True]:
            for user_top_n in [1, 3, 5, 10, 15, None]:
                model = ImplicitKNNUserConcatenation(self.embeddings_dir, self.embeddings_filename, self.k, user_mean=user_mean, user_top_n=user_top_n)            
                self.comitee.append(model)
        for user_item_weight in [(0.9, 0.1), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.1, 0.9)]:
            model = UserItemWeightedSimilarity(self.embeddings_dir, self.embeddings_filename, self.k, user_item_weight)            
            self.comitee.append(model)

        self.scores = list()
        if self.use_ndcg:            
            df_train, df_val = recsys_train_test_split(df, train_size=0.9, val_size=0.1)
            df_train = cut_by_minimal_interactions(df_train, min_interactions=2)
            for i in range(len(self.comitee)):
                self.comitee[i].fit(df_train)                
                pred = self.comitee[i].predict(df_val, top_n=self.num_votes)
                self.scores.append(ndcg_score(df_val, pred, top_n=[self.num_votes]))                
        else:
            self.scores = np.ones(len(self.comitee))
            for i in range(len(self.comitee)):
                self.comitee[i].fit(df)
    
    def predict(self, df, top_n=10):
        votes = pd.DataFrame()
        for model, score in zip(self.comitee, self.scores):            
            pred = model.predict(df, top_n)
            pred['score'] = score
            votes = pd.concat([votes, pred])
        if self.use_rank:
            votes[rr.COLUMN_RANK] = ((top_n + 1) - votes[rr.COLUMN_RANK]) * votes['score']
            counting = votes.groupby([rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID])[rr.COLUMN_RANK].sum().divide(np.sum(self.scores))
        else:
            counting = votes.groupby([rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID])['score'].sum()
        recommendations = counting.sort_values(ascending=False).groupby(rr.COLUMN_USER_ID).head(top_n).to_frame('score').reset_index().drop(columns='score').sort_values(rr.COLUMN_USER_ID, ascending=True)
        recommendations[rr.COLUMN_RANK] = np.concatenate(recommendations.groupby(rr.COLUMN_USER_ID).size().sort_index(ascending=True).apply(lambda x:np.arange(1, x+1)).values)
        return recommendations


class NeuralModelOutput(object):
    def __init__(self, embeddings_dir, embeddings_filename):        
        self.sparse_repr = pickle.load(open(os.path.join(embeddings_dir, rr.FILE_SPARSE_REPR.format(embeddings_filename)), 'rb'))
        self.item_embeddings = pickle.load(open(os.path.join(embeddings_dir, rr.FILE_ITEM_EMBEDDINGS.format(embeddings_filename)), 'rb'))
        self.user_embeddings = pickle.load(open(os.path.join(embeddings_dir, rr.FILE_USER_EMBEDDINGS.format(embeddings_filename)), 'rb'))

    def fit(self, df):
        self.df_train = df.copy()
        known_items = set(self.sparse_repr._item_encoder.classes_)
        self.df_train = self.df_train[self.df_train[rr.COLUMN_ITEM_ID].isin(known_items)]

    def predict(self, df, top_n=10):
        target_users = sorted(df[rr.COLUMN_USER_ID].unique())
        top_n_items = np.empty((len(target_users), top_n), dtype=np.int32)
        users_per_batch = int(rr.MEM_SIZE_LIMIT / (8 * self.sparse_repr.get_n_items()))
        for u in range(0, len(target_users), users_per_batch):
            batch_users = target_users[u:u+users_per_batch]
            batch_encoder = LabelEncoder()
            batch_encoder.fit(batch_users)
            batch_users = batch_encoder.inverse_transform(np.arange(len(batch_users)))
            users_idx = self.sparse_repr.get_idx_of_user(batch_users)
            batch_sims = 1/(1+np.exp(-np.dot(self.user_embeddings[users_idx], self.item_embeddings.T))) # Calculo igual eh feito na rede
            known_interactions = self.df_train[self.df_train[rr.COLUMN_USER_ID].isin(batch_users)]
            batch_sims[
                batch_encoder.transform(known_interactions[rr.COLUMN_USER_ID]), 
                self.sparse_repr.get_idx_of_item(known_interactions[rr.COLUMN_ITEM_ID].values)
            ] = -np.inf
            top_n_items[u:u+users_per_batch] = np.flip(np.argsort(batch_sims, axis=1), axis=1)[:, :top_n]
        recommendations = pd.DataFrame([], columns=[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID, rr.COLUMN_RANK])
        recommendations[rr.COLUMN_USER_ID] = np.repeat(target_users, top_n)
        recommendations[rr.COLUMN_ITEM_ID] = self.sparse_repr.get_item_of_idx(top_n_items.flatten())
        recommendations = recommendations.sort_values(rr.COLUMN_USER_ID, ascending=True)
        recommendations[rr.COLUMN_RANK] = np.concatenate(recommendations.groupby(rr.COLUMN_USER_ID).size().sort_index(ascending=True).apply(lambda x:np.arange(1, x+1)).values)
        return recommendations