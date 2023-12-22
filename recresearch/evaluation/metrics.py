import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import recresearch as rr

def mae_score(real, pred):
    return real.subtract(pred, fill_value=(rr.RATING_SCALE_HI+rr.RATING_SCALE_LO)/2).abs().mean()

def rmse_score(real, pred):
    return np.sqrt(real.subtract(pred, fill_value=(rr.RATING_SCALE_HI+rr.RATING_SCALE_LO)/2).pow(2).mean())

def precision_recall_score(real, pred, top_n=[10]):
    precision, recall = dict(), dict()
    for k in top_n:
        top_n_pred = pred[pred[rr.COLUMN_RANK]<=k]
        correct = pd.merge(real, top_n_pred, how='inner', on=[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID])
        precision[k] = correct.groupby(rr.COLUMN_USER_ID).size().divide(pd.Series(k, index=real[rr.COLUMN_USER_ID].unique()), fill_value=0).mean()
        recall[k] = correct.groupby(rr.COLUMN_USER_ID).size().divide(real.groupby(rr.COLUMN_USER_ID).size(), fill_value=0).mean()
    return precision if len(precision) > 1 else list(precision.values())[0], recall if len(recall) > 1 else list(recall.values())[0]

def ndcg_score(real, pred, top_n=[10]):
    ndcg = dict()
    for k in top_n:
        top_n_pred = pred[pred[rr.COLUMN_RANK]<=k]
        correct = pd.merge(real, top_n_pred, how='inner', on=[rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID])
        correct['dcg'] = 1/np.log2(correct[rr.COLUMN_RANK]+1)
        dcg_per_n_items = {n_items: (1/np.log2(np.arange(1, k+1)+1))[:n_items].sum() for n_items in range(1, k+1)}
        ideal_dcg = real.groupby(rr.COLUMN_USER_ID).size().clip(upper=k).replace(dcg_per_n_items)
        ndcg[k] = correct.groupby(rr.COLUMN_USER_ID)['dcg'].sum().divide(ideal_dcg, fill_value=0).mean()
    return ndcg if len(ndcg) > 1 else list(ndcg.values())[0]

def content_based_ndcg_score(content_matrix, embeddings, top_n=[10]):
    logs = np.log2(np.arange(2, len(embeddings)+1))
    n_items = len(embeddings)
    MEM_SIZE_LIMIT = 100_000_000
    items_per_batch = int(MEM_SIZE_LIMIT / (8 * n_items))
    cbndcg = {k: list() for k in top_n}
    for i in range(0, n_items, items_per_batch):
        print('{}/{}'.format(i, n_items), end='\r', flush=True)        
        content_sims = cosine_similarity(content_matrix[i:i+items_per_batch], content_matrix)
        np.fill_diagonal(content_sims[:, i:i+items_per_batch], -np.inf)
        embeddings_sims = cosine_similarity(embeddings[i:i+items_per_batch], embeddings)
        np.fill_diagonal(embeddings_sims[:, i:i+items_per_batch], -np.inf)
        sorted_content_sims = np.flip(np.sort(content_sims, axis=1), axis=1)[:, :-1]
        sorted_embeddings_neighbors = np.flip(np.argsort(embeddings_sims, axis=1), axis=1)[:, :-1]
        sorted_embeddings_sims = content_sims[
            np.tile(np.arange(len(sorted_content_sims)).reshape(-1, 1), (1, len(embeddings)-1)),
            sorted_embeddings_neighbors
        ]
        for k in top_n:
            dcg = np.sum(sorted_embeddings_sims[:, :k]/logs[:k], axis=1)
            idcg = np.sum(sorted_content_sims[:, :k]/logs[:k], axis=1)
            cbndcg[k].extend(dcg[np.where(idcg!=0)]/idcg[np.where(idcg!=0)])
    for k in top_n:
        cbndcg[k] = np.mean(cbndcg[k])
    return cbndcg


def autotagging_score(content_matrix, embeddings, top_n=[10]):    
    n_items = len(embeddings)
    MEM_SIZE_LIMIT = 100_000_000
    items_per_batch = int(MEM_SIZE_LIMIT / (8 * n_items))
    
    precision_matrix = np.empty((n_items, len(top_n)))
    recall_matrix = np.empty((n_items, len(top_n)))

    for i in range(0, n_items, items_per_batch):

        embeddings_sims = cosine_similarity(embeddings[i:i+items_per_batch], embeddings)
        np.fill_diagonal(embeddings_sims[:, i:i+items_per_batch], -np.inf)
        embeddings_neighbors = np.flip(np.argsort(embeddings_sims, axis=1), axis=1)[:,:-1]
        
        for real_emb, neighbors in enumerate(embeddings_neighbors, start=i):
            print('{}/{}'.format(real_emb, n_items), end='\r', flush=True)
            for k_idx, k in enumerate(top_n):                    
                pred_features = content_matrix[real_emb][content_matrix[neighbors[:k]].sum(axis=0)>=k/2]
                precision_matrix[real_emb, k_idx] = (pred_features.sum()/pred_features.shape[1]) if pred_features.shape[1] != 0 else 0
                recall_matrix[real_emb, k_idx] = pred_features.sum()/content_matrix[real_emb].sum()
    
    return precision_matrix.mean(axis=0), recall_matrix.mean(axis=0)