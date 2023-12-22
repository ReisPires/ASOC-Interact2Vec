import recresearch as rr
from recresearch.methods.embeddings.static.recommenders import ImplicitKNN, UserItemSimilarity

KNN_PARAMS = {
    'k': [20, 40, 60, 80, 100]
}

IS_PARAMS = {
    'k': [64]
}

MF_PARAMS = {
    'n_factors': [50, 100, 200],
    'regularization': [1e-6, 1e-4, 1e-2],
    'n_epochs': [10, 20, 50, 100],
    'learning_rate': [0.0025, 0.025, 0.25]
}

BASIC_EMBEDDINGS_PARAMS = {
    'embedding_dim': [96],
    'n_epochs': [50, 100, 150],
    'negative_sampling': [5],
    'negative_exponent': [-1.0, -0.5, 0.5, 1.0],
    'subsampling_p': [1e-5, 1e-4, 1e-3],
    'regularization_lambda': [1e-6]
}

PONDERADO_EMBEDDINGS_PARAMS = {
    'embedding_dim': [96],
    'n_epochs': [50, 100, 150],
    'negative_sampling': [5],
    'negative_exponent': [-1.0, -0.5, 0.5, 1.0],
    'subsampling_p': [1e-5, 1e-4, 1e-3],
    'regularization_lambda': [1e-6],
    'negative_weight': [1,1.5,2]
}

SEQUENCIAL_EMBEDDINGS_PARAMS = {
    'embedding_dim': [96],
    'n_epochs': [50, 100, 150],
    'negative_sampling': [5],
    'negative_exponent': [-1.0, -0.5, 0.5, 1.0],
    'subsampling_p': [1e-5, 1e-4, 1e-3],
    'regularization_lambda': [1e-6],
    'window_size': [1,3,5,7,9],
    'cutoff_interval': [182, 365, 730]
}

INTERACT2VEC_REDUCED_EMBEDDINGS_PARAMS = {
    'embedding_dim': [96],
    'n_epochs': [50, 100, 150],
    'negative_sampling': [5],
    'negative_exponent': [-1.0, -0.5, 0.5, 1.0],
    'subsampling_p': [1e-5, 1e-4, 1e-3],
    'regularization_lambda': [1e-6]
}

INTERACT2VEC_FULL_EMBEDDINGS_PARAMS = {
    'embedding_dim': [100], #[25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300],
    'n_epochs': [5], #[5, 25, 50, 75, 100, 125, 150, 175, 200],
    'negative_sampling': [5], #[3, 5, 7, 10, 12, 15, 18, 20, 22, 25, 28, 30],
    'negative_exponent': [0.75], #[-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0],
    'subsampling_p': [1e-3], #[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, None],
    'regularization_lambda': [None], #[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, None],
    'learning_rate': [0.0025, 0.0075, 0.025, 0.075, 0.25, 0.75]
}

ITEM2VEC_IMPLICIT_PARAMS = {
    rr.EMB_PARAMS_EMBEDDINGS: BASIC_EMBEDDINGS_PARAMS,
    rr.EMB_PARAMS_RECOMMENDERS: {'Implicit KNN': IS_PARAMS}
}

ITEM2VEC_EXPLICIT_PARAMS = {
    rr.EMB_PARAMS_EMBEDDINGS: BASIC_EMBEDDINGS_PARAMS,
    rr.EMB_PARAMS_RECOMMENDERS: {'Explicit KNN': KNN_PARAMS}
}

ITEM2VEC_SEQUENCIAL_EXPLICIT_PARAMS = {
    rr.EMB_PARAMS_EMBEDDINGS: SEQUENCIAL_EMBEDDINGS_PARAMS,
    rr.EMB_PARAMS_RECOMMENDERS: {'Explicit KNN': KNN_PARAMS}
}

ITEM2VEC_SEQUENCIAL_IMPLICIT_PARAMS = {
    rr.EMB_PARAMS_EMBEDDINGS: SEQUENCIAL_EMBEDDINGS_PARAMS,
    rr.EMB_PARAMS_RECOMMENDERS: {'Implicit KNN': IS_PARAMS}
}

ITEM2VEC_PONDERADO_IMPLICIT_PARAMS = {
    rr.EMB_PARAMS_EMBEDDINGS: PONDERADO_EMBEDDINGS_PARAMS,
    rr.EMB_PARAMS_RECOMMENDERS: {'Implicit KNN': IS_PARAMS}
}

USER2VEC_IMPLICIT_PARAMS = {
    rr.EMB_PARAMS_EMBEDDINGS: BASIC_EMBEDDINGS_PARAMS,
    rr.EMB_PARAMS_RECOMMENDERS: {'UIS': dict()}
}

USER2VEC_EXPLICIT_PARAMS = {
    rr.EMB_PARAMS_EMBEDDINGS: BASIC_EMBEDDINGS_PARAMS,
    rr.EMB_PARAMS_RECOMMENDERS: {'Explicit KNN': KNN_PARAMS}
}

KNN_USER_CONCAT_PARAMS = {
    'k': [64], 
    'user_mean': [True, False],
    'user_top_n': [1, 5, 10, 15, None]
}

UIWS_PARAMS = {
    'k': [64], 
    'user_item_weights': [(0.90, 0.10), (0.75, 0.25), (0.50, 0.50), (0.25, 0.75), (0.10, 0.90)],
}

COMITEE_PARAMS = {
    'k': [64], 
    'use_rank': [True, False],
    'use_ndcg': [True, False], 
    'num_votes': [15, 30, 45]
}

INTERACT2VEC_IMPLICIT_PARAMS = {
    rr.EMB_PARAMS_EMBEDDINGS: INTERACT2VEC_REDUCED_EMBEDDINGS_PARAMS,
    rr.EMB_PARAMS_RECOMMENDERS: {
        'Implicit KNN': IS_PARAMS,
        'UIS': dict(),
        'Implicit KNN User Concat': KNN_USER_CONCAT_PARAMS,
        'UIWS': UIWS_PARAMS,
        'Comitee': COMITEE_PARAMS,
        'ModelOutput': dict() 
    }
}

INTERACT2VEC_EXPLICIT_PARAMS = {
    rr.EMB_PARAMS_EMBEDDINGS: INTERACT2VEC_REDUCED_EMBEDDINGS_PARAMS,
    rr.EMB_PARAMS_RECOMMENDERS: {'Explicit KNN': KNN_PARAMS}
}

RECENCY_PARAMS = {
    'threshold': [0.25, 0.5, 0.75, 20, 40, 60, 80, 100], 
    'alpha': [0.01, 0.1, 1, 2, 10]
}

MF_EMBEDDINGS_PARAMS = {
    rr.EMB_PARAMS_EMBEDDINGS: MF_PARAMS,
    rr.EMB_PARAMS_RECOMMENDERS: {
        'UIS': {'similarity_metric': ['cosine', 'dot']},
        'UIWS': {**UIWS_PARAMS, 'similarity_metric': ['cosine', 'dot']}
    }
}


WI2V_EMBEDDINGS_PARAMS = {
    'embedding_dim': [96],
    'n_epochs': [100],
    'negative_sampling': [5],
    'negative_exponent': [1.0],
    'subsampling_p': [1e-5],
    'regularization_lambda': [0.0001],
    'posw_method': ['i2i-mean'], 
    'negw_method': ['default'], 
    'posw': [0.5, 0.75, 1.0], 
    'negw': [0.5, 0.75, 1.0]
}

WI2V_IMPLICIT_PARAMS = {
    rr.EMB_PARAMS_EMBEDDINGS: WI2V_EMBEDDINGS_PARAMS,
    rr.EMB_PARAMS_RECOMMENDERS: {'Implicit KNN': IS_PARAMS}
}

WI2V_EXPLICIT_PARAMS = {
    rr.EMB_PARAMS_EMBEDDINGS: WI2V_EMBEDDINGS_PARAMS,
    rr.EMB_PARAMS_RECOMMENDERS: {'Explicit KNN': KNN_PARAMS}
}