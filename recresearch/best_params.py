EXPLICIT_FM_PARAMS = {
    'Book-Crossing': {'n_factors': 100, 'regularization': 0.01},
    'CiaoDVD': {'n_factors': 50, 'regularization': 1e-06},
    'Filmtrust': {'n_factors': 200, 'regularization': 0.01},
    'Jester': {'n_factors': 200, 'regularization': 0.01},
    'MovieLens': {'n_factors': 200, 'regularization': 0.01},
    'NetflixPrize': {'n_factors': 200, 'regularization': 1e-06},
}

EXPLICIT_INTERACT2VEC_KNN_PARAMS = {
    'embeddings': {
        'Book-Crossing': {'embedding_dim': 50, 'n_epochs': 25, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 1e-05},
        'CiaoDVD': {'embedding_dim': 50, 'n_epochs': 75, 'negative_exponent': 0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
        'Filmtrust': {'embedding_dim': 50, 'n_epochs': 25, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
        'Jester': {'embedding_dim': 50, 'n_epochs': 75, 'negative_exponent': 0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
        'MovieLens': {'embedding_dim': 50, 'n_epochs': 25, 'negative_exponent': 0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
        'NetflixPrize': {'embedding_dim': 50, 'n_epochs': 25, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 1e-05},
    },
    'recommenders': {
        'Book-Crossing': {'k': 60},
        'CiaoDVD': {'k': 20},
        'Filmtrust': {'k': 100},
        'Jester': {'k': 20},
        'MovieLens': {'k': 20},
        'NetflixPrize': {'k': 20},
    },
}

EXPLICIT_ITEM2VEC_KNN_PARAMS = {
    'embeddings': {
        'Book-Crossing': {'embedding_dim': 50, 'n_epochs': 25, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
        'CiaoDVD': {'embedding_dim': 50, 'n_epochs': 75, 'negative_exponent': 0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 1e-05},
        'Filmtrust': {'embedding_dim': 50, 'n_epochs': 25, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
        'Jester': {'embedding_dim': 50, 'n_epochs': 25, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 1e-05},
        'MovieLens': {'embedding_dim': 50, 'n_epochs': 25, 'negative_exponent': 0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 1e-05},
        'NetflixPrize': {'embedding_dim': 50, 'n_epochs': 25, 'negative_exponent': 0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 1e-05},
    },
    'recommenders': {
        'Book-Crossing': {'k': 100},
        'CiaoDVD': {'k': 20},
        'Filmtrust': {'k': 60},
        'Jester': {'k': 100},
        'MovieLens': {'k': 20},
        'NetflixPrize': {'k': 20},
    },
}

EXPLICIT_KNN_PARAMS = {
    'Book-Crossing': {'k': 100},
    'CiaoDVD': {'k': 20},
    'Filmtrust': {'k': 100},
    'Jester': {'k': 100},
    'MovieLens': {'k': 20},
    'NetflixPrize': {'k': 20},
}

EXPLICIT_SVD_PARAMS = {
    'Book-Crossing': {'n_factors': 50, 'regularization': 0.01},
    'CiaoDVD': {'n_factors': 50, 'regularization': 0.01},
    'Filmtrust': {'n_factors': 50, 'regularization': 0.01},
    'Jester': {'n_factors': 50, 'regularization': 0.01},
    'MovieLens': {'n_factors': 50, 'regularization': 0.01},
    'NetflixPrize': {'n_factors': 50, 'regularization': 0.01},
}

EXPLICIT_USER2VEC_KNN_PARAMS = {
    'embeddings': {
        'Book-Crossing': {'embedding_dim': 50, 'n_epochs': 75, 'negative_exponent': 0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
        'CiaoDVD': {'embedding_dim': 50, 'n_epochs': 25, 'negative_exponent': 0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 1e-05},
        'Filmtrust': {'embedding_dim': 50, 'n_epochs': 25, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
        'Jester': {'embedding_dim': 50, 'n_epochs': 25, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 1e-05},
        'MovieLens': {'embedding_dim': 50, 'n_epochs': 75, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
        'NetflixPrize': {'embedding_dim': 50, 'n_epochs': 75, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
    },
    'recommenders': {
        'Book-Crossing': {'k': 40},
        'CiaoDVD': {'k': 20},
        'Filmtrust': {'k': 60},
        'Jester': {'k': 100},
        'MovieLens': {'k': 20},
        'NetflixPrize': {'k': 20},
    },
}

IMPLICIT_ALS_PARAMS = {
    'BestBuy': {'n_factors': 200, 'regularization': 0.01},
    'Book-Crossing': {'n_factors': 200, 'regularization': 0.0001},
    'CiaoDVD': {'n_factors': 50, 'regularization': 1e-06},
    'DeliciousBookmarks': {'n_factors': 200, 'regularization': 0.01},
    'Filmtrust': {'n_factors': 50, 'regularization': 1e-06},
    'Jester': {'n_factors': 50, 'regularization': 0.0001},
    'Last.FM - Listened': {'n_factors': 50, 'regularization': 1e-06},
    'MovieLens': {'n_factors': 50, 'regularization': 1e-06},
    'RetailRocket-Transactions': {'n_factors': 200, 'regularization': 0.0001},
}

IMPLICIT_BPR_PARAMS = {
    'BestBuy': {'n_factors': 50, 'regularization': 1e-06},
    'Book-Crossing': {'n_factors': 50, 'regularization': 1e-06},
    'CiaoDVD': {'n_factors': 50, 'regularization': 1e-06},
    'DeliciousBookmarks': {'n_factors': 50, 'regularization': 0.01},
    'Filmtrust': {'n_factors': 50, 'regularization': 0.01},
    'Jester': {'n_factors': 50, 'regularization': 0.01},
    'Last.FM - Listened': {'n_factors': 50, 'regularization': 0.0001},
    'MovieLens': {'n_factors': 100, 'regularization': 0.01},
    'RetailRocket-Transactions': {'n_factors': 50, 'regularization': 1e-06},
}

IMPLICIT_INTERACT2VEC_UIS_PARAMS = {
    'embeddings': {
        'Anime Recommendations': {'embedding_dim': 50, 'n_epochs': 75, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 1e-05},
        'BestBuy': {'embedding_dim': 50, 'n_epochs': 75, 'negative_exponent': 0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 1e-05},
        'Book-Crossing': {'embedding_dim': 50, 'n_epochs': 75, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 1e-05},
        'CiaoDVD': {'embedding_dim': 50, 'n_epochs': 25, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 1e-05},
        'DeliciousBookmarks': {'embedding_dim': 50, 'n_epochs': 25, 'negative_exponent': 0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 1e-05},
        'Filmtrust': {'embedding_dim': 50, 'n_epochs': 75, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 1e-05},
        'Jester': {'embedding_dim': 50, 'n_epochs': 25, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 1e-05},
        'Last.FM - Listened': {'embedding_dim': 50, 'n_epochs': 75, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 1e-05},
        'MovieLens': {'embedding_dim': 50, 'n_epochs': 25, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
        'NetflixPrize': {'embedding_dim': 50, 'n_epochs': 25, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
        'RetailRocket-Transactions': {'embedding_dim': 50, 'n_epochs': 25, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 1e-05},
    },
    'recommenders': {
        'Anime Recommendations': dict(),
        'BestBuy': dict(),
        'Book-Crossing': dict(),
        'CiaoDVD': dict(),
        'DeliciousBookmarks': dict(),
        'Filmtrust': dict(),
        'Jester': dict(),
        'Last.FM - Listened': dict(),
        'MovieLens': dict(),
        'NetflixPrize': dict(),
        'RetailRocket-Transactions': dict(),
    },
}

IMPLICIT_ITEM2VEC_KNN_PARAMS = {
    'embeddings': {
        'BestBuy': {'embedding_dim': 50, 'n_epochs': 75, 'negative_exponent': 0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 1e-05},
        'Book-Crossing': {'embedding_dim': 50, 'n_epochs': 75, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
        'CiaoDVD': {'embedding_dim': 50, 'n_epochs': 75, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
        'DeliciousBookmarks': {'embedding_dim': 50, 'n_epochs': 75, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
        'Filmtrust': {'embedding_dim': 50, 'n_epochs': 75, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
        'Jester': {'embedding_dim': 50, 'n_epochs': 75, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
        'Last.FM - Listened': {'embedding_dim': 50, 'n_epochs': 75, 'negative_exponent': 0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
        'MovieLens': {'embedding_dim': 50, 'n_epochs': 75, 'negative_exponent': 0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
        'NetflixPrize': {'embedding_dim': 50, 'n_epochs': 75, 'negative_exponent': 0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
        'RetailRocket-Transactions': {'embedding_dim': 50, 'n_epochs': 75, 'negative_exponent': 0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
    },
    'recommenders': {
        'BestBuy': {'k': 20},
        'Book-Crossing': {'k': 80},
        'CiaoDVD': {'k': 20},
        'DeliciousBookmarks': {'k': 100},
        'Filmtrust': {'k': 60},
        'Jester': {'k': 80},
        'Last.FM - Listened': {'k': 20},
        'MovieLens': {'k': 100},
        'NetflixPrize': {'k': 100},
        'RetailRocket-Transactions': {'k': 20},
    },
}

IMPLICIT_KNN_PARAMS = {
    'BestBuy': {'k': 100},
    'Book-Crossing': {'k': 60},
    'CiaoDVD': {'k': 100},
    'DeliciousBookmarks': {'k': 100},
    'Filmtrust': {'k': 60},
    'Jester': {'k': 100},
    'Last.FM - Listened': {'k': 100},
    'MovieLens': {'k': 100},
    'NetflixPrize': {'k': 100},
    'RetailRocket-Transactions': {'k': 100},
}

IMPLICIT_LMF_PARAMS = {
    'BestBuy': {'n_factors': 100, 'regularization': 0.0001},
    'Book-Crossing': {'n_factors': 200, 'regularization': 0.01},
    'CiaoDVD': {'n_factors': 200, 'regularization': 0.01},
    'DeliciousBookmarks': {'n_factors': 200, 'regularization': 0.01},
    'Filmtrust': {'n_factors': 50, 'regularization': 0.01},
    'Jester': {'n_factors': 100, 'regularization': 0.01},
    'Last.FM - Listened': {'n_factors': 100, 'regularization': 1e-06},
    'MovieLens': {'n_factors': 50, 'regularization': 1e-06},
    'RetailRocket-Transactions': {'n_factors': 200, 'regularization': 0.01},
}

IMPLICIT_USER2VEC_UIS_PARAMS = {
    'embeddings': {
        'BestBuy': {'embedding_dim': 50, 'n_epochs': 25, 'negative_exponent': 0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
        'Book-Crossing': {'embedding_dim': 50, 'n_epochs': 75, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 1e-05},
        'CiaoDVD': {'embedding_dim': 50, 'n_epochs': 75, 'negative_exponent': 0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
        'DeliciousBookmarks': {'embedding_dim': 50, 'n_epochs': 75, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
        'Filmtrust': {'embedding_dim': 50, 'n_epochs': 25, 'negative_exponent': 0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
        'Jester': {'embedding_dim': 50, 'n_epochs': 25, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
        'Last.FM - Listened': {'embedding_dim': 50, 'n_epochs': 75, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
        'MovieLens': {'embedding_dim': 50, 'n_epochs': 25, 'negative_exponent': -0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 1e-05},
        'NetflixPrize': {'embedding_dim': 50, 'n_epochs': 25, 'negative_exponent': 0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 1e-05},
        'RetailRocket-Transactions': {'embedding_dim': 50, 'n_epochs': 75, 'negative_exponent': 0.5, 'negative_sampling': 5, 'regularization_lambda': 1e-06, 'subsampling_p': 0.0001},
    },
    'recommenders': {
        'BestBuy': dict(),
        'Book-Crossing': dict(),
        'CiaoDVD': dict(),
        'DeliciousBookmarks': dict(),
        'Filmtrust': dict(),
        'Jester': dict(),
        'Last.FM - Listened': dict(),
        'MovieLens': dict(),
        'NetflixPrize': dict(),
        'RetailRocket-Transactions': dict(),
    },
}






CONCAT_PARAMS_M0 = {
    'recommenders': {
        'BestBuy': {'k': 100, 'user_mean': True, 'user_top_n': None},
        'Book-Crossing': {'k': 60, 'user_mean': True, 'user_top_n': None},
        'CiaoDVD': {'k': 100, 'user_mean': True, 'user_top_n': None},
        'DeliciousBookmarks': {'k': 100, 'user_mean': True, 'user_top_n': None},
        'Filmtrust': {'k': 60, 'user_mean': True, 'user_top_n': None},
        'Jester': {'k': 100, 'user_mean': True, 'user_top_n': None},
        'Last.FM - Listened': {'k': 100, 'user_mean': True, 'user_top_n': None},
        'MovieLens': {'k': 100, 'user_mean': True, 'user_top_n': None},
        'NetflixPrize': {'k': 100, 'user_mean': True, 'user_top_n': None},
        'RetailRocket-Transactions': {'k': 100, 'user_mean': True, 'user_top_n': None},
    },
}


CONCAT_PARAMS_M3 = {
    'recommenders': {
        'BestBuy': {'k': 100, 'user_mean': True, 'user_top_n': 3},
        'Book-Crossing': {'k': 60, 'user_mean': True, 'user_top_n': 3},
        'CiaoDVD': {'k': 100, 'user_mean': True, 'user_top_n': 3},
        'DeliciousBookmarks': {'k': 100, 'user_mean': True, 'user_top_n': 3},
        'Filmtrust': {'k': 60, 'user_mean': True, 'user_top_n': 3},
        'Jester': {'k': 100, 'user_mean': True, 'user_top_n': 3},
        'Last.FM - Listened': {'k': 100, 'user_mean': True, 'user_top_n': 3},
        'MovieLens': {'k': 100, 'user_mean': True, 'user_top_n': 3},
        'NetflixPrize': {'k': 100, 'user_mean': True, 'user_top_n': 3},
        'RetailRocket-Transactions': {'k': 100, 'user_mean': True, 'user_top_n': 3},        
    },
}

CONCAT_PARAMS_M5 = {
    'recommenders': {
        'BestBuy': {'k': 100, 'user_mean': True, 'user_top_n': 5},
        'Book-Crossing': {'k': 60, 'user_mean': True, 'user_top_n': 5},
        'CiaoDVD': {'k': 100, 'user_mean': True, 'user_top_n': 5},
        'DeliciousBookmarks': {'k': 100, 'user_mean': True, 'user_top_n': 5},
        'Filmtrust': {'k': 60, 'user_mean': True, 'user_top_n': 5},
        'Jester': {'k': 100, 'user_mean': True, 'user_top_n': 5},
        'Last.FM - Listened': {'k': 100, 'user_mean': True, 'user_top_n': 5},
        'MovieLens': {'k': 100, 'user_mean': True, 'user_top_n': 5},
        'NetflixPrize': {'k': 100, 'user_mean': True, 'user_top_n': 5},
        'RetailRocket-Transactions': {'k': 100, 'user_mean': True, 'user_top_n': 5},        
    },
}

CONCAT_PARAMS_C3 = {
    'recommenders': {
        'BestBuy': {'k': 100, 'user_mean': False, 'user_top_n': 3},
        'Book-Crossing': {'k': 60, 'user_mean': False, 'user_top_n': 3},
        'CiaoDVD': {'k': 100, 'user_mean': False, 'user_top_n': 3},
        'DeliciousBookmarks': {'k': 100, 'user_mean': False, 'user_top_n': 3},
        'Filmtrust': {'k': 60, 'user_mean': False, 'user_top_n': 3},
        'Jester': {'k': 100, 'user_mean': False, 'user_top_n': 3},
        'Last.FM - Listened': {'k': 100, 'user_mean': False, 'user_top_n': 3},
        'MovieLens': {'k': 100, 'user_mean': False, 'user_top_n': 3},
        'NetflixPrize': {'k': 100, 'user_mean': False, 'user_top_n': 3},
        'RetailRocket-Transactions': {'k': 100, 'user_mean': False, 'user_top_n': 3},
    },
}

CONCAT_PARAMS_C5 = {
    'recommenders': {
        'BestBuy': {'k': 100, 'user_mean': False, 'user_top_n': 5},
        'Book-Crossing': {'k': 60, 'user_mean': False, 'user_top_n': 5},
        'CiaoDVD': {'k': 100, 'user_mean': False, 'user_top_n': 5},
        'DeliciousBookmarks': {'k': 100, 'user_mean': False, 'user_top_n': 5},
        'Filmtrust': {'k': 60, 'user_mean': False, 'user_top_n': 5},
        'Jester': {'k': 100, 'user_mean': False, 'user_top_n': 5},
        'Last.FM - Listened': {'k': 100, 'user_mean': False, 'user_top_n': 5},
        'MovieLens': {'k': 100, 'user_mean': False, 'user_top_n': 5},
        'NetflixPrize': {'k': 100, 'user_mean': False, 'user_top_n': 5},
        'RetailRocket-Transactions': {'k': 100, 'user_mean': False, 'user_top_n': 5},
    },
}

IMPLICIT_INTERACT2VEC_JOINT_PARAMS = {
    'recommenders': {
        'BestBuy': {'k': 20, 'user_weight': None, 'item_weight': None},
        'Book-Crossing': {'k': 80, 'user_weight': None, 'item_weight': None},
        'CiaoDVD': {'k': 20, 'user_weight': None, 'item_weight': None},
        'DeliciousBookmarks': {'k': 100, 'user_weight': None, 'item_weight': None},
        'Filmtrust': {'k': 60, 'user_weight': None, 'item_weight': None},
        'Jester': {'k': 80, 'user_weight': None, 'item_weight': None},
        'Last.FM - Listened': {'k': 20, 'user_weight': None, 'item_weight': None},
        'MovieLens': {'k': 100, 'user_weight': None, 'item_weight': None},
        'NetflixPrize': {'k': 100, 'user_weight': None, 'item_weight': None},
        'RetailRocket-Transactions': {'k': 20, 'user_weight': None, 'item_weight': None},
    },
}



COMITEE_COUNT_SIMPLE_PARAMS = {        
        'BestBuy': {'k': 20, 'use_rank': False, 'use_ndcg': False},
        'Book-Crossing': {'k': 80, 'use_rank': False, 'use_ndcg': False},
        'CiaoDVD': {'k': 20, 'use_rank': False, 'use_ndcg': False},
        'DeliciousBookmarks': {'k': 100, 'use_rank': False, 'use_ndcg': False},
        'Filmtrust': {'k': 60, 'use_rank': False, 'use_ndcg': False},
        'Jester': {'k': 80, 'use_rank': False, 'use_ndcg': False},
        'Last.FM - Listened': {'k': 20, 'use_rank': False, 'use_ndcg': False},
        'MovieLens': {'k': 100, 'use_rank': False, 'use_ndcg': False},
        'NetflixPrize': {'k': 100, 'use_rank': False, 'use_ndcg': False},
        'RetailRocket-Transactions': {'k': 20, 'use_rank': False, 'use_ndcg': False},    
}
COMITEE_COUNT_NDCG_PARAMS = {        
    'BestBuy': {'k': 20, 'use_rank': False, 'use_ndcg': True},
    'Book-Crossing': {'k': 80, 'use_rank': False, 'use_ndcg': True},
    'CiaoDVD': {'k': 20, 'use_rank': False, 'use_ndcg': True},
    'DeliciousBookmarks': {'k': 100, 'use_rank': False, 'use_ndcg': True},
    'Filmtrust': {'k': 60, 'use_rank': False, 'use_ndcg': True},
    'Jester': {'k': 80, 'use_rank': False, 'use_ndcg': True},
    'Last.FM - Listened': {'k': 20, 'use_rank': False, 'use_ndcg': True},
    'MovieLens': {'k': 100, 'use_rank': False, 'use_ndcg': True},
    'NetflixPrize': {'k': 100, 'use_rank': False, 'use_ndcg': True},
    'RetailRocket-Transactions': {'k': 20, 'use_rank': False, 'use_ndcg': True},    
}
COMITEE_RANK_SIMPLE_PARAMS = {        
        'BestBuy': {'k': 20, 'use_rank': True, 'use_ndcg': False},
        'Book-Crossing': {'k': 80, 'use_rank': True, 'use_ndcg': False},
        'CiaoDVD': {'k': 20, 'use_rank': True, 'use_ndcg': False},
        'DeliciousBookmarks': {'k': 100, 'use_rank': True, 'use_ndcg': False},
        'Filmtrust': {'k': 60, 'use_rank': True, 'use_ndcg': False},
        'Jester': {'k': 80, 'use_rank': True, 'use_ndcg': False},
        'Last.FM - Listened': {'k': 20, 'use_rank': True, 'use_ndcg': False},
        'MovieLens': {'k': 100, 'use_rank': True, 'use_ndcg': False},
        'NetflixPrize': {'k': 100, 'use_rank': True, 'use_ndcg': False},
        'RetailRocket-Transactions': {'k': 20, 'use_rank': True, 'use_ndcg': False},    
}
COMITEE_RANK_NDCG_PARAMS = {        
        'BestBuy': {'k': 20, 'use_rank': True, 'use_ndcg': True},
        'Book-Crossing': {'k': 80, 'use_rank': True, 'use_ndcg': True},
        'CiaoDVD': {'k': 20, 'use_rank': True, 'use_ndcg': True},
        'DeliciousBookmarks': {'k': 100, 'use_rank': True, 'use_ndcg': True},
        'Filmtrust': {'k': 60, 'use_rank': True, 'use_ndcg': True},
        'Jester': {'k': 80, 'use_rank': True, 'use_ndcg': True},
        'Last.FM - Listened': {'k': 20, 'use_rank': True, 'use_ndcg': True},
        'MovieLens': {'k': 100, 'use_rank': True, 'use_ndcg': True},
        'NetflixPrize': {'k': 100, 'use_rank': True, 'use_ndcg': True},
        'RetailRocket-Transactions': {'k': 20, 'use_rank': True, 'use_ndcg': True},    
}