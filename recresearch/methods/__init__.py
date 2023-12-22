import pandas as pd

import recresearch as rr

# Modelos
from recresearch.methods.explicit.static import KNNRaw, SVDSurprise, FMStaticTuricreate, RecencyStaticRaw
from recresearch.methods.explicit.temporal import RecencyTemporalRaw, FMTemporalTuricreate
from recresearch.methods.implicit.static import ISStaticTuricreate, ALSImplicit, BPRImplicit, LMFImplicit
from recresearch.methods.implicit.temporal import ISTemporalTuricreate
from recresearch.methods.embeddings.temporal import Item2VecSequencial, Item2VecPonderado, Item2VecPonderadoGensim
from recresearch.methods.embeddings.static import Item2VecKeras, Item2VecGensim, Interact2VecKeras, User2Vec, ALSEmbeddingsImplicit, BPREmbeddingsImplicit
from recresearch.methods.embeddings.static.recommenders import ImplicitKNN, UserItemSimilarity, ExplicitKNN
from recresearch.methods.embeddings.static.recommenders import ImplicitKNNUserConcatenation, UserItemWeightedSimilarity, Comitee, NeuralModelOutput

# Parametros
from recresearch.parameters.grid_search import KNN_PARAMS, MF_PARAMS, MF_EMBEDDINGS_PARAMS
from recresearch.parameters.grid_search import ITEM2VEC_IMPLICIT_PARAMS, ITEM2VEC_EXPLICIT_PARAMS, ITEM2VEC_SEQUENCIAL_IMPLICIT_PARAMS, ITEM2VEC_SEQUENCIAL_EXPLICIT_PARAMS, ITEM2VEC_PONDERADO_IMPLICIT_PARAMS
from recresearch.parameters.grid_search import USER2VEC_IMPLICIT_PARAMS, USER2VEC_EXPLICIT_PARAMS
from recresearch.parameters.grid_search import INTERACT2VEC_IMPLICIT_PARAMS, INTERACT2VEC_EXPLICIT_PARAMS
from recresearch.parameters.grid_search import RECENCY_PARAMS
from recresearch.parameters.grid_search import WI2V_IMPLICIT_PARAMS, WI2V_EXPLICIT_PARAMS

# ===================== Tabela de modelos =====================
MODELS_TABLE = pd.DataFrame(
    [[1,   'KNN Static',        'E',    rr.PIPELINE_RECOMMENDER,   KNNRaw,                  KNN_PARAMS],     
     [2,   'SVD',               'E',    rr.PIPELINE_RECOMMENDER,   SVDSurprise,             MF_PARAMS],
     [3,   'FM',                'E',    rr.PIPELINE_RECOMMENDER,   FMStaticTuricreate,      MF_PARAMS],
     [4,   'Item2Vec',          'E',    rr.PIPELINE_EMBEDDINGS,    Item2VecGensim,          ITEM2VEC_EXPLICIT_PARAMS],
     [5,   'User2Vec',          'E',    rr.PIPELINE_EMBEDDINGS,    User2Vec,                USER2VEC_EXPLICIT_PARAMS],
     [6,   'Interact2Vec',      'E',    rr.PIPELINE_EMBEDDINGS,    Interact2VecKeras,       INTERACT2VEC_EXPLICIT_PARAMS],
     [7,   'IS',                'I',    rr.PIPELINE_RECOMMENDER,   ISStaticTuricreate,      KNN_PARAMS],
     [8,   'ALS',               'I',    rr.PIPELINE_RECOMMENDER,   ALSImplicit,             MF_PARAMS],
     [9,   'BPR',               'I',    rr.PIPELINE_RECOMMENDER,   BPRImplicit,             MF_PARAMS],
     [10,  'LMF',               'I',    rr.PIPELINE_RECOMMENDER,   LMFImplicit,             MF_PARAMS],
     [11,  'Item2Vec',          'I',    rr.PIPELINE_EMBEDDINGS,    Item2VecGensim,          ITEM2VEC_IMPLICIT_PARAMS],
     [12,  'User2Vec',          'I',    rr.PIPELINE_EMBEDDINGS,    User2Vec,                USER2VEC_IMPLICIT_PARAMS],
     [13,  'Interact2Vec',      'I',    rr.PIPELINE_EMBEDDINGS,    Interact2VecKeras,       INTERACT2VEC_IMPLICIT_PARAMS],
     [14,  'Recency Static',    'E',    rr.PIPELINE_RECOMMENDER,   RecencyStaticRaw,        RECENCY_PARAMS],
     [15,  'Recency Temporal',  'E',    rr.PIPELINE_RECOMMENDER,   RecencyTemporalRaw,      RECENCY_PARAMS],
     [16,  'IS Timestamp',      'I',    rr.PIPELINE_RECOMMENDER,   ISTemporalTuricreate,    KNN_PARAMS],
     [17,  'FM Temporal',       'E',    rr.PIPELINE_RECOMMENDER,   FMTemporalTuricreate,    MF_PARAMS],
     [18,  'Item2VecKeras',     'I',    rr.PIPELINE_EMBEDDINGS,    Item2VecKeras,           ITEM2VEC_IMPLICIT_PARAMS],
     [19,  'Item2VecPonderado', 'I',    rr.PIPELINE_EMBEDDINGS,    Item2VecPonderado,       ITEM2VEC_PONDERADO_IMPLICIT_PARAMS],
     [20,  'Item2VecSeq',       'I',    rr.PIPELINE_EMBEDDINGS,    Item2VecSequencial,      ITEM2VEC_SEQUENCIAL_IMPLICIT_PARAMS],
     [21,  'ALS-Embeddings',    'I',    rr.PIPELINE_EMBEDDINGS,    ALSEmbeddingsImplicit,   MF_EMBEDDINGS_PARAMS],
     [22,  'BPR-Embeddings',    'I',    rr.PIPELINE_EMBEDDINGS,    BPREmbeddingsImplicit,   MF_EMBEDDINGS_PARAMS],
     [23,  'Item2VecSeq',       'E',    rr.PIPELINE_EMBEDDINGS,    Item2VecSequencial,      ITEM2VEC_SEQUENCIAL_EXPLICIT_PARAMS],
     [24,  'WeightedIt2V',      'I',    rr.PIPELINE_EMBEDDINGS,    Item2VecPonderadoGensim, WI2V_IMPLICIT_PARAMS],
     [25,  'WeightedIt2V',      'E',    rr.PIPELINE_EMBEDDINGS,    Item2VecPonderadoGensim, WI2V_EXPLICIT_PARAMS]],
    columns=[rr.MODEL_ID, rr.MODEL_NAME, rr.MODEL_RECOMMENDATION_TYPE, rr.MODEL_PIPELINE_TYPE, rr.MODEL_CLASS, rr.MODEL_GRID_SEARCH_PARAMS]
)

# ======= Dicionario de recomendadores para modelos de embeddings =======
EMBEDDINGS_RECOMMENDERS = {
    'Implicit KNN': ImplicitKNN,
    'Explicit KNN': ExplicitKNN,
    'UIS': UserItemSimilarity,
    'Implicit KNN User Concat': ImplicitKNNUserConcatenation,
    'UIWS': UserItemWeightedSimilarity,
    'Comitee': Comitee,
    'ModelOutput': NeuralModelOutput
}