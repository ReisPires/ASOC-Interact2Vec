import csv

# Nomes de arquivos da base de dados
FILE_ITEMS = 'items.csv'
FILE_USERS = 'users.csv'
FILE_INTERACTIONS = 'interactions.csv'

# Dados dos CSVs
DELIMITER = ';'
QUOTECHAR = '"'
QUOTING = csv.QUOTE_ALL
ENCODING = "ISO-8859-1"

# Diretorios
DIR_LOG = 'logs'
DIR_DATASETS = 'datasets'
DIR_EMBEDDINGS_GRID_SEARCH = 'embeddings/grid_search'
DIR_EMBEDDINGS_FINAL_EXPERIMENT = 'embeddings/final_experiment'

# Colunas dos arquivos e dataframmes
COLUMN_ITEM_ID = 'id_item'
COLUMN_ITEM_NAME = 'name_item'
COLUMN_USER_ID = 'id_user'
COLUMN_USER_NAME = 'name_user'
COLUMN_INTERACTION = 'interaction'
COLUMN_RANK = 'rank'
COLUMN_DATETIME = 'datetime'
COLUMN_TIMESTAMP = 'timestamp'
COLUMN_DILUTED_INTERACTION = 'diluted_interaction'
COLUMN_WEIGHTS = 'weights'

# Colunas da tabela de dataset
DATASET_ID = 'id'
DATASET_NAME = 'name'
DATASET_TYPE = 'type'
DATASET_TEMPORAL_BEHAVIOUR = 'temporal_behaviour'
DATASET_GDRIVE = 'gdrive'

# Colunas da tabela de modelos
MODEL_ID = 'id'
MODEL_NAME = 'name'
MODEL_RECOMMENDATION_TYPE = 'recommendation_type'
MODEL_PIPELINE_TYPE = 'pipeline_type'
MODEL_CLASS = 'model_class'
MODEL_GRID_SEARCH_PARAMS = 'params'

# Tipos de pipeline
PIPELINE_RECOMMENDER = 'R'
PIPELINE_EMBEDDINGS = 'E'

# Parametros do experimento
EXPERIMENT_TYPE = 'type'
EXPERIMENT_TEMPORAL_BEHAVIOUR = 'temporal_behaviour'
EXPERIMENT_DATASETS = 'datasets'
EXPERIMENT_MODELS = 'models'
EXPERIMENT_GRID_SEARCH = 'grid_search'
EXPERIMENT_FINAL_EXPERIMENT = 'final_experiment'
EXPERIMENT_FAST_MODE = 'fast_mode'
EXPERIMENT_OVERWRITE = 'overwrite'

# Formato temporal
DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S'

# Semente de aleatoriedade
RANDOM_SEED = 2020

# Escala para as avaliacoes
RATING_SCALE_LO = 1
RATING_SCALE_HI = 2

# Taxas de amostragem
SAMPLING_RATE_HYPERPARAMETERIZATION = {
    'Jester': 0.10,
    'LibimSeTi': 0.10,
    'MovieLens': 0.20,
    'NetflixPrize': 0.05
}
SAMPLING_RATE_EXPERIMENT = {    
    'Jester': 1.00,
    'LibimSeTi': 1.00,
    'MovieLens': 1.00,    
    'NetflixPrize': 0.25,
}

# Limite de memória RAM
MEM_SIZE_LIMIT = 1.28e+11

# Diretorios e arquivos para salvar embeddings
FILE_SPARSE_REPR = '{}_sparse_repr.pkl'
FILE_ITEM_EMBEDDINGS = '{}_item_embeddings.pkl'
FILE_USER_EMBEDDINGS = '{}_user_embeddings.pkl'

# Parametros das embeddings
EMB_PARAMS_EMBEDDINGS = 'embeddings'
EMB_PARAMS_RECOMMENDERS = 'recommenders'
EMB_PARAMS_REC_NAME = 'name'
EMB_PARAMS_REC_PARAMS = 'params'

# Marcadores de tempos dos experimentos
TIME_FIT = 'fit'
TIME_PREDICT = 'predict'
TIME_EMBEDDINGS = 'embeddings'
TIME_RECOMMENDERS = 'recommenders'

# Scores
SCORE_RMSE = 'rmse'
SCORE_MAE = 'mae'
SCORE_PRECISION = 'prec'
SCORE_RECALL = 'rec'
SCORE_NDCG = 'ndcg'

# Arquivos JSON
DIR_JSON = 'jsons'
JSON_BEST_PARAMS = 'best_params.json'
JSON_PAST_RESULTS_GRID_SEARCH = 'grid_search.json'
JSON_PAST_RESULTS_FINAL_EXPERIMENT = 'final_experiment.json'

# Valores para o top-n
TOP_N_VALUES = list(range(1, 31))
TOP_N_GRID_SEARCH = 15