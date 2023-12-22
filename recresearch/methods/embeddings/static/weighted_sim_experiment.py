import numpy as np
from sklearn.model_selection import KFold, ParameterGrid
import sys

import recresearch as rr
from recresearch.experiments import parameter_handler as ph
from recresearch.dataset import get_datasets

# ========== Propriedades do experimento ==========
params = ph.parse_parameters(sys.args)
# Datasets e modelos
target_datasets = params[rr.EXPERIMENT_DATASETS][rr.DATASET_NAME].values
target_models = params[rr.EXPERIMENT_MODELS]
# Tipo de experimento
run_grid_search = params[rr.EXPERIMENT_GRID_SEARCH]
run_final_experiment = params[rr.EXPERIMENT_FINAL_EXPERIMENT]
fast_mode = params[rr.EXPERIMENT_FAST_MODE]
# =================================================


# =========== Parametros do grid search ===========
EMB_PARAMS = {
    'ALS': {
        'n_factors': [50, 100, 200, 300],
        'regularization': [1e-6, 1e-4, 1e-2],
        'n_epochs': [10, 20, 50, 100]
    },
    'BPR': {
        'n_factors': [50, 100, 200, 300],
        'regularization': [1e-6, 1e-4, 1e-2],
        'n_epochs': [10, 20, 50, 100],
        'learning_rate': [0.0025, 0.025, 0.25]
    }
}

REC_PARAMS = {
    'UIWS': {
        'k': [64], 
        'user_item_weights': [(0.90, 0.10), (0.75, 0.25), (0.50, 0.50), (0.25, 0.75), (0.10, 0.90)],
        'similarity_metric': ['dot']
    },
    'DOT': {
        'similarity_metric': ['dot']
    }
}

# =================================================


# =========== Parametros do experimento ===========
# K-fold
n_folds = 5

# Caminhos de arquivos
embeddings_dir = 'embeddings/weighted_sim_experiment'
# =================================================


# ================ Inicializacoes =================
# Verifica se ja executou experimento especifico
def already_executed_inner(results, emb_name, str_repr_emb_params, rec_name, str_repr_rec_params, kf):
    if (
        emb_name in results['inner'] 
        and str_repr_emb_params in results['inner'][emb_name]
        and rec_name in results['inner'][emb_name][str_repr_emb_params]
        and str_repr_rec_params in results['inner'][emb_name][str_repr_emb_params][rec_name]
        and kf in results['inner'][emb_name][str_repr_emb_params][rec_name][str_repr_rec_params]
    ):
        return True
    else:
        return False

def already_executed_outer(results, emb_name, rec_name, kf):
    if (
        emb_name in results['outer'] 
        and rec_name in results['outer'][emb_name]
        and kf in results['outer'][emb_name][rec_name]
    ):
        return True
    else:
        return False

# Dicionario de resultados
results = {'inner': dict(), 'outer': dict()}
# =================================================

# Percorre os datasets
for dataset in get_datasets(rr.DIR_DATASETS, target_datasets, ds_type='I', temporal_behaviour='S'):

    # Recupera o nome e o dataframe
    dataset_name = dataset.get_name()
    df_full = dataset.get_dataframe()

    # Outer CV
    outer_kf = KFold(n_splits=n_folds, shuffle=True, random_state=rr.RANDOM_SEED)
    for okf, (outer_index_train, outer_index_test) in enumerate(outer_kf.split(df_full), start=1):
        outer_df_train = df_full.iloc[outer_index_train]
        outer_df_test = df_full.iloc[outer_index_test]
        
        # Inner CV
        inner_kf = KFold(n_splits=n_folds, shuffle=True, random_state=rr.RANDOM_SEED+1)
        for ikf, (inner_index_train, inner_index_test) in enumerate(inner_kf.split(outer_df_train), start=1):
            inner_df_train = outer_df_train.iloc[inner_index_train]
            inner_df_test = outer_df_train.iloc[inner_index_test]

            # Percorre modelos de embeddings
            for emb_name, EmbModel in [('ALS', ALSEmbeddingsImplicit), ('BPR', BPREmbeddingsImplicit)]:
                # Percorre parametros de modelos de embeddings
                emb_params_grid = ParameterGrid(EMB_PARAMS[emb_name])
                for emb_params_ in emb_params_grid:
                    str_repr_emb_params = str(emb_params_)
                    # Percorre modelos de recomendacao
                    for rec_name, RecModel in [('DOT', dot), ('UIWS', uiws)]:
                        # Percorre parametros de modelos de recomendacao
                        rec_params_grid = ParameterGrid(REC_PARAMS[rec_name])                        
                        for rec_params_ in rec_params_grid:
                            str_repr_rec_params = str(rec_params_)
                            # Verifica se ja executou
                            if not already_executed_inner(results, 'inner', emb_name, str_repr_emb_params, rec_name, str_repr_rec_params, ikf):
                                # Gera embeddings
                                emb_model = EmbModel()
                                emb_model.generate_embeddings(inner_df_train, embeddings_dir, embeddings_filename, **emb_params_)
                                # Gera recomendacao
                                rec_model = RecModel(embeddings_dir, embeddings_filename, **rec_params_)
                                rec_model.fit(inner_df_train)
                                recommendations = rec_model.predict(inner_df_test, top_n=max(rr.TOP_N_VALUES))
                                # Avalia
                                prec_value, rec_value = precision_recall_score(inner_df_test, recommendations, top_n=rr.TOP_N_VALUES)
                                ndcg_value = ndcg_score(inner_df_test, recommendations, top_n=rr.TOP_N_VALUES)
                                # Salva resultados
                                if emb_name not in results['inner']:
                                    results['inner'][emb_name] = dict()
                                if str_repr_emb_params not in results['inner'][emb_name]:
                                    results['inner'][emb_name][str_repr_emb_params] = dict()
                                if rec_name not in results['inner'][emb_name][str_repr_emb_params]:
                                    results['inner'][emb_name][str_repr_emb_params][rec_name] = dict()
                                if str_repr_rec_params not in results['inner'][emb_name][str_repr_emb_params][rec_name]:
                                    results['inner'][emb_name][str_repr_emb_params][rec_name][str_repr_rec_params] = dict()
                                results['inner'][emb_name][str_repr_emb_params][rec_name][str_repr_rec_params][ikf] = {rr.SCORE_PRECISION: prec_value, rr.SCORE_RECALL: rec_value, rr.SCORE_NDCG: ndcg_value}

        # Encontra melhores parametros        
        best_params = {
            'ALS': {
                'DOT': {'params': dict(), 'score': 0},
                'UIWS': {'params': dict(), 'score': 0}
            },
            'BPR': {
                'DOT': {'params': dict(), 'score': 0},
                'UIWS': {'params': dict(), 'score': 0}
            }
        }
        for emb_name in ['ALS', 'BPR']:
            for rec_name in ['DOT', 'UIWS']:
                emb_params_grid = ParameterGrid(EMB_PARAMS[emb_name])
                rec_params_grid = ParameterGrid(REC_PARAMS[rec_name])
                for emb_params_ in emb_params_grid:
                    str_repr_emb_params = str(emb_params_)
                    for rec_params_ in rec_params_grid:
                        str_repr_rec_params = str(rec_params_)
                        score = np.mean([results['inner'][emb_name][str_repr_emb_params][rec_name][str_repr_rec_params][bkf] for bkf in range(1, n_folds+1)])
                        if score > best_params[emb_name][rec_name]['score']:
                            best_params[emb_name][rec_name]['score'] = score
                            best_params[emb_name][rec_name]['params'] = {'emb': emb_params_, 'rec': rec_params_}

        # Treina outer
        for emb_name in ['ALS', 'BPR']:
            for rec_name in ['DOT', 'UIWS']:
                emb_params_ = best_params[emb_name][rec_name]['params']['emb']                
                rec_params_ = best_params[emb_name][rec_name]['params']['rec']                
                # Verifica se ja executou
                if not already_executed_outer(results, emb_name, rec_name, okf):
                    # Gera embeddings
                    emb_model = EmbModel()
                    emb_model.generate_embeddings(outer_df_train, embeddings_dir, embeddings_filename, **emb_params_)
                    # Gera recomendacao
                    rec_model = RecModel(embeddings_dir, embeddings_filename, **rec_params_)
                    rec_model.fit(outer_df_train)
                    recommendations = rec_model.predict(outer_df_test, top_n=max(rr.TOP_N_VALUES))
                    # Avalia
                    prec_value, rec_value = precision_recall_score(outer_df_test, recommendations, top_n=rr.TOP_N_VALUES)
                    ndcg_value = ndcg_score(outer_df_test, recommendations, top_n=rr.TOP_N_VALUES)
                    # Salva resultados
                    if emb_name not in results['outer']:
                        results['outer'][emb_name] = dict()
                    if rec_name not in results['outer'][emb_name]:
                        results['outer'][emb_name][rec_name] = dict()                    
                    results['outer'][emb_name][rec_name][okf] = {rr.SCORE_PRECISION: prec_value, rr.SCORE_RECALL: rec_value, rr.SCORE_NDCG: ndcg_value}

        