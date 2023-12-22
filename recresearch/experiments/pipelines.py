import numpy as np
from sklearn.model_selection import ParameterGrid
import time

import recresearch as rr
from recresearch.evaluation.metrics import precision_recall_score, ndcg_score, mae_score, rmse_score
from recresearch.experiments.results import get_past_results, update_past_results
from recresearch.methods import EMBEDDINGS_RECOMMENDERS
from recresearch.parameters.best_params import get_best_params, update_best_params

# ===================== Pipeline base =====================
class Pipeline(object):
    def __init__(self, model_info, dataset_name, temporal_behaviour, logger, error_logger):
        self.model_class = model_info[rr.MODEL_CLASS]
        self.model_name = model_info[rr.MODEL_NAME]
        self.dataset_name = dataset_name
        self.temporal_behaviour = temporal_behaviour
        self.logger = logger
        self.error_logger = error_logger
        # Apenas para evitar mensagens de erro
        self.rec_model = None 
        self.recommendation_type = None
        self.is_grid_search = None        

    def _fit(self, model_params, df_train):
        return {rr.TIME_FIT: 0.0}

    def _predict(self, df_test):
        return None, {rr.TIME_PREDICT: 0.0}

    def _evaluate(self, pred, df_test):
        return dict()

    def _log(self, model_params, scores, elapsed_time):
        pass

    def _recomend(self, model_params, df_train, df_test, fast_mode=False, overwrite_json=False):
        scores, elapsed_time = get_past_results(self.temporal_behaviour, self.recommendation_type, self.dataset_name, self.model_name, model_params, self.is_grid_search)        
        if scores is None or overwrite_json:
            if fast_mode:                
                return None
            try:
                fit_time = self._fit(model_params, df_train)
                pred, predict_time = self._predict(df_test)
            except Exception as e:
                self.error_logger.log('Problemas durante {}'.format('grid search' if self.is_grid_search else 'experimento final'))
                self.error_logger.log('Base: {}'.format(self.dataset_name))
                self.error_logger.log('Modelo: {}'.format(self.model_name))                
                self.error_logger.log('Parametros: {}'.format(str(model_params)))
                self.error_logger.log('Erro: {}'.format(e))
                self.error_logger.log('---')
                return None
            scores = self._evaluate(pred, df_test)
            elapsed_time = {**fit_time, **predict_time}
            update_past_results(self.temporal_behaviour, self.recommendation_type, self.dataset_name, self.model_name, model_params, self.is_grid_search, scores, elapsed_time)
        #self._log(model_params, scores, elapsed_time)
        return scores


# ============== Pipeline base do Grid Search ==============
class GridSearchPipeline(Pipeline):
    def __init__(self, model_info, dataset_name, temporal_behaviour, logger, error_logger):
        super().__init__(model_info, dataset_name, temporal_behaviour, logger, error_logger)
        self.is_grid_search = True
        self.grid_search_params = model_info[rr.MODEL_GRID_SEARCH_PARAMS]
    
    def _parameter_iterator(self):
        yield None

    def _get_grid_search_score(self, scores):
        return 0.0

    def run(self, df_train, df_test, fast_mode=False, overwrite_json=False):
        best_score = -np.inf        
        for model_params in self._parameter_iterator():
            scores = self._recomend(model_params, df_train, df_test, fast_mode, overwrite_json)
            if scores is not None:
                grid_search_score = self._get_grid_search_score(scores)
                if best_score < grid_search_score:
                    with open('teste.out', 'a') as f:
                        f.write('{} {}\n'.format(best_score, grid_search_score))
                        f.write('{}\n\n'.format(model_params))
                    best_score = grid_search_score
                    update_best_params(self.temporal_behaviour, self.recommendation_type, self.dataset_name, self.model_name, model_params)


# =========== Pipeline base do Experimento Final ============
class FinalExperimentPipeline(Pipeline):
    def __init__(self, model_info, dataset_name, temporal_behaviour, logger, error_logger):
        super().__init__(model_info, dataset_name, temporal_behaviour, logger, error_logger)
        self.is_grid_search = False

    def run(self, df_train, df_test, fast_mode=False, overwrite_json=False):        
        best_params = get_best_params(self.temporal_behaviour, self.recommendation_type, self.dataset_name, self.model_name)
        print(best_params)
        self._recomend(best_params, df_train, df_test, fast_mode=False, overwrite_json=False) # Fast mode funciona apenas para o grid search


# ========= Pipeline base da Recomendacao Implicita ==========
class ImplicitPipeline(Pipeline):
    def __init__(self, model_info, dataset_name, temporal_behaviour, logger, error_logger):
        super().__init__(model_info, dataset_name, temporal_behaviour, logger, error_logger)
        self.recommendation_type = 'I'

    def _predict(self, df_test):
        start = time.time()
        pred = self.rec_model.predict(df_test, top_n=max(rr.TOP_N_VALUES))
        end = time.time()
        elapsed_time = {rr.TIME_PREDICT: end-start}
        return pred, elapsed_time

    def _evaluate(self, pred, df_test):
        prec_value, rec_value = precision_recall_score(df_test, pred, top_n=rr.TOP_N_VALUES)
        ndcg_value = ndcg_score(df_test, pred, top_n=rr.TOP_N_VALUES)
        scores = {rr.SCORE_PRECISION: prec_value, rr.SCORE_RECALL: rec_value, rr.SCORE_NDCG: ndcg_value}
        return scores

    def _log(self, model_params, scores, elapsed_time):
        self.logger.log(
            dataset=self.dataset_name,
            model=self.model_name,
            params=str(model_params),
            prec=str(scores[rr.SCORE_PRECISION]),
            rec=str(scores[rr.SCORE_RECALL]),
            ndcg=str(scores[rr.SCORE_NDCG]),
            time=str(elapsed_time)
        )

    def _get_grid_search_score(self, scores):
        return scores[rr.SCORE_NDCG][rr.TOP_N_GRID_SEARCH]


# ========= Pipeline base da Recomendacao Explicita ==========
class ExplicitPipeline(Pipeline):
    def __init__(self, model_info, dataset_name, temporal_behaviour, logger, error_logger):
        super().__init__(model_info, dataset_name, temporal_behaviour, logger, error_logger)
        self.recommendation_type = 'E'

    def _predict(self, df_test):
        start = time.time()
        pred = self.rec_model.predict(df_test)
        end = time.time()
        elapsed_time = {rr.TIME_PREDICT: end-start}
        return pred, elapsed_time

    def _evaluate(self, pred, df_test):
        real = df_test.set_index([rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID])[rr.COLUMN_INTERACTION]
        mae_value = mae_score(real, pred)
        rmse_value = rmse_score(real, pred)
        scores = {rr.SCORE_MAE: mae_value, rr.SCORE_RMSE: rmse_value}
        return scores

    def _log(self, model_params, scores, elapsed_time):
        self.logger.log(
            dataset=self.dataset_name,
            model=self.model_name,
            params=str(model_params),
            mae=str(scores[rr.SCORE_MAE]),
            rmse=str(scores[rr.SCORE_RMSE]),
            time=str(elapsed_time)
        )
    
    def _get_grid_search_score(self, scores):
        return -1 * scores[rr.SCORE_RMSE]


# =============== Recomendadores tradicionais ===============
class RecommenderPipeline(Pipeline):
    def _fit(self, model_params, df_train):
        start = time.time()
        self.rec_model = self.model_class(**model_params)
        self.rec_model.fit(df_train)
        end = time.time()
        elapsed_time = {rr.TIME_FIT: end-start}
        return elapsed_time


# ------- Experimento Final -------
class RecommenderFinalExperimentPipeline(RecommenderPipeline, FinalExperimentPipeline):
    pass


class RecommenderImplicitFinalExperimentPipeline(ImplicitPipeline, RecommenderFinalExperimentPipeline): 
    pass

    
class RecommenderExplicitFinalExperimentPipeline(ExplicitPipeline, RecommenderFinalExperimentPipeline): 
    pass


# ---------- Grid Search ----------
class RecommenderGridSearchPipeline(RecommenderPipeline, GridSearchPipeline):
    def _parameter_iterator(self):
        params_grid = ParameterGrid(self.grid_search_params)
        for model_params in params_grid:
            yield model_params

class RecommenderImplicitGridSearchPipeline(ImplicitPipeline, RecommenderGridSearchPipeline): 
    pass

    
class RecommenderExplicitGridSearchPipeline(ExplicitPipeline, RecommenderGridSearchPipeline): 
    pass



# ================== Modelos de embeddings ==================
class EmbeddingsPipeline(Pipeline):
    def _get_embeddings_filepath(self, embeddings_params):
        embeddings_dir = rr.DIR_EMBEDDINGS_GRID_SEARCH if self.is_grid_search else rr.DIR_EMBEDDINGS_FINAL_EXPERIMENT
        params_repr = '_'.join(['[{}]{}'.format(k, v) for k, v in sorted(embeddings_params.items())])
        embeddings_filename = '_'.join([self.temporal_behaviour, self.dataset_name, self.model_name, params_repr])        
        return embeddings_dir, embeddings_filename

    def _generate_embeddings(self, embeddings_dir, embeddings_filename, embeddings_params, df_train):
        emb_model = self.model_class()
        emb_model.generate_embeddings(
            df=df_train,
            embeddings_dir=embeddings_dir,
            embeddings_filename=embeddings_filename,
            **embeddings_params
        )

    def _fit(self, model_params, df_train):
        elapsed_time = {rr.TIME_FIT: dict()}
        # Gera embedding        
        embeddings_params = model_params[rr.EMB_PARAMS_EMBEDDINGS]
        embeddings_dir, embeddings_filename = self._get_embeddings_filepath(embeddings_params)
        start = time.time()
        self._generate_embeddings(embeddings_dir, embeddings_filename, embeddings_params, df_train)
        end = time.time()
        elapsed_time[rr.TIME_FIT][rr.TIME_EMBEDDINGS] = end-start
        # Treina o recomendador        
        recommender_name = model_params[rr.EMB_PARAMS_RECOMMENDERS][rr.EMB_PARAMS_REC_NAME]
        recommender_params = model_params[rr.EMB_PARAMS_RECOMMENDERS][rr.EMB_PARAMS_REC_PARAMS]
        recommender_class = EMBEDDINGS_RECOMMENDERS[recommender_name]
        start = time.time()
        self.rec_model = recommender_class(
            embeddings_dir=embeddings_dir,
            embeddings_filename=embeddings_filename,
            **recommender_params
        )
        self.rec_model.fit(df_train)
        end = time.time()
        elapsed_time[rr.TIME_FIT][rr.TIME_RECOMMENDERS] = end-start
        return elapsed_time


# ------- Experimento Final -------
class EmbeddingsFinalExperimentPipeline(EmbeddingsPipeline, FinalExperimentPipeline):
    pass


class EmbeddingsImplicitFinalExperimentPipeline(ImplicitPipeline, EmbeddingsFinalExperimentPipeline): 
    pass

    
class EmbeddingsExplicitFinalExperimentPipeline(ExplicitPipeline, EmbeddingsFinalExperimentPipeline): 
    pass


# ---------- Grid Search ----------
class EmbeddingsGridSearchPipeline(EmbeddingsPipeline, GridSearchPipeline):
    def _parameter_iterator(self):
        grid_search_embeddings_params = self.grid_search_params[rr.EMB_PARAMS_EMBEDDINGS]
        grid_search_recommenders = self.grid_search_params[rr.EMB_PARAMS_RECOMMENDERS]
        embeddings_params_grid = ParameterGrid(grid_search_embeddings_params)
        for embeddings_params in embeddings_params_grid:
            for recommender_name, grid_search_recommender_params in grid_search_recommenders.items():
                recommender_params_grid = ParameterGrid(grid_search_recommender_params)
                for recommender_params in recommender_params_grid:
                    model_params = {
                        rr.EMB_PARAMS_EMBEDDINGS: embeddings_params,
                        rr.EMB_PARAMS_RECOMMENDERS: {
                            rr.EMB_PARAMS_REC_NAME: recommender_name, 
                            rr.EMB_PARAMS_REC_PARAMS: recommender_params
                        }
                    }
                    yield model_params

class EmbeddingsImplicitGridSearchPipeline(ImplicitPipeline, EmbeddingsGridSearchPipeline): 
    pass

    
class EmbeddingsExplicitGridSearchPipeline(ExplicitPipeline, EmbeddingsGridSearchPipeline): 
    pass




# ================== Seleciona e executa o pipeline adequado ==================
def run_pipeline(model, dataset_name, temporal_behaviour, logger, error_logger, df_train, df_test, grid_search=False, fast_mode=False, overwrite_json = False):
    # Recupera informacoes relevantes do pipeline
    recommendation_type = model[rr.MODEL_RECOMMENDATION_TYPE]
    pipeline_type = model[rr.MODEL_PIPELINE_TYPE]
    
    # Seleciona o pipeline
    if pipeline_type == rr.PIPELINE_RECOMMENDER and recommendation_type == 'E' and grid_search == True:
        pipeline = RecommenderExplicitGridSearchPipeline(model, dataset_name, temporal_behaviour, logger, error_logger)
    elif pipeline_type == rr.PIPELINE_RECOMMENDER and recommendation_type == 'E' and grid_search == False:
        pipeline = RecommenderExplicitFinalExperimentPipeline(model, dataset_name, temporal_behaviour, logger, error_logger)
    elif pipeline_type == rr.PIPELINE_RECOMMENDER and recommendation_type == 'I' and grid_search == True:
        pipeline = RecommenderImplicitGridSearchPipeline(model, dataset_name, temporal_behaviour, logger, error_logger)
    elif pipeline_type == rr.PIPELINE_RECOMMENDER and recommendation_type == 'I' and grid_search == False:
        pipeline = RecommenderImplicitFinalExperimentPipeline(model, dataset_name, temporal_behaviour, logger, error_logger)
    elif pipeline_type == rr.PIPELINE_EMBEDDINGS and recommendation_type == 'E' and grid_search == True:
        pipeline = EmbeddingsExplicitGridSearchPipeline(model, dataset_name, temporal_behaviour, logger, error_logger)
    elif pipeline_type == rr.PIPELINE_EMBEDDINGS and recommendation_type == 'E' and grid_search == False:
        pipeline = EmbeddingsExplicitFinalExperimentPipeline(model, dataset_name, temporal_behaviour, logger, error_logger)
    elif pipeline_type == rr.PIPELINE_EMBEDDINGS and recommendation_type == 'I' and grid_search == True:
        pipeline = EmbeddingsImplicitGridSearchPipeline(model, dataset_name, temporal_behaviour, logger, error_logger)
    elif pipeline_type == rr.PIPELINE_EMBEDDINGS and recommendation_type == 'I' and grid_search == False:
        pipeline = EmbeddingsImplicitFinalExperimentPipeline(model, dataset_name, temporal_behaviour, logger, error_logger)
    else:
        raise Exception('Invalid recommendation or pipeline type!')
    pipeline.run(df_train, df_test, fast_mode, overwrite_json)
