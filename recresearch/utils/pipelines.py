import numpy as np
from sklearn.model_selection import ParameterGrid
import time

import recresearch as rr
from recresearch.evaluation.metrics import precision_recall_score, ndcg_score, mae_score, rmse_score
from recresearch.methods import EMBEDDINGS_RECOMMENDERS
from recresearch.parameters.best_params import get_best_params, update_best_params

# ===================== Pipeline base =====================
class Pipeline(object):
    def __init__(self, model_info, dataset_name, temporal_behaviour, logger):
        self.model_class = model_info[rr.MODEL_CLASS]
        self.model_name = model_info[rr.MODEL_NAME]
        self.dataset_name = dataset_name
        self.temporal_behaviour = temporal_behaviour
        self.logger = logger
        self.rec_model = None # Apenas para evitar mensagens de erro

    def _fit(self, model_params, df_train):
        return {'fit': 0.0}

    def _predict(self, df_test, top_n=None):
        return None, {'predict': 0.0}

    def _evaluate(self, pred, df_test, top_n=None):
        return dict()

    def _log(self, model_params, scores, elapsed_time):
        pass

    def _recomend(self, model_params, df_train, df_test, top_n=None):
        fit_time = self._fit(model_params, df_train)
        pred, predict_time = self._predict(df_test, top_n)
        scores = self._evaluate(pred, df_test, top_n)
        elapsed_time = {**fit_time, **predict_time}
        self._log(model_params, scores, elapsed_time)
        return scores


# ============== Pipeline base do Grid Search ==============
class GridSearchPipeline(Pipeline):
    def __init__(self, model_info, dataset_name, temporal_behaviour, logger):
        super().__init__(model_info, dataset_name, temporal_behaviour, logger)
        self.is_grid_search = True
        self.grid_search_params = model_info[rr.MODEL_GRID_SEARCH_PARAMS]
    
    def _parameter_iterator(self):
        yield None

    def _get_grid_search_score(self, scores):
        return 0.0

    def run(self, df_train, df_test, top_n=None):
        best_score = -np.inf
        best_params = dict()
        for model_params in self._parameter_iterator():
            scores = self._recomend(model_params, df_train, df_test, top_n)
            grid_search_score = self._get_grid_search_score(scores)
            if best_score < grid_search_score:
                best_score = grid_search_score
                best_params = model_params
        update_best_params(self.temporal_behaviour, self.dataset_name, self.model_name, best_params)


# =========== Pipeline base do Experimento Final ============
class FinalPipeline(Pipeline):
    def __init__(self, model_info, dataset_name, temporal_behaviour, logger):
        super().__init__(model_info, dataset_name, temporal_behaviour, logger)
        self.is_grid_search = False

    def run(self, df_train, df_test, top_n=None):
        best_params = get_best_params(self.temporal_behaviour, self.dataset_name, self.model_name)
        self._recomend(best_params, df_train, df_test, top_n)


# ========= Pipeline base da Recomendacao Implicita ==========
class ImplicitPipeline(Pipeline):    
    def _predict(self, df_test, top_n):
        start = time.time()
        pred = self.rec_model.predict(df_test, top_n=max(top_n))
        end = time.time()
        elapsed_time = {'predict': end-start}
        return pred, elapsed_time

    def _evaluate(self, pred, df_test, top_n):
        prec_value, rec_value = precision_recall_score(df_test, pred, top_n=top_n)
        ndcg_value = ndcg_score(df_test, pred, top_n=top_n)
        scores = {'prec': prec_value, 'rec': rec_value, 'ndcg': ndcg_value}
        return scores

    def _log(self, model_params, scores, elapsed_time):
        self.logger.log(
            dataset=self.dataset_name,
            model=self.model_name,
            params=str(model_params),
            prec=str(scores['prec']),
            rec=str(scores['rec']),
            ndcg=str(scores['ndcg']),
            time=str(elapsed_time)
        )

    def _get_grid_search_score(self, scores):
        return scores['ndcg']


# ========= Pipeline base da Recomendacao Explicita ==========
class ExplicitPipeline(Pipeline):
    def _predict(self, df_test, top_n=None):
        start = time.time()
        pred = self.rec_model.predict(df_test)
        end = time.time()
        elapsed_time = {'predict': end-start}
        return pred, elapsed_time

    def _evaluate(self, pred, df_test, top_n=None):
        real = df_test.set_index([rr.COLUMN_USER_ID, rr.COLUMN_ITEM_ID])[rr.COLUMN_INTERACTION]
        mae_value = mae_score(real, pred)
        rmse_value = rmse_score(real, pred)
        scores = {'mae': mae_value, 'rmse': rmse_value}        
        return scores

    def _log(self, model_params, scores, elapsed_time):
        self.logger.log(
            dataset=self.dataset_name,
            model=self.model_name,
            params=str(model_params),
            mae=str(scores['mae']),
            rmse=str(scores['rmse']),
            time=str(elapsed_time)
        )
    
    def _get_grid_search_score(self, scores):
        return scores['rmse']


# =============== Recomendadores tradicionais ===============
class RecommenderFinalPipeline(FinalPipeline):
    def _fit(self, model_params, df_train):
        start = time.time()
        self.rec_model = self.model_class(**model_params)
        self.rec_model.fit(df_train)
        end = time.time()
        elapsed_time = {'fit': end-start}
        return elapsed_time


class RecommenderImplicitFinalPipeline(ImplicitPipeline, RecommenderFinalPipeline): 
    pass

    
class RecommenderExplicitFinalPipeline(ExplicitPipeline, RecommenderFinalPipeline): 
    pass

# ---------- Grid Search ----------
class RecommenderGridSearchPipeline(GridSearchPipeline):
    def _parameter_iterator(self):
        params_grid = ParameterGrid(self.grid_search_params)
        for model_params in params_grid:
            yield model_params

class RecommenderImplicitGridSearchPipeline(ImplicitPipeline, RecommenderGridSearchPipeline): 
    pass

    
class RecommenderExplicitGridSearchPipeline(ExplicitPipeline, RecommenderGridSearchPipeline): 
    pass


# ================== Modelos de embeddings ==================
class EmbeddingsFinalPipeline(FinalPipeline):
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
        elapsed_time = {'fit': dict()}
        # Gera embedding        
        embeddings_params = model_params['embeddings']
        embeddings_dir, embeddings_filename = self._get_embeddings_filepath(embeddings_params)
        start = time.time()
        self._generate_embeddings(embeddings_dir, embeddings_filename, embeddings_params, df_train)
        end = time.time()
        elapsed_time['fit']['embeddings'] = end-start
        # Treina o recomendador        
        recommender_name = model_params['recommenders']['name']
        recommender_params = model_params['recommenders']['params']
        recommender_class = EMBEDDINGS_RECOMMENDERS[recommender_name]
        start = time.time()
        self.rec_model = recommender_class(
            embeddings_dir=embeddings_dir,
            embeddings_filename=embeddings_filename,
            **recommender_params
        )
        self.rec_model.fit(df_train)
        end = time.time()
        elapsed_time['fit']['recommenders'] = end-start
        return elapsed_time


class EmbeddingsImplicitFinalPipeline(ImplicitPipeline, RecommenderFinalPipeline): 
    pass

    
class EmbeddingsExplicitFinalPipeline(ExplicitPipeline, RecommenderFinalPipeline): 
    pass


# ---------- Grid Search ----------
class EmbeddingsGridSearchPipeline(GridSearchPipeline):
    def _parameter_iterator(self):
        grid_search_embeddings_params = self.grid_search_params['embeddings']
        grid_search_recommenders = self.grid_search_params['recommenders']
        embeddings_params_grid = ParameterGrid(grid_search_embeddings_params)
        for embeddings_params in embeddings_params_grid:
            for recommender_name, grid_search_recommender_params in grid_search_recommenders.items():
                recommender_params_grid = ParameterGrid(grid_search_recommender_params)
                for recommender_params in recommender_params_grid:
                    model_params = {
                        'embeddings': embeddings_params,
                        'recommenders': {'name': recommender_name, 'params': recommender_params}
                    }
                    yield model_params

class EmbeddingsImplicitGridSearchPipeline(ImplicitPipeline, RecommenderGridSearchPipeline): 
    pass

    
class EmbeddingsExplicitGridSearchPipeline(ExplicitPipeline, RecommenderGridSearchPipeline): 
    pass




# ================== Seleciona e executa o pipeline adequado ==================
def run_pipeline(model, dataset_name, temporal_behaviour, logger, df_train, df_test, top_n=None, grid_search=False):
    # Recupera informacoes relevantes do pipeline
    recommendation_type = model[rr.MODEL_RECOMMENDATION_TYPE]
    pipeline_type = model[rr.PIPELINE_RECOMMENDER]
    
    # Seleciona o pipeline
    if pipeline_type == rr.PIPELINE_RECOMMENDER and recommendation_type == 'E' and grid_search == True:
        pipeline = RecommenderExplicitGridSearchPipeline(model, dataset_name, temporal_behaviour, logger)
    elif pipeline_type == rr.PIPELINE_RECOMMENDER and recommendation_type == 'E' and grid_search == False:
        pipeline = RecommenderExplicitFinalPipeline(model, dataset_name, temporal_behaviour, logger)
    elif pipeline_type == rr.PIPELINE_RECOMMENDER and recommendation_type == 'I' and grid_search == True:
        pipeline = RecommenderImplicitGridSearchPipeline(model, dataset_name, temporal_behaviour, logger)
    elif pipeline_type == rr.PIPELINE_RECOMMENDER and recommendation_type == 'I' and grid_search == False:
        pipeline = RecommenderImplicitFinalPipeline(model, dataset_name, temporal_behaviour, logger)
    elif pipeline_type == rr.PIPELINE_EMBEDDINGS and recommendation_type == 'E' and grid_search == True:
        pipeline = EmbeddingsExplicitGridSearchPipeline(model, dataset_name, temporal_behaviour, logger)
    elif pipeline_type == rr.PIPELINE_EMBEDDINGS and recommendation_type == 'E' and grid_search == False:
        pipeline = EmbeddingsExplicitFinalPipeline(model, dataset_name, temporal_behaviour, logger)
    elif pipeline_type == rr.PIPELINE_EMBEDDINGS and recommendation_type == 'I' and grid_search == True:
        pipeline = EmbeddingsImplicitGridSearchPipeline(model, dataset_name, temporal_behaviour, logger)
    elif pipeline_type == rr.PIPELINE_EMBEDDINGS and recommendation_type == 'I' and grid_search == False:
        pipeline = EmbeddingsImplicitFinalPipeline(model, dataset_name, temporal_behaviour, logger)
    else:
        raise Exception('Invalid recommendation or pipeline type!')
    pipeline.run(df_train, df_test, top_n)
