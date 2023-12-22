from datetime import datetime
import pandas as pd

import recresearch as rr
from recresearch.dataset import get_datasets
from recresearch.experiments import parameter_handler as ph
from recresearch.experiments import pipelines as pp
from recresearch.experiments.logger import BasicLogger, ExplicitResultsLogger, ImplicitResultsLogger
from recresearch.utils.model_selection import recsys_train_test_split
from recresearch.utils.preprocessing import recsys_sampling, cut_by_minimal_interactions, remove_cold_start

class ExperimentalProtocol(object):
    def __init__(self, args):        
        # ========== Propriedades do experimento ==========
        self.params = ph.parse_parameters(args)
        # Datasets e modelos
        self.target_datasets = self.params[rr.EXPERIMENT_DATASETS][rr.DATASET_NAME].values
        self.target_models = self.params[rr.EXPERIMENT_MODELS]
        # Tipo de experimento
        self.temporal_behaviour = self.params[rr.EXPERIMENT_TEMPORAL_BEHAVIOUR]
        self.experiment_type = self.params[rr.EXPERIMENT_TYPE]
        self.run_grid_search = self.params[rr.EXPERIMENT_GRID_SEARCH]
        self.run_final_experiment = self.params[rr.EXPERIMENT_FINAL_EXPERIMENT]
        self.fast_mode = self.params[rr.EXPERIMENT_FAST_MODE]
        self.overwrite = self.params[rr.EXPERIMENT_OVERWRITE]
        # =============== Inicializa loggers ===============
        self._set_loggers()

    def get_params(self):
        return self.params

    def _set_loggers(self):
        # Define o nome do logger
        logger_name = '{}_{}_{}'.format(
            self.temporal_behaviour, 
            self.experiment_type, 
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )
        # Constroi o nome de cada arquivo de log
        grid_search_logger_name = logger_name + '_gridsearch.log'
        final_experiment_logger_name = logger_name + '_results.log'
        error_logger_name = logger_name + '_error.log'
        # Instancia as classes de logger
        if self.experiment_type == 'E':
            if self.run_grid_search:
                self.grid_search_logger = ExplicitResultsLogger(grid_search_logger_name)
            self.final_experiment_logger = ExplicitResultsLogger(final_experiment_logger_name)
        elif self.experiment_type == 'I':
            if self.run_grid_search:
                self.grid_search_logger = ImplicitResultsLogger(grid_search_logger_name)
            self.final_experiment_logger = ImplicitResultsLogger(final_experiment_logger_name)
        self.error_logger = BasicLogger(error_logger_name)

    def run(self):
        # Regras do split
        train_size = 0.8
        val_size = 0.1
        test_size = 0.1
        
        # Percorre os datasets
        for dataset in get_datasets(rr.DIR_DATASETS, self.target_datasets, ds_type=self.experiment_type, temporal_behaviour=self.temporal_behaviour):
            
            # Recupera o nome e o dataframe
            dataset_name = dataset.get_name()
            df_full = dataset.get_dataframe()

            # Separa o treino, validacao e teste
            df_train, df_val, df_test = recsys_train_test_split(df_full, train_size, val_size, test_size, temporal_behaviour=self.temporal_behaviour)

            # =================== GRID SEARCH ===================
            if self.run_grid_search:

                # Realiza undersampling para otimizacao de parametros
                df_train_gs = recsys_sampling(df_train, dataset_name, rr.SAMPLING_RATE_HYPERPARAMETERIZATION)

                # Remove usuarios com uma unica interacao
                df_train_gs = cut_by_minimal_interactions(df_train_gs, min_interactions=2)

                # Remove cold start da validacao
                df_val_gs = remove_cold_start(df_train_gs, df_val)

                # Executa os metodos                
                for _, model in self.target_models.iterrows():
                    try:
                        pp.run_pipeline(model, dataset_name, self.temporal_behaviour, self.grid_search_logger, self.error_logger, df_train_gs, df_val_gs, grid_search=True, fast_mode=self.fast_mode, overwrite_json=self.overwrite)
                    except Exception as e:
                        self.error_logger.log('Problemas no grid search do modelo {} na base {}'.format(model[rr.MODEL_NAME], dataset_name))
                        self.error_logger.log(e)
                        continue
                
                # Exclui dataframes para liberar memoria
                del df_train_gs, df_val_gs
            
            # ================ EXPERIMENTO FINAL ================
            if self.run_final_experiment:
                # Junta o treino com a validacao
                df_train_fe = pd.concat([df_train, df_val])

                # Realiza undersampling para experimento
                df_train_fe = recsys_sampling(df_train_fe, dataset_name, rr.SAMPLING_RATE_EXPERIMENT)

                # Remove usuarios com uma unica interacao
                df_train_fe = cut_by_minimal_interactions(df_train_fe, min_interactions=2)

                # Remove cold start do teste
                df_test_fe = remove_cold_start(df_train_fe, df_test)

                # Executa os metodos
                for _, model in self.target_models.iterrows():
                    try:                    
                        pp.run_pipeline(model, dataset_name, self.temporal_behaviour, self.final_experiment_logger, self.error_logger, df_train_fe, df_test_fe, grid_search=False, overwrite_json = self.overwrite)
                    except Exception as e:
                        self.error_logger.log('Problemas no experimento final do modelo {} na base {}'.format(model[rr.MODEL_NAME], dataset_name))
                        self.error_logger.log(e)
                        continue

        print('Experimentos realizados!')