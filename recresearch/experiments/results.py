from ast import literal_eval
import json
import numpy as np
import os

import recresearch as rr

def _read_past_results(grid_search):
    json_filepath = os.path.join(
        rr.DIR_JSON, 
        (rr.JSON_PAST_RESULTS_GRID_SEARCH if grid_search else rr.JSON_PAST_RESULTS_FINAL_EXPERIMENT)
    )
    if not os.path.exists(json_filepath):
        return dict()
    with open(json_filepath, 'r') as prf:
        past_results_json = json.load(prf)
    return past_results_json


def _write_past_results(grid_search, past_results_json):
    os.makedirs(rr.DIR_JSON, exist_ok=True)
    json_filepath = os.path.join(
        rr.DIR_JSON, 
        (rr.JSON_PAST_RESULTS_GRID_SEARCH if grid_search else rr.JSON_PAST_RESULTS_FINAL_EXPERIMENT)
    )
    with open(json_filepath, 'w') as prf:
        json.dump(past_results_json, prf, indent=4, sort_keys=True)
        prf.flush()
        os.fsync(prf.fileno())


def _dict_to_str(dict_var):
    key_value_pairs = list()
    for key, value in sorted(dict_var.items(), key=lambda x: x[0]):
        key_value_pairs.append((
            "'{}'".format(key) if type(key) == str else "{}".format(key),
            "'{}'".format(value) if type(value) == str else "{}".format(value)
        ))    
    return '{' + ', '.join(['{}: {}'.format(key, value) for key, value in key_value_pairs]) + '}'


def _str_to_dict(str_var):
    return literal_eval(str_var)


def get_past_results(temporal_behaviour, recommendation_type, dataset_name, model_name, model_params, grid_search):
    # Le json de historico de resultados
    past_results_json = _read_past_results(grid_search)
    # Converte parametros para formato hasheavel
    model_params = _dict_to_str(model_params)
    # Recupera resultados anteriores
    if (temporal_behaviour in past_results_json
            and recommendation_type in past_results_json[temporal_behaviour]
            and dataset_name in past_results_json[temporal_behaviour][recommendation_type]
            and model_name in past_results_json[temporal_behaviour][recommendation_type][dataset_name]
            and model_params in past_results_json[temporal_behaviour][recommendation_type][dataset_name][model_name]):
        past_results = past_results_json[temporal_behaviour][recommendation_type][dataset_name][model_name][model_params]
    else:
        return None, 0.0
    # Separa scores e tempo de execucao
    scores = _str_to_dict(past_results['scores'])
    elapsed_time = _str_to_dict(past_results['elapsed_time'])
    # Retorna resultados
    return scores, elapsed_time


def update_past_results(temporal_behaviour, recommendation_type, dataset_name, model_name, model_params, grid_search, scores, elapsed_time):
    # Le json de historico de resultados
    past_results_json = _read_past_results(grid_search)
    # Converte parametros e resultados para formato hasheavel
    model_params = _dict_to_str(model_params)    
    # Escreve resultados novos
    if temporal_behaviour not in past_results_json:
        past_results_json[temporal_behaviour] = dict()
    if recommendation_type not in past_results_json[temporal_behaviour]:
        past_results_json[temporal_behaviour][recommendation_type] = dict()
    if dataset_name not in past_results_json[temporal_behaviour][recommendation_type]:
        past_results_json[temporal_behaviour][recommendation_type][dataset_name] = dict()
    if model_name not in past_results_json[temporal_behaviour][recommendation_type][dataset_name]:
        past_results_json[temporal_behaviour][recommendation_type][dataset_name][model_name] = dict()    
    past_results_json[temporal_behaviour][recommendation_type][dataset_name][model_name][model_params] = {
        'scores': _dict_to_str(scores), 
        'elapsed_time': _dict_to_str(elapsed_time)
    }
    # Atualiza arquivo json
    _write_past_results(grid_search, past_results_json)


def delete_past_results(temporal_behaviour=None, recommendation_type=None, dataset_name=None, model_name=None, model_params=None, grid_search=None):
    # Verifica quais arquivos deve mudar
    grid_search = [True, False] if grid_search is None else ([grid_search] if type(grid_search) not in [list, np.ndarray] else grid_search)
    for gs in grid_search:
        # Le json de historico de resultados
        past_results_json = _read_past_results(gs)
        if len(past_results_json) == 0:
            continue
        # Deleta os resultados iterativamente
        delete_temporal_behaviour = list(past_results_json.keys()) if temporal_behaviour is None else ([temporal_behaviour] if type(temporal_behaviour) not in [list, np.ndarray] else temporal_behaviour)
        for tb in delete_temporal_behaviour:
            if tb in past_results_json:
                delete_recommendation_type = list(past_results_json[tb].keys()) if recommendation_type is None else ([recommendation_type] if type(recommendation_type) not in [list, np.ndarray] else recommendation_type)
                for rt in delete_recommendation_type:
                    if rt in past_results_json[tb]:
                        delete_dataset_name = list(past_results_json[tb][rt].keys()) if dataset_name is None else ([dataset_name] if type(dataset_name) not in [list, np.ndarray] else dataset_name)
                        for dn in delete_dataset_name:
                            if dn in past_results_json[tb][rt]:
                                delete_model_name = list(past_results_json[tb][rt][dn].keys()) if model_name is None else ([model_name] if type(model_name) not in [list, np.ndarray] else model_name)
                                for mn in delete_model_name:
                                    if mn in past_results_json[tb][rt][dn]:                                        
                                        delete_model_params = list(past_results_json[tb][rt][dn][mn].keys()) if model_params is None else ([_dict_to_str(model_params)] if type(model_params) not in [list, np.ndarray] else [_dict_to_str(mp) for mp in model_params])
                                        for mp in delete_model_params:
                                            if mp in past_results_json[tb][rt][dn][mn]:
                                                past_results_json[tb][rt][dn][mn].pop(mp)
                                        if len(past_results_json[tb][rt][dn][mn]) == 0:
                                            past_results_json[tb][rt][dn].pop(mn)
                                if len(past_results_json[tb][rt][dn]) == 0:
                                    past_results_json[tb][rt].pop(dn)
                        if len(past_results_json[tb][rt]) == 0:
                            past_results_json[tb].pop(rt)
                if len(past_results_json[tb]) == 0:
                    past_results_json.pop(tb)
        # Escreve o arquivo novo
        _write_past_results(gs, past_results_json)