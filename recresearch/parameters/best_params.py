import json
import os

import recresearch as rr

def _read_best_params_file(create_new=False):
    json_filepath = os.path.join(rr.DIR_JSON, rr.JSON_BEST_PARAMS)
    if not os.path.exists(json_filepath):
        if create_new:
            return dict()
        else:
            raise RuntimeError('json file with best parameters does not exist')
    else:
        with open(json_filepath, 'r') as bpf:
            best_params_json = json.load(bpf)
        return best_params_json
        
def _write_best_params_file(best_params_json):
    os.makedirs(rr.DIR_JSON, exist_ok=True)
    json_filepath = os.path.join(rr.DIR_JSON, rr.JSON_BEST_PARAMS)
    with open(json_filepath, 'w') as bpf:
        json.dump(best_params_json, bpf, indent=4, sort_keys=True)
        bpf.flush()
        os.fsync(bpf.fileno())

def get_best_params(temporal_behaviour, recommendation_type, dataset_name, model_name):
    best_params_json = _read_best_params_file()    
    return best_params_json[temporal_behaviour][recommendation_type][dataset_name][model_name]

def update_best_params(temporal_behaviour, recommendation_type, dataset_name, model_name, best_params):
    best_params_json = _read_best_params_file(create_new=True)
    if temporal_behaviour not in best_params_json:
        best_params_json[temporal_behaviour] = dict()
    if recommendation_type not in best_params_json[temporal_behaviour]:
        best_params_json[temporal_behaviour][recommendation_type] = dict()
    if dataset_name not in best_params_json[temporal_behaviour][recommendation_type]:
        best_params_json[temporal_behaviour][recommendation_type][dataset_name] = dict()
    best_params_json[temporal_behaviour][recommendation_type][dataset_name][model_name] = best_params
    _write_best_params_file(best_params_json)