from ast import literal_eval
import sys

import recresearch as rr
from recresearch.dataset import DATASETS_TABLE
from recresearch.methods import MODELS_TABLE


# Asks when parameter was not defined in argvs
def _parameter_asker(param_name, options, multiple=False, is_str_input=False):
    print('Select {} of experiment:'.format(param_name))
    while True:
        for i, option in options.items():
            print('[{}] {}'.format(i, option))
        print('[0] Cancel')        
        inp = input() if is_str_input else literal_eval(input())
        if (type(inp) == list and multiple == True and all(op in options.keys() for op in inp)) or (((type(inp) == str and is_str_input) or type(inp) == int) and inp in options.keys()):
            return inp if type(inp) == list or multiple == False or is_str_input else [inp]
        elif inp == 0 or inp == '0':
            sys.exit(0)
        else:
            print('Select a valid option')


# Check if parameter was defined in argvs (for mutually exclusive parameters)
def _mutually_exclusive_selector(args, param_name, param_keys_1, param_keys_2, options):
    have_param_1 = any(pv in args for pv in param_keys_1)
    have_param_2 = any(pv in args for pv in param_keys_2)    
    if have_param_1 and have_param_2:
        raise RuntimeError('conflicting values for experiment parameter "{}"'.format(param_name))
    elif have_param_1 or have_param_2:
        return 1 if have_param_1 else 2    
    else:
        return _parameter_asker(param_name, options, multiple=False)

    
def _multiple_selector(args, param_name, param_keys, val_types, options):
    have_param = list(pv in args for pv in param_keys)
    if sum(have_param) == 1:
        param_key = param_keys[have_param.index(True)]
        if val_types != str:
            param_val = literal_eval(args[args.index(param_key)+1])
            if type(param_val) not in val_types:
                raise RuntimeError('invalid value for experiment parameter "{}'.format(param_name))
        else:
            param_val = args[args.index(param_key)+1].lower().strip()
        return param_val
    elif sum(have_param) == 0:
        return _parameter_asker(param_name, options, multiple=True, is_str_input=(True if val_types == str else False))
    else:
        raise RuntimeError('multiple values for experiment parameter "{}"'.format(param_name))


def _type_selector(args):
    op = _mutually_exclusive_selector(args=args, param_name='type', param_keys_1=['--explicit', '-e'], param_keys_2=['--implicit', '-i'], options={1: 'Explicit', 2: 'Implicit'})
    return 'E' if op == 1 else 'I'


def _temporal_behaviour_selector(args):
    op = _mutually_exclusive_selector(args=args, param_name='temporal behaviour', param_keys_1=['--static', '-s'], param_keys_2=['--temporal', '-t'], options={1: 'Static', 2: 'Temporal'})
    return 'S' if op == 1 else 'T'


def _dataset_selector(args, params):    
    valid_datasets = DATASETS_TABLE.copy()
    if params[rr.EXPERIMENT_TYPE] == 'E':
        valid_datasets = valid_datasets[valid_datasets[rr.DATASET_TYPE]=='E']
    if params[rr.EXPERIMENT_TEMPORAL_BEHAVIOUR] == 'T':
        valid_datasets = valid_datasets[valid_datasets[rr.DATASET_TEMPORAL_BEHAVIOUR]=='T']
    valid_datasets = valid_datasets.set_index(rr.DATASET_ID)[rr.DATASET_NAME].to_dict()
    op = _multiple_selector(args=args, param_name='datasets', param_keys=['--dataset', '--datasets', '-d'], val_types=[list, int], options=valid_datasets)
    return DATASETS_TABLE[DATASETS_TABLE[rr.DATASET_ID].isin(op)]


def _model_selector(args, params):    
    valid_models = MODELS_TABLE.copy()
    if params[rr.EXPERIMENT_TYPE] == 'E':
        valid_models = valid_models[valid_models[rr.MODEL_RECOMMENDATION_TYPE]=='E']
    else:
        valid_models = valid_models[valid_models[rr.MODEL_RECOMMENDATION_TYPE]=='I']
    valid_models = valid_models.set_index(rr.MODEL_ID)[rr.MODEL_NAME].to_dict()
    op = _multiple_selector(args=args, param_name='models', param_keys=['--model', '--models', '-m'], val_types=[list, int], options=valid_models)
    return MODELS_TABLE[MODELS_TABLE[rr.MODEL_ID].isin(op)]


def _grid_search_selector(args):
    op = _multiple_selector(
        args=args, param_name='execution of grid search', 
        param_keys=['--grid-search', '--grid', '--gridsearch', '--grid_search', '--gs', '-gs', '-g'], 
        val_types=str, 
        options={'y': 'Yes', 'n': 'No'}
    )    
    return True if op == 'y' else False


def _final_experiment_selector(args):
    op = _multiple_selector(
        args=args, param_name='execution of final experiment', 
        param_keys=['--final-experiment', '--final', '--finalexperiment', '--final_experiment', '--fe', '-fe', '-f'], 
        val_types=str, 
        options={'y': 'Yes', 'n': 'No'}
    )    
    return True if op == 'y' else False


def _fast_mode_selector(args):
    op = _multiple_selector(
        args=args, param_name='execution of fast mode', 
        param_keys=['--fast-mode', '--fast', '--fastmode', '--fast_mode', '--fm', '-fm'], 
        val_types=str, 
        options={'y': 'Yes', 'n': 'No'}
    )    
    return True if op == 'y' else False

def _overwrite_jsons(args):
    op = _multiple_selector(
        args=args, param_name='overwrite of jsons', 
        param_keys=['--overwrite', '-o'], 
        val_types=str, 
        options={'y': 'Yes', 'n': 'No'}
    )    
    return True if op == 'y' else False



def parse_parameters(args):    
    params = dict()
    params[rr.EXPERIMENT_TYPE] = _type_selector(args)
    params[rr.EXPERIMENT_TEMPORAL_BEHAVIOUR] = _temporal_behaviour_selector(args)
    params[rr.EXPERIMENT_DATASETS] = _dataset_selector(args, params)
    params[rr.EXPERIMENT_MODELS] = _model_selector(args, params)
    params[rr.EXPERIMENT_GRID_SEARCH] = _grid_search_selector(args)
    params[rr.EXPERIMENT_FINAL_EXPERIMENT] = _final_experiment_selector(args)
    params[rr.EXPERIMENT_FAST_MODE] = _fast_mode_selector(args)
    params[rr.EXPERIMENT_OVERWRITE] = _overwrite_jsons(args)
    return params