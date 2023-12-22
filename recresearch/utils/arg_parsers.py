import ast

def _selector(args, flag_singular, flag_plural, ENTITIES):
    if flag_singular in args:
        target_entities = [int(args[args.index(flag_singular)+1])]        
    elif flag_plural in args:
        target_entities = ast.literal_eval(args[args.index(flag_plural)+1])
    else:
        return ENTITIES
    return [ENTITIES[te] for te in target_entities]

def dataset_selector(args, DATASETS):
    return _selector(args, '--dataset', '--datasets', DATASETS)

def model_selector(args, MODELS):
    return _selector(args, '--model', '--models', MODELS)

