from sklearn.model_selection import train_test_split

import recresearch as rr
from recresearch.utils.dilution_utils import get_dilution_parameter
from recresearch.utils.dilution_utils import linear, quadratic, exponential










def remove_implicit_interactions(df_list):
    clean_df_list = list()
    for df in df_list:
        clean_df_list.append(df[(df[rr.COLUMN_INTERACTION]>=rr.RATING_SCALE_LO)&(df[rr.COLUMN_INTERACTION]<=rr.RATING_SCALE_HI)])
    return clean_df_list


def remove_explicit_interactions(df_list):
    clean_df_list = list()
    for df in df_list:
        clean_df_list.append(df[(df[rr.COLUMN_INTERACTION]<rr.RATING_SCALE_LO)|(df[rr.COLUMN_INTERACTION]>rr.RATING_SCALE_HI)])
    return clean_df_list



# def temporal_dilution(df, current_date, days_until_dilution=365, dilution_function='linear'):
#     # Recupera função de diluição
#     if dilution_function == 'linear':
#         dilution_function = linear
#     elif dilution_function == 'quadratic':
#         dilution_function = quadratic
#     elif dilution_function == 'exponential':
#         dilution_function = exponential
#     else:
#         raise Exception('Invalid dilution function')

#     # Captura parâmetro
#     param = get_dilution_parameter(dilution_function, days_until_dilution)
    
#     # Realiza a diluição    
#     elapsed_days = (current_date - df[rr.COLUMN_DATETIME]).apply(lambda d: d.days)    
#     df[rr.COLUMN_DILUTED_INTERACTION] = dilution_function(param, elapsed_days.values)
#     mean_interaction = (rr.RATING_SCALE_HI + rr.RATING_SCALE_LO) / 2
    
#     pos_interactions = df[df[rr.COLUMN_INTERACTION]>=mean_interaction]
#     neg_interactions = df[df[rr.COLUMN_INTERACTION]<mean_interaction]

#     pos_interactions = pos_interactions[rr.COLUMN_DILUTED_INTERACTION] * (pos_interactions[rr.COLUMN_INTERACTION] - mean_interaction) + mean_interaction
#     neg_interactions = (1 - neg_interactions[rr.COLUMN_DILUTED_INTERACTION]) * (mean_interaction - neg_interactions[rr.COLUMN_INTERACTION]) + neg_interactions[rr.COLUMN_INTERACTION]

#     df.loc[pos_interactions.index, rr.COLUMN_DILUTED_INTERACTION] = pos_interactions
#     df.loc[neg_interactions.index, rr.COLUMN_DILUTED_INTERACTION] = neg_interactions

#     return df