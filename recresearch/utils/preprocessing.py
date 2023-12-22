import recresearch as rr


# Realiza sampling da base
def recsys_sampling(df, ds_name, sampling_rates, random_seed=rr.RANDOM_SEED):       
    if ds_name in sampling_rates:
        us_rate = sampling_rates[ds_name] if type(sampling_rates[ds_name]) == int else int(len(df) * sampling_rates[ds_name])
        return df.sample(us_rate, random_state=random_seed).copy()
    return df.copy()


# Corta itens e usuarios com menos interacoes que o limite
def cut_by_minimal_interactions(df, min_interactions=1, cut_users=True, cut_items=True):
    # Inicializa variaveis de controle e dataframe
    users_invalid, items_invalid = cut_users, cut_items    
    cut_df = df.copy()

    # Enquanto nao satisfazer os numeros minimos...
    while users_invalid or items_invalid:
        
        # Corta usuarios
        if cut_users and users_invalid:
            qt_interactions = cut_df.groupby(rr.COLUMN_USER_ID).size()
            if (qt_interactions<min_interactions).any():
                cut_df = cut_df[cut_df[rr.COLUMN_USER_ID].isin(qt_interactions[qt_interactions>=min_interactions].index)]
                items_invalid = True and cut_items
            else:
                users_invalid = False
        
        # Corta itens
        if cut_items and items_invalid:
            qt_interactions = cut_df.groupby(rr.COLUMN_ITEM_ID).size()
            if (qt_interactions<min_interactions).any():
                cut_df = cut_df[cut_df[rr.COLUMN_ITEM_ID].isin(qt_interactions[qt_interactions>=min_interactions].index)]
                users_invalid = True and cut_users
            else:
                items_invalid = False
    return cut_df


# Remove itens e usuarios que nao estao presentes no teste
def remove_cold_start(df_train, df_test):
    return df_test[
        (df_test[rr.COLUMN_USER_ID].isin(df_train[rr.COLUMN_USER_ID]))
        &(df_test[rr.COLUMN_ITEM_ID].isin(df_train[rr.COLUMN_ITEM_ID]))
    ].copy()