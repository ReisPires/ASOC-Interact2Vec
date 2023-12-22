from sklearn.model_selection import train_test_split

import recresearch as rr

# Separa em treino, validacao e teste
def recsys_train_test_split(df, train_size=1, val_size=0, test_size=0, temporal_behaviour='S', shuffle=True, random_seed=rr.RANDOM_SEED):    
    # Recupera a quantidade de amostras por particao
    n_samples = len(df)
    n_samples_train = round(n_samples * train_size)
    n_samples_val = round(n_samples * val_size) if val_size != 0 else 0
    if test_size != 0:
        n_samples_test = round(n_samples * test_size) if train_size + val_size + test_size != 1 else (n_samples - n_samples_train - n_samples_val)
    else:
        n_samples_test = 0

    # Faz o split
    if temporal_behaviour == 'T': # Temporal
        df = df.sort_values(rr.COLUMN_DATETIME)
        train_limit_date = df.iloc[n_samples_train-1][rr.COLUMN_DATETIME]
        val_limit_date = df.iloc[n_samples_train+n_samples_val-1][rr.COLUMN_DATETIME]
        test_limit_date = df.iloc[n_samples_train+n_samples_val+n_samples_test-1][rr.COLUMN_DATETIME]
        df_train = df[df[rr.COLUMN_DATETIME]<=train_limit_date]
        df_val = df[(df[rr.COLUMN_DATETIME]>train_limit_date)&(df[rr.COLUMN_DATETIME]<=val_limit_date)]
        df_test = df[(df[rr.COLUMN_DATETIME]>val_limit_date)&(df[rr.COLUMN_DATETIME]<=test_limit_date)]
    
    else: # Estatico
        if shuffle:
            df = df.sample(frac=1, random_state=random_seed)
        df_train, remaining_data = train_test_split(df, train_size=n_samples_train, shuffle=False, random_state=random_seed)        
        if n_samples_val != 0 and n_samples_test != 0:
            df_val, df_test = train_test_split(remaining_data, train_size=n_samples_val, test_size=n_samples_test, shuffle=False, random_state=random_seed)
        elif n_samples_val != 0:
            if n_samples_val < len(remaining_data):
                df_val, _ = train_test_split(remaining_data, train_size=n_samples_val, shuffle=False, random_state=random_seed)
            else:
                df_val = remaining_data.copy()
        elif n_samples_test != 0:
            if n_samples_test < len(remaining_data):
                df_test, _ = train_test_split(remaining_data, train_size=n_samples_test, shuffle=False, random_state=random_seed)
            else:
                df_test = remaining_data.copy()

    # Retorno da funcao
    if n_samples_val != 0 and n_samples_test != 0:
        return df_train, df_val, df_test
    elif n_samples_val != 0:
        return df_train, df_val
    elif n_samples_test != 0:
        return df_train, df_test

