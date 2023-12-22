from sklearn.preprocessing import MinMaxScaler
from gensim.models import Word2Vec
import itertools
from keras import Model, regularizers, initializers
from keras.layers import Input, Embedding, Activation, dot, Reshape
from keras.optimizers import Adam
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
import pickle
import scipy

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf
import recresearch as rr
from recresearch.dataset import SparseRepr

class Item2VecPonderadoGensim(object):
    def generate_embeddings(self, df, embeddings_dir, embeddings_filename, embedding_dim=100, n_epochs=5, negative_sampling=5, negative_exponent=0.75, subsampling_p=None, regularization_lambda=None, batch_size=10000, posw_method='default', negw_method='default', posw=1.0, negw=1.0):
        # Verifica se embeddings ja foram criadas previamente
        sparse_repr_filepath = os.path.join(embeddings_dir, rr.FILE_SPARSE_REPR.format(embeddings_filename))
        item_embeddings_filepath = os.path.join(embeddings_dir, rr.FILE_ITEM_EMBEDDINGS.format(embeddings_filename))
        if os.path.exists(sparse_repr_filepath) and os.path.exists(item_embeddings_filepath):        
            print('Embeddings já criadas...')
            return
        
        sparse_repr = SparseRepr(df)
        sparse_matrix = sparse_repr.get_matrix(df[rr.COLUMN_USER_ID], df[rr.COLUMN_ITEM_ID])

        print('Gerando arquivo de interações...')        
        fid = 0
        interactions_file = 'item2vec_interactions_{}.temp'.format(fid)
        while os.path.exists(interactions_file):
            fid += 1
            interactions_file = 'item2vec_interactions_{}.temp'.format(fid)        
        with open(interactions_file, 'w') as f:
            for user in range(sparse_matrix.shape[0]):
                f.write(' '.join(sparse_matrix[user].nonzero()[1].astype(str)) + '\n')

        print('Gerando dataframe de pesos...')
        # Verifica se ja gerou
        dataset_name = embeddings_filename.split('_')[1]
        weights_df_filepath = os.path.join('weights_dfs', dataset_name + '.pkl')
        if os.path.exists(weights_df_filepath):
            print('Dataframe de pesos ja construido...')
            weights_df = pd.read_pickle(weights_df_filepath)
        else:
            # Normaliza timestamps        
            df[rr.COLUMN_WEIGHTS] = MinMaxScaler((0, 1)).fit_transform(df[[rr.COLUMN_TIMESTAMP]])
            # Funcao de combinacao par-a-par entre itens e pesos
            def item_weight_diff(x):
                return pd.concat([
                    pd.DataFrame(itertools.combinations(x[rr.COLUMN_ITEM_ID], 2), columns=['Item1', 'Item2']), 
                    pd.DataFrame(itertools.combinations(x[rr.COLUMN_WEIGHTS], 2), columns=['Weight1', 'Weight2'])
                ], axis=1)        
            weights_df = df.groupby(rr.COLUMN_USER_ID).apply(item_weight_diff).reset_index(level=1, drop=True)
            # Calcula a diferenca entre pesos
            weights_df['Weight'] = 1.0 - (weights_df['Weight1'] - weights_df['Weight2']).abs()
            # Tira a media
            weights_df = weights_df.groupby(['Item1', 'Item2'])['Weight'].mean().reset_index()
            # Troca o nome
            weights_df['Item1'] = sparse_repr.get_idx_of_item(weights_df['Item1'].values)
            weights_df['Item2'] = sparse_repr.get_idx_of_item(weights_df['Item2'].values)
            # Salva o df
            os.makedirs('weights_dfs', exist_ok=True)
            weights_df.to_pickle(weights_df_filepath)        
                
        print('Gerando embeddings do Item2Vec...')            
        model = Word2Vec(
            corpus_file=interactions_file,
            vector_size=embedding_dim,
            window=sparse_matrix.sum(axis=1).max()*10000,
            min_count=1,
            workers=cpu_count(),
            sg=1,
            hs=0,
            negative=negative_sampling,
            ns_exponent=negative_exponent,
            sample=0 if subsampling_p is None else subsampling_p,
            max_vocab_size=None,
            max_final_vocab=None,
            epochs=n_epochs,
            trim_rule=None,
            sorted_vocab=0,
            batch_words=batch_size,
            compute_loss=False,
            seed=rr.RANDOM_SEED,
            weights_df=weights_df, 
            default_positive_weight=posw,
            default_negative_weight=negw
        )
        embeddings = model.wv.vectors[np.argsort(np.fromiter(model.wv.index_to_key, dtype=np.int32, count=len(model.wv.index_to_key)))]
        os.remove(interactions_file)
       
        os.makedirs(embeddings_dir, exist_ok=True)
        pickle.dump(sparse_repr, open(sparse_repr_filepath, 'wb'))
        pickle.dump(embeddings, open(item_embeddings_filepath, 'wb'))
        print('Embeddings do Item2Vec geradas!')




class Item2VecSequencial(object):
    def generate_embeddings(self, df, embeddings_dir, embeddings_filename, embedding_dim=100, n_epochs=25, negative_sampling=15, negative_exponent=0.75, subsampling_p=None, regularization_lambda=None, batch_size=10000, window_size=3, cutoff_interval=None):
        # Verifica se embeddings ja foram criadas previamente
        sparse_repr_filepath = os.path.join(embeddings_dir, rr.FILE_SPARSE_REPR.format(embeddings_filename))
        item_embeddings_filepath = os.path.join(embeddings_dir, rr.FILE_ITEM_EMBEDDINGS.format(embeddings_filename))
        if os.path.exists(sparse_repr_filepath) and os.path.exists(item_embeddings_filepath):        
            print('Embeddings já criadas...')
            return

        sparse_repr = SparseRepr(df)
        sorted_df = df.sort_values([rr.COLUMN_USER_ID,rr.COLUMN_TIMESTAMP],ignore_index=True).copy()

        if cutoff_interval is not None:
            #diferenca de dias
            sorted_df['DATES_BETWEEN']= np.append(np.diff(sorted_df[rr.COLUMN_DATETIME])/np.timedelta64(1,'D'),0)
            #coluna com true ou false (mesmo user ou nao)
            sorted_df['DIFF_USER'] = (sorted_df[rr.COLUMN_USER_ID]==sorted_df[rr.COLUMN_USER_ID].shift(-1))
            #usuarios temporarios
            cutoff = sorted_df[(sorted_df['DIFF_USER']==False) | (sorted_df['DATES_BETWEEN']>=cutoff_interval)].index
            sorted_df['TEMP_USERS'] = 0
            for c in cutoff:
                sorted_df.loc[c+1:, 'TEMP_USERS'] = sorted_df.loc[c+1:, 'TEMP_USERS']+1
        else:
            sorted_df['TEMP_USERS'] = sorted_df[rr.COLUMN_USER_ID]

        print('Gerando arquivo de interações...')        
        fid = 0
        interactions_file = 'item2vec_temporal_interactions_{}.temp'.format(fid)
        while os.path.exists(interactions_file):
            fid += 1
            interactions_file = 'item2vec_temporal_interactions_{}.temp'.format(fid)        
        with open(interactions_file, 'w') as f:
            #for user in sparse_repr.get_user_of_idx(np.arange(sparse_repr.get_n_users())):    
            for user in sorted_df['TEMP_USERS'].unique():  
                #user_interactions = sorted_df[sorted_df[rr.COLUMN_USER_ID]==user]
                user_interactions = sorted_df[sorted_df['TEMP_USERS']==user]
                user_items = sparse_repr.get_idx_of_item(user_interactions[rr.COLUMN_ITEM_ID].values).astype(str)              
                f.write(' '.join(user_items) + '\n')
        
        
        # Gera embeddings
        print('Gerando embeddings do Item2Vec...')            
        model = Word2Vec(
            corpus_file=interactions_file,
            size=embedding_dim,
            window=window_size,
            min_count=1,
            workers=1,
            sg=1,
            hs=0,
            negative=negative_sampling,
            ns_exponent=negative_exponent,
            sample=0 if subsampling_p is None else subsampling_p,
            max_vocab_size=None,
            max_final_vocab=None,            
            iter=n_epochs,
            trim_rule=None,
            sorted_vocab=0,
            batch_words=batch_size,
            compute_loss=False,
            seed=rr.RANDOM_SEED
        )
        embeddings = model.wv.vectors[np.argsort(np.fromiter(model.wv.index2word, dtype=np.int32, count=len(model.wv.index2word)))]
        os.remove(interactions_file)
       
        os.makedirs(embeddings_dir, exist_ok=True)
        pickle.dump(sparse_repr, open(sparse_repr_filepath, 'wb'))
        pickle.dump(embeddings, open(item_embeddings_filepath, 'wb'))
        print('Embeddings do Item2Vec geradas!')


class Item2VecPonderado(object):
    def __init__(self):
        self.rng = np.random.RandomState(seed=rr.RANDOM_SEED)

    def _create_model(self, catalog_size, embedding_dim, regularization_lambda):
        target_item = Input(shape=(1,), name='target_item')
        context_item = Input(shape=(1,), name='context_item')
        regularizer = None if regularization_lambda is None else regularizers.l2(regularization_lambda)
        embedding_lookup = Embedding(catalog_size, embedding_dim, input_length=1, name='embedding', embeddings_regularizer=regularizer, embeddings_initializer=initializers.RandomUniform(seed=rr.RANDOM_SEED))
        embedding_target = embedding_lookup(target_item)
        embedding_context = embedding_lookup(context_item)
        merged_vector = dot([embedding_target, embedding_context], axes=-1)
        reshaped_vector = Reshape((1,), input_shape=(1,1))(merged_vector)
        prediction = Activation('sigmoid')(reshaped_vector)
        model = Model(inputs=[target_item, context_item], outputs=prediction)
        model.compile(optimizer=Adam(learning_rate=0.025), loss='binary_crossentropy')
        return model


    def _make_cum_table(self, sparse_matrix, n_interactions, negative_exponent):        
        item_neg_probs = np.power(sparse_matrix.sum(axis=0).A1, negative_exponent) / np.power(n_interactions, negative_exponent)        
        item_neg_probs[item_neg_probs==np.inf] = 0
        valid_min = item_neg_probs[np.where(item_neg_probs>0)].min()
        item_count = np.round(item_neg_probs * (1 / valid_min))
        cum_table = np.cumsum(item_count)
        return cum_table
    

    def _get_negative_items(self, user_items, cum_table, negative_sampling):
        negative_items = set(user_items)
        while len(negative_items) < len(user_items) + negative_sampling:
            negative_items.add(np.searchsorted(cum_table, self.rng.randint(1, cum_table[-1]+1)))
        return negative_items - set(user_items)

    def _calculate_sample_weights(self, sampled_df):
        sampled_df[rr.COLUMN_WEIGHTS]=0
        mms = MinMaxScaler((1,2))
        sampled_df[rr.COLUMN_WEIGHTS] = mms.fit_transform(sampled_df[[rr.COLUMN_TIMESTAMP]])
        return sampled_df

    #lista a mais para sample_weight
    def _generate_samples(self, sparse_matrix, cum_table, negative_sampling, batch_size, negative_weight=1):
        while True:
            X_target, X_context, y = list(), list(), list()
            for user in range(sparse_matrix.shape[0]):
                user_items = sparse_matrix[user].nonzero()[1]
                num_user_items = len(user_items)
                
                # Amostras positivas
                X_target.extend(np.repeat(user_items, num_user_items-1))
                X_context.extend(np.tile(user_items, num_user_items)[np.tile(np.arange(1, num_user_items+1), num_user_items-1) + np.repeat(np.arange(num_user_items-1)*(num_user_items+1), num_user_items)])
                y.extend(np.ones(num_user_items * (num_user_items-1)))

                # Amostras negativas
                X_target.extend(np.repeat(user_items, negative_sampling))
                for _ in user_items:
                    X_context.extend(self._get_negative_items(user_items, cum_table, negative_sampling))
                y.extend(np.zeros(len(user_items)*negative_sampling))                

                # Retorno dos batches
                num_batches = int(len(X_target)/batch_size)
                if num_batches > 0:
                    # Peso das amostras abordagem 1 (pega o peso determinado para o X_target da matriz esparsa)                
                    weights_target = sparse_matrix[user, X_target].todense().A1
                    weights_context = sparse_matrix[user, X_context].todense().A1
                    sample_weights = 2 - np.absolute(weights_target - weights_context)
                    #tratamento para as amostras negativas
                    sample_weights[np.where(weights_context==0)]= negative_weight
                    # Retorna os batches
                    for i in range(0, num_batches*batch_size, batch_size):                        
                        yield [np.array(X_target[i:i+batch_size]), np.array(X_context[i:i+batch_size])], np.array(y[i:i+batch_size]), np.array(sample_weights[i:i+batch_size])
                    # Atualiza o conteudo
                    X_target = X_target[num_batches*batch_size:]
                    X_context = X_context[num_batches*batch_size:]
                    y = y[num_batches*batch_size:]                    
            
            # Retorna o que sobrar
            if len(X_target) > 0:
                weights_target = sparse_matrix[user, X_target].todense().A1
                weights_context = sparse_matrix[user, X_context].todense().A1
                sample_weights = 2 - np.absolute(weights_target - weights_context)
                yield [np.array(X_target), np.array(X_context)], np.array(y), np.array(sample_weights)


    def _subsample_items(self, df, subsampling_p):
        if subsampling_p is not None:
            freq = df.groupby(rr.COLUMN_ITEM_ID).size()
            n_interactions = len(df)
            z = freq / n_interactions
            discard_prob = (np.sqrt(z/subsampling_p) + 1) * (subsampling_p/z)
            discard_prob = discard_prob.reindex(df[rr.COLUMN_ITEM_ID])
            discard_prob.index = df.index
            discarded_interactions = self.rng.rand(n_interactions) < discard_prob
            return df[~discarded_interactions].copy()
        return df.copy()

    def generate_embeddings(self, df, embeddings_dir, embeddings_filename, embedding_dim=100, n_epochs=5, negative_sampling=5, negative_exponent=0.75, subsampling_p=1e-3, regularization_lambda=None, batch_size=2**14, negative_weight=1):
        # Verifica se embeddings ja foram criadas previamente
        sparse_repr_filepath = os.path.join(embeddings_dir, rr.FILE_SPARSE_REPR.format(embeddings_filename))
        item_embeddings_filepath = os.path.join(embeddings_dir, rr.FILE_ITEM_EMBEDDINGS.format(embeddings_filename))
        if os.path.exists(sparse_repr_filepath) and os.path.exists(item_embeddings_filepath):        
            print('Embeddings já criadas...')
            return

        print('Montando representacao esparsa...')
        sparse_repr = SparseRepr(df)

        print('Realizando subamostragem de itens frequentes...')
        sampled_df = self._subsample_items(df, subsampling_p)
        
        print('Calculando pesos das amostras...')
        sampled_df = self._calculate_sample_weights(sampled_df)

        print('Recuperando informacoes da base...')
        catalog_size = sparse_repr.get_n_items()
        sparse_matrix = sparse_repr.get_matrix(sampled_df[rr.COLUMN_USER_ID], sampled_df[rr.COLUMN_ITEM_ID], sampled_df[rr.COLUMN_WEIGHTS])
        n_interactions = len(sampled_df)
        n_samples = n_interactions * negative_sampling + scipy.special.comb(sampled_df.groupby(rr.COLUMN_USER_ID).size().values, 2).sum()
        
        print('Gerando modelo...')
        model = self._create_model(catalog_size, embedding_dim, regularization_lambda)

        print('Gerando tabela cumulativa...')
        cum_table = self._make_cum_table(sparse_matrix, n_interactions, negative_exponent)        

        print('Gerando gerador de amostras...')
        samples_generator = self._generate_samples(sparse_matrix, cum_table, negative_sampling, batch_size, negative_weight)
        
        print('Gerando embeddings do Item2Vec...')
        model.fit_generator(samples_generator, steps_per_epoch=np.ceil(n_samples/batch_size), epochs=n_epochs, verbose=1)        
        item_embeddings = model.get_weights()[0]
        
        print('Salvando embeddings do Item2Vec...')
        os.makedirs(embeddings_dir, exist_ok=True)
        pickle.dump(sparse_repr, open(sparse_repr_filepath, 'wb'))
        pickle.dump(item_embeddings, open(item_embeddings_filepath, 'wb'))
        print('Embeddings do Item2Vec geradas!')