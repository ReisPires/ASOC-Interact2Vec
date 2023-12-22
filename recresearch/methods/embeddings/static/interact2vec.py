from gensim.models import Doc2Vec
from keras import Model, regularizers, initializers
from keras.layers import Input, Embedding, Activation, dot, Reshape
from keras.optimizers import Adam, SGD
from multiprocessing import cpu_count
import numpy as np
import pickle
import recresearch as rr
from recresearch.dataset import SparseRepr

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf

class Interact2VecGensim(object):
    
    def generate_embeddings(self, df, embeddings_dir, embeddings_filename, embedding_dim=100, n_epochs=5, negative_sampling=5, negative_exponent=0.75, subsampling_p=1e-3, regularization_lambda=None, batch_size=2**14):
        sparse_repr = SparseRepr(df)
        sparse_matrix = sparse_repr.get_matrix(df[rr.COLUMN_USER_ID], df[rr.COLUMN_ITEM_ID])
        
        print('Gerando arquivo de interações...')
        fid = 0
        interactions_file = 'interact2vec_interactions_{}.temp'.format(fid)
        while os.path.exists(interactions_file):
            fid += 1
            interactions_file = 'interact2vec_interactions_{}.temp'.format(fid)        
        with open(interactions_file, 'w') as f:
            for user in range(sparse_matrix.shape[0]):
                f.write(' '.join(sparse_matrix[user].nonzero()[1].astype(str)) + '\n')
        
        print('Gerando embeddings do Interact2Vec...')
        model = Doc2Vec(
            corpus_file=interactions_file,
            dm=0,
            vector_size=embedding_dim,
            window=sparse_matrix.sum(axis=1).max()*10000,
            min_count=1,            
            max_vocab_size=None,
            sample=0 if subsampling_p is None else subsampling_p,
            workers=cpu_count(),
            epochs=n_epochs,
            hs=0,
            negative=negative_sampling,
            ns_exponent=negative_exponent,
            dbow_words=1,
            trim_rule=None,
            seed=rr.RANDOM_SEED
        )
        items_embeddings = model.wv.vectors[np.argsort(np.fromiter(model.wv.index2word, dtype=np.int32, count=len(model.wv.index2word)))]        
        users_embeddings = model.docvecs.vectors_docs                
        os.remove(interactions_file)

        os.makedirs(embeddings_dir, exist_ok=True)
        pickle.dump(sparse_repr, open(os.path.join(embeddings_dir, rr.FILE_SPARSE_REPR.format(embeddings_filename)), 'wb'))
        pickle.dump(users_embeddings, open(os.path.join(embeddings_dir, rr.FILE_USER_EMBEDDINGS.format(embeddings_filename)), 'wb'))
        pickle.dump(items_embeddings, open(os.path.join(embeddings_dir, rr.FILE_ITEM_EMBEDDINGS.format(embeddings_filename)), 'wb'))
        print('Embeddings do Interact2Vec geradas!')



class Interact2VecKeras(object):
    def __init__(self):
        self.rng = np.random.RandomState(seed=rr.RANDOM_SEED)
    
    def _create_model(self, users_size, catalog_size, embedding_dim, learning_rate, regularization_lambda):
        target_user = Input(shape=(1,), name='target_user')
        context_item = Input(shape=(1,), name='context_item')
        regularizer = None if regularization_lambda is None else regularizers.l2(regularization_lambda)
        user_embedding_lookup = Embedding(users_size, embedding_dim, input_length=1, name='user_embedding', embeddings_regularizer=regularizer, embeddings_initializer=initializers.RandomUniform(seed=rr.RANDOM_SEED))
        item_embedding_lookup = Embedding(catalog_size, embedding_dim, input_length=1, name='item_embedding', embeddings_regularizer=regularizer, embeddings_initializer=initializers.RandomUniform(seed=rr.RANDOM_SEED))
        embedding_user = user_embedding_lookup(target_user)
        embedding_item = item_embedding_lookup(context_item)
        merged_vector = dot([embedding_user, embedding_item], axes=-1)
        reshaped_vector = Reshape((1,), input_shape=(1,1))(merged_vector)
        prediction = Activation('sigmoid')(reshaped_vector)
        model = Model(inputs=[target_user, context_item], outputs=prediction)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy')
        return model


    def _make_cum_table(self, sparse_matrix, negative_exponent):
        item_freq = sparse_matrix.sum(axis=0).A1
        item_neg_probs = np.nan_to_num(np.power(item_freq, negative_exponent), posinf=0, neginf=0)
        cum_table = np.round((np.cumsum(item_neg_probs) / np.sum(item_neg_probs)) * np.sum(item_freq))
        return cum_table
    

    def _get_negative_items(self, user_item, cum_table, negative_sampling):
        negative_items = list()
        while len(negative_items) < negative_sampling:            
            neg_sample = np.searchsorted(cum_table, self.rng.randint(1, cum_table[-1]+1))
            if neg_sample != user_item:
                negative_items.append(neg_sample)
        return negative_items


    def _generate_samples(self, sparse_matrix, cum_table, negative_sampling, batch_size):
        while True:
            X_user, X_item, y = list(), list(), list()
            for user in range(sparse_matrix.shape[0]):
                user_items = sparse_matrix[user].nonzero()[1]
                num_user_items = len(user_items)
                if num_user_items == 0:
                    continue
                
                # Amostras positivas
                X_user.extend(np.repeat(user, num_user_items))
                X_item.extend(user_items)
                y.extend(np.ones(num_user_items))
                
                # Amostras negativas
                X_user.extend(np.repeat(user, num_user_items*negative_sampling))
                for user_item in user_items:
                    X_item.extend(self._get_negative_items(user_item, cum_table, negative_sampling))
                y.extend(np.zeros(num_user_items*negative_sampling))

                # Retorno dos batches
                num_batches = int(len(X_user)/batch_size)
                if num_batches > 0:
                    for i in range(0, num_batches*batch_size, batch_size):
                        yield [np.array(X_user[i:i+batch_size]), np.array(X_item[i:i+batch_size])], np.array(y[i:i+batch_size])
                    X_user = X_user[num_batches*batch_size:]
                    X_item = X_item[num_batches*batch_size:]
                    y = y[num_batches*batch_size:]
            
            # Retorna o que sobrar
            if len(X_user) > 0:
                yield [np.array(X_user), np.array(X_item)], np.array(y)


    def _subsample_items(self, df, subsampling_p):
        if subsampling_p is not None:
            freq = df.groupby(rr.COLUMN_ITEM_ID).size()
            n_interactions = len(df)
            z = freq / n_interactions
            keep_prob = (np.sqrt(z/subsampling_p) + 1) * (subsampling_p/z)
            keep_prob = keep_prob.reindex(df[rr.COLUMN_ITEM_ID])
            keep_prob.index = df.index
            discarded_interactions = keep_prob < self.rng.rand(n_interactions)
            return df[~discarded_interactions].copy()
        return df.copy()


    def _save_embeddings(self, model, sparse_repr, embeddings_dir, sparse_repr_filepath, user_embeddings_filepath, item_embeddings_filepath):
        print('Salvando embeddings do Interact2Vec...')
        user_embeddings = model.get_weights()[0]
        item_embeddings = model.get_weights()[1]
        os.makedirs(embeddings_dir, exist_ok=True)
        pickle.dump(sparse_repr, open(sparse_repr_filepath, 'wb'))
        pickle.dump(user_embeddings, open(user_embeddings_filepath, 'wb'))
        pickle.dump(item_embeddings, open(item_embeddings_filepath, 'wb'))
        print('Embeddings do Interact2Vec geradas!')


    def generate_embeddings(self, df, embeddings_dir, embeddings_filename, embedding_dim=100, n_epochs=5, negative_sampling=5, negative_exponent=0.75, subsampling_p=1e-3, learning_rate=0.025, regularization_lambda=None, batch_size=2**13, save_embeddings=True, verbose=True):
        # Verifica se embeddings ja foram criadas previamente
        sparse_repr_filepath = os.path.join(embeddings_dir, rr.FILE_SPARSE_REPR.format(embeddings_filename))
        user_embeddings_filepath = os.path.join(embeddings_dir, rr.FILE_USER_EMBEDDINGS.format(embeddings_filename))
        item_embeddings_filepath = os.path.join(embeddings_dir, rr.FILE_ITEM_EMBEDDINGS.format(embeddings_filename))
        if os.path.exists(sparse_repr_filepath) and os.path.exists(user_embeddings_filepath) and os.path.exists(item_embeddings_filepath):
            if verbose:
                print('Embeddings já criadas...')            
            return
        
        if verbose:
            print('Montando representacao esparsa...')
        sparse_repr = SparseRepr(df)
        
        if verbose:
            print('Realizando subamostragem de itens frequentes...')
        sampled_df = self._subsample_items(df, subsampling_p)        

        if verbose:
            print('Recuperando informacoes da base...')        
        users_size = sparse_repr.get_n_users()
        catalog_size = sparse_repr.get_n_items()
        sparse_matrix = sparse_repr.get_matrix(sampled_df[rr.COLUMN_USER_ID], sampled_df[rr.COLUMN_ITEM_ID])
        n_interactions = len(sampled_df)
        n_samples = n_interactions * negative_sampling + n_interactions        

        if verbose:          
            print('Gerando modelo...')
        model = self._create_model(users_size, catalog_size, embedding_dim, learning_rate, regularization_lambda)        
        if n_interactions == 0 or sampled_df[rr.COLUMN_ITEM_ID].nunique() <= 1:
            if verbose:
                print('Nao ha interacoes suficientes para treinamento...')
            if save_embeddings:
                self._save_embeddings(model, sparse_repr, embeddings_dir, sparse_repr_filepath, user_embeddings_filepath, item_embeddings_filepath)
            return

        if verbose:
            print('Gerando tabela cumulativa...')        
        cum_table = self._make_cum_table(sparse_matrix, negative_exponent)        

        if verbose:
            print('Gerando gerador de amostras...')
        samples_generator = self._generate_samples(sparse_matrix, cum_table, negative_sampling, batch_size)
        
        if verbose:
            print('Gerando embeddings do Interact2Vec...')
        model.fit(samples_generator, steps_per_epoch=np.ceil(n_samples/batch_size), epochs=n_epochs, verbose=0)
        
        if save_embeddings:
            self._save_embeddings(model, sparse_repr, embeddings_dir, sparse_repr_filepath, user_embeddings_filepath, item_embeddings_filepath)
