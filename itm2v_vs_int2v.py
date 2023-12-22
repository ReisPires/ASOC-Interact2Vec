from datetime import datetime
import time

import recresearch as rr
from recresearch.dataset import get_dataset
from recresearch.experiments.logger import BasicLogger
from recresearch.utils.preprocessing import cut_by_minimal_interactions
from recresearch.methods.embeddings.static import Item2VecKeras, Interact2VecKeras

# Define parametros das embeddings
EMBEDDING_DIM = 100
N_EPOCHS = 5
NEGATIVE_SAMPLING = 5
NEGATIVE_EXPONENT = 0.75
SUBSAMPLING_P = 1e-3
LEARNING_RATE = 0.025
REGULARIZATION_LAMBDA = None
BATCH_SIZE = 2**8

# Inicializa o logger
results_logger = BasicLogger('i2v_duration_{}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

# Le a base de dados
dataset = get_dataset('Filmtrust', ds_dir='datasets', ds_type='I', temporal_behaviour='S')
df_full = dataset.get_dataframe()

# Itera entre os multiplos oversamplings
for sampling_rate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    # Gera dataset reduzido    
    n_interactions = int(len(df_full) * sampling_rate)
    df_duration = cut_by_minimal_interactions(df_full.sample(n_interactions, random_state=rr.RANDOM_SEED).copy(), min_interactions=2)

    # Treina cada modelo
    for model_name, Model in [('Item2Vec', Item2VecKeras), ('Interact2Vec', Interact2VecKeras)]:
        model = Model()
        
        # Cronometra o tempo
        star_time = time.time()
        model.generate_embeddings(
            df_duration, 'i2v_duration_embeddings', 'temp_emb', save_embeddings=False, verbose=False,
            embedding_dim=EMBEDDING_DIM, n_epochs=N_EPOCHS, negative_sampling=NEGATIVE_SAMPLING, negative_exponent=NEGATIVE_EXPONENT, 
            subsampling_p=SUBSAMPLING_P, learning_rate=LEARNING_RATE, regularization_lambda=REGULARIZATION_LAMBDA, batch_size=BATCH_SIZE
        )
        end_time = time.time()

        # Salva o tempo
        elapsed_time = end_time-star_time
        results_logger.log('{};{};{};{}'.format(sampling_rate, n_interactions, model_name, elapsed_time))

print('OK')