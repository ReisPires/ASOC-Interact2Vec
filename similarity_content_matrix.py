import numpy as np
import pickle as pk
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
import recresearch as rr
import os

os.makedirs('similarity_content_matrix', exist_ok=True)

QTD_EX = 3
QTD_VIZ = 5

lastfm = {'embeddings':'./content_matrices/Last.FM - Listened_',
            'dataset':'./datasets/Last.FM - Listened/'}

caminhos_tag = [lastfm]

datasets_tag = ['Last.FM - Listened']

exemplos_lastfm = {'Linkin Park': 377,
                    'Shakira': 701,
                    'ACDC': 706,
                    'Johnny Cash': 718,
                    'One Direction': 5752,
                    'Bob Dylan': 212,
                    'Eminem': 475,
                    'Elvis Presley': 1244,
                    'Arctic Monkeys': 207,
                    'The Beatles': 227}

exemplos_tag = [exemplos_lastfm]

print("Calculando as similaridades para as bases com tag... ")
for caminhos, dataset, exemplos in  zip(caminhos_tag, datasets_tag, exemplos_tag):
    print('\tAbrindo arquivo de items e tags da base ' + dataset + '...')
    os.makedirs('similarity_content_matrix/' + dataset, exist_ok=True)
    items = np.array(pd.read_csv(caminhos['dataset'] + '/items.csv', sep = rr.DELIMITER, quotechar= rr.QUOTECHAR, quoting= rr.QUOTING, encoding=rr.ENCODING))
    tags = np.array(pd.read_csv(caminhos['dataset'] + '/tags.csv', sep = rr.DELIMITER, quotechar= rr.QUOTECHAR, quoting= rr.QUOTING, encoding=rr.ENCODING))
    print("\t\tCalculando as similaridades para a matriz " + '...')
    matrix = pk.load(open(caminhos['embeddings'] + 'matrix.pkl', 'rb'))
    sparse = pk.load(open(caminhos['embeddings'] + 'encoder.pkl', 'rb'))
    resultado = pd.DataFrame([], columns = ['Label', 'Nome', 'Tags'])

    print('\t\t\tCalculando...')
    sims = cosine_distances(matrix)
    np.fill_diagonal(sims, np.inf)

    print("\t\t\tOrdenando...")
    sims_ord = np.argsort(sims, axis=1)[:, 1:QTD_VIZ + 1]

    print('\t\t\tEscrevendo...')
    for alvo_name in exemplos:
        alvo_id = exemplos[alvo_name]
        alvo_sims = sims_ord[sparse.transform([alvo_id])]
        
        pos_item = np.where(items[:, 0] == alvo_id)[0][0]
        name = items[pos_item][1]
        label = 'Alvo'
        name_tag = ''
        if isinstance(items[pos_item][3], str):
            tok_tag = items[pos_item][3].split('/')
            for tg in tok_tag:
                tag = tg.split()
                pos_tag = np.where(tags[:, 0] == int(tag[0]))[0][0]
                name_tag += tags[pos_tag][1] + " " + tag[1] + "/ "
        df_aux = pd.DataFrame(np.array([[label, name, name_tag]]), columns = ['Label', 'Nome', 'Tags'])
        resultado = pd.concat([resultado, df_aux])
        for label, viz in enumerate(alvo_sims[0]):
            idx_item = sparse.inverse_transform([viz])
            pos_item = np.where(items[:, 0] == idx_item)[0][0]
            name = items[pos_item][1]
            name_tag = ''
            if isinstance(items[pos_item][3], str):
                tok_tag = items[pos_item][3].split('/')
                for tg in tok_tag:
                    tag = tg.split()
                    pos_tag = np.where(tags[:, 0] == int(tag[0]))[0][0]
                    name_tag += tags[pos_tag][1] + " " + tag[1] + "/ "
            df_aux = pd.DataFrame(np.array([[label+1, name, name_tag]]), columns = ['Label', 'Nome', 'Tags'])
            resultado = pd.concat([resultado, df_aux])
        df_aux = pd.DataFrame(np.array([[np.NaN, np.NaN, np.NaN]]), columns = ['Label', 'Nome', 'Tags'])
        resultado = pd.concat([resultado, df_aux])
    resultado.to_csv('similarity_content_matrix/' + dataset + "/similarity_" + dataset + '_content_matrix' + '.csv', index=False)
    print('\t\t\tConcluído!')

print('Similaridades calculadas com sucesso!')