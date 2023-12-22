import numpy as np
import pickle as pk
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
import recresearch as rr
import os

os.makedirs('similarity_qualitative', exist_ok=True)

QTD_EX = 3
QTD_VIZ = 5

movielens = {'embeddings':'./embeddings/MovieLens_',
            'dataset':'./datasets/MovieLens/'}

lastfm = {'embeddings':'./embeddings/Last.FM - Listened_',
            'dataset':'./datasets/Last.FM - Listened/'}

caminhos_tag = [lastfm]

caminhos_cat = [movielens]

datasets_tag = ['Last.FM - Listened']

datasets_cat = ['MovieLens']

metodos = ['ALSImplicit',
            'Interact2Vec',
            'Item2Vec']

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

exemplos_movielens = {'Toy Story': 1,
                        'Grease': 1380,
                        'X-Men': 3793,
                        'Titanic': 1721,
                        'Friday The 13th': 1974,
                        'Fight Club': 2959,
                        '10 Things I Hate About You': 2572,
                        'Citizen Kane': 923,
                        'The Godfather': 845,
                        'Lord of the Rings - The Fellowship of the Ring': 4993}

exemplos_tag = [exemplos_lastfm]
exemplos_cat = [exemplos_movielens]

print("Calculando as similaridades para as bases com tag... ")
for caminhos, dataset, exemplos in  zip(caminhos_tag, datasets_tag, exemplos_tag):
    print('\tAbrindo arquivo de items e tags da base ' + dataset + '...')
    os.makedirs('similarity_qualitative/' + dataset, exist_ok=True)
    items = np.array(pd.read_csv(caminhos['dataset'] + '/items.csv', sep = rr.DELIMITER, quotechar= rr.QUOTECHAR, quoting= rr.QUOTING, encoding=rr.ENCODING))
    tags = np.array(pd.read_csv(caminhos['dataset'] + '/tags.csv', sep = rr.DELIMITER, quotechar= rr.QUOTECHAR, quoting= rr.QUOTING, encoding=rr.ENCODING))
    for metodo in metodos:
        print("\t\tCalculando as similaridades para as embeddings " + metodo + '...')
        embeddings = pk.load(open(caminhos['embeddings'] + metodo + '_item_embeddings.pkl', 'rb'))
        sparse = pk.load(open(caminhos['embeddings'] + metodo + '_sparse_repr.pkl', 'rb'))
        resultado = pd.DataFrame([], columns = ['Label', 'Nome', 'Tags'])

        print('\t\t\tCalculando...')
        sims = cosine_distances(embeddings)
        np.fill_diagonal(sims, np.inf)

        print("\t\t\tOrdenando...")
        sims_ord = np.argsort(sims, axis=1)[:, 1:QTD_VIZ + 1]

        print('\t\t\tEscrevendo...')
        for alvo_name in exemplos:
            alvo_id = exemplos[alvo_name]
            alvo_sims = sims_ord[sparse.get_idx_of_item(alvo_id)]
        
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
            for label, viz in enumerate(alvo_sims):
                idx_item = sparse.get_item_of_idx(viz)
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
        resultado.to_csv('similarity_qualitative/' + dataset + "/similarity_" + dataset + '_' + metodo + '.csv', index=False)
        print('\t\t\tConcluído!')

print("Calculando as similaridades para as bases com categoria... ")
for caminhos, dataset, exemplos in  zip(caminhos_cat, datasets_cat, exemplos_cat):
    print('\tAbrindo arquivo de items da base ' + dataset + '...')
    os.makedirs('similarity_qualitative/' + dataset, exist_ok=True)
    items = np.array(pd.read_csv(caminhos['dataset'] + '/items.csv', sep = rr.DELIMITER, quotechar= rr.QUOTECHAR, quoting= rr.QUOTING, encoding=rr.ENCODING))
    for metodo in metodos:
        print("\t\tCalculando as similaridades para as embeddings " + metodo + '...')
        embeddings = pk.load(open(caminhos['embeddings'] + metodo + '_item_embeddings.pkl', 'rb'))
        sparse = pk.load(open(caminhos['embeddings'] + metodo + '_sparse_repr.pkl', 'rb'))
        resultado = pd.DataFrame([], columns = ['Label', 'Nome', 'Categorias', 'Tags'])

        print('\t\t\tCalculando...')
        sims = cosine_distances(embeddings)
        np.fill_diagonal(sims, np.inf)

        print("\t\t\tOrdenando...")
        sims_ord = np.argsort(sims, axis=1)[:, 1:QTD_VIZ + 1]

        print('\t\t\tEscrevendo...')
        for alvo_name in exemplos:
            alvo_id = exemplos[alvo_name]
            alvo_sims = sims_ord[sparse.get_idx_of_item(alvo_id)]
        
            pos_item = np.where(items[:, 0] == alvo_id)[0][0]
            name = items[pos_item][1]
            cats = items[pos_item][2]
            tags = items[pos_item][3]
            label = 'Alvo'
            df_aux = pd.DataFrame(np.array([[label, name, cats, tags]]), columns = ['Label', 'Nome', 'Categorias', 'Tags'])
            resultado = pd.concat([resultado, df_aux])
            for label, viz in enumerate(alvo_sims):
                idx_item = sparse.get_item_of_idx(viz)
                pos_item = np.where(items[:, 0] == idx_item)[0][0]
                name = items[pos_item][1]
                cats = items[pos_item][2]
                tags = items[pos_item][3]
                df_aux = pd.DataFrame(np.array([[label+1, name, cats, tags]]), columns = ['Label', 'Nome', 'Categorias', 'Tags'])
                resultado = pd.concat([resultado, df_aux])
            df_aux = pd.DataFrame(np.array([[np.NaN, np.NaN, np.NaN, np.NaN]]), columns = ['Label', 'Nome', 'Categorias', 'Tags'])
            resultado = pd.concat([resultado, df_aux])
        resultado.to_csv('similarity_qualitative/' + dataset + "/similarity_" + dataset + '_' + metodo + '.csv', index=False)
        print('\t\t\tConcluído!')


print('Similaridades calculadas com sucesso!')

            
        


