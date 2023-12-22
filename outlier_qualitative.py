import pickle as pk
import pandas as pd
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from recresearch.evaluation.outlier_DeliciousBookmarks import OutlierDeliciousBookmarks
from recresearch.evaluation.outlier_LastFM import OutlierLastFM
from recresearch.evaluation.outlier_MovieLens import OutlierMovieLens
from recresearch.evaluation.outlier_AnimeRecommendations import OutlierAnimeRecommendations
import recresearch as rr
import os

os.makedirs('outlier_qualitative', exist_ok=True)

N_SAMPLES_RAN = 10

out_deliciousbookmarks = OutlierDeliciousBookmarks()
out_lastfm = OutlierLastFM()
out_movielens = OutlierMovieLens()
out_animerecommendations = OutlierAnimeRecommendations()

out_tag = [out_deliciousbookmarks, out_lastfm]

out_cat = [out_animerecommendations, out_movielens]

movielens = {'embeddings':'./embeddings/full/MovieLens_',
            'dataset':'./datasets/MovieLens/'}

lastfm = {'embeddings':'./embeddings/full/Last.FM - Listened_',
            'dataset':'./datasets/Last.FM - Listened/'}

deliciousbookmarks = {'embeddings':'./embeddings/full/DeliciousBookmarks_',
                        'dataset':'./datasets/DeliciousBookmarks/'}

animerecommendations = {'embeddings':'./embeddings/full/Anime Recommendations_',
                        'dataset':'./datasets/Anime Recommendations/'}            

caminhos_tag = [deliciousbookmarks, lastfm]

caminhos_cat = [animerecommendations, movielens]

datasets_tag = ['DeliciousBookmarks',
                'Last.FM - Listened']

datasets_cat = ['Anime Recommendations',
                'MovieLens']

metodos = ['ALSImplicit',
            'BPRImplicit',
            'User2Vec',
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

exemplos_animerecommendations = {'Fullmetal Alchemist: Brotherhood':5114,
                                'Hunter x Hunter (2011)':11061,
                                'Hajime no Ippo': 263,
                                'One Punch Man': 30276,
                                'Cowboy Bebop': 1,
                                'Monster': 19,
                                'Naruto': 20,
                                'Dragon Ball Z': 813,
                                'Haikyuu!!':20583,
                                'Non Non Biyori': 17549}

exemplos_deliciousbookmarks = {'Second Life: Your World. Your Imagination.':37,
                                'Crisis Forum - Forum for the Study of Crisis in the 21st Century': 810,
                                'IstoÉ': 972,
                                'My Home Library': 25,
                                'About Google Scholar': 55956,
                                'Adobe': 67333,
                                'Windows': 67340,
                                'eBooksBay': 90193,
                                'Google I/O 2010': 81512,
                                'Graffiti Analysis': 50555}

exemplos_tag = [exemplos_deliciousbookmarks, exemplos_lastfm]
exemplos_cat = [exemplos_animerecommendations, exemplos_movielens]

print("Criando teste para bases com tags...")
for dt, caminhos, exemplos, out in zip(datasets_tag, caminhos_tag, exemplos_tag, out_tag):
        print("\t" + dt + "...")
        items = np.array(pd.read_csv(caminhos['dataset'] + "items.csv", sep = rr.DELIMITER, quotechar= rr.QUOTECHAR, quoting= rr.QUOTING, encoding=rr.ENCODING))
        tags = np.array(pd.read_csv(caminhos['dataset'] + "tags.csv", sep = rr.DELIMITER, quotechar= rr.QUOTECHAR, quoting= rr.QUOTING, encoding=rr.ENCODING))
        for metodo in metodos:
            print("\t\t" + metodo + "...")
            embeddings = pd.read_pickle(caminhos['embeddings'] + metodo + "_item_embeddings.pkl")
            sparse = pd.read_pickle(caminhos['embeddings'] + metodo + "_sparse_repr.pkl")
            resultado = pd.DataFrame([], columns = ['Label', 'Nome', 'Tags'])

            print('\t\t\tCalculando...')
            sims = cosine_distances(embeddings)
            np.fill_diagonal(sims, np.inf)

            print("\t\t\tOrdenando...")
            sims_ord = np.argsort(sims, axis=1)[:, 1:3]

            lista_resultado = list()
            lista_resposta = list()

            print("\t\t\tEscrevendo...")
            for alvo_name in exemplos:
                alvo_id = exemplos[alvo_name]
                alvo_sims = sims_ord[sparse.get_idx_of_item(alvo_id)]
                r = random.randint(0, len(embeddings))
                while r == alvo_sims[0] or r == alvo_sims[1] or r == sparse.get_idx_of_item(alvo_id):
                    r = random.randint(0, len(embeddings))
                tupla = [alvo_id, sparse.get_item_of_idx(alvo_sims[0]), sparse.get_item_of_idx(alvo_sims[1]), sparse.get_item_of_idx(r)]
                random.shuffle(tupla)
                lista_resultado.append(tupla)
                lista_resposta.append(tupla.index(sparse.get_item_of_idx(r)))
            
            for i_rand in range(N_SAMPLES_RAN):
                alvo_id = random.randint(0, len(embeddings))
                alvo_sims = sims_ord[alvo_id]
                r = random.randint(0, len(embeddings))
                while r == alvo_sims[0] or r == alvo_sims[1] or r == alvo_id:
                    r = random.randint(0, len(embeddings))
                tupla = [sparse.get_item_of_idx(alvo_id), sparse.get_item_of_idx(alvo_sims[0]), sparse.get_item_of_idx(alvo_sims[1]), sparse.get_item_of_idx(r)]
                random.shuffle(tupla)
                lista_resultado.append(tupla)
                lista_resposta.append(tupla.index(sparse.get_item_of_idx(r)))

            #random.shuffle(lista_resultado)

            for exs in lista_resultado:
                for i, ex in enumerate(exs):
                    pos_item = np.where(items[:, 0] == ex)[0][0]
                    linha = out.line_metadata(items[pos_item], tags)
                    df_aux = pd.DataFrame(np.array([[i+1, linha[0], linha[1]]]), columns = ['Label', 'Nome', 'Tags'])
                    resultado = pd.concat([resultado, df_aux])
                df_aux = pd.DataFrame(np.array([[np.NaN, np.NaN, np.NaN]]), columns = ['Label', 'Nome', 'Tags'])
                resultado = pd.concat([resultado, df_aux])
            
            resultado.to_csv('outlier_qualitative/' + "outlier_" + dt + '_' + metodo + '.csv', index=False)
            f=open('outlier_qualitative/' + "resposta_" + dt + '_' + metodo + '.txt','w')
            for ele in lista_resposta:
                f.write(str(ele)+'\n')
            f.close()
            print('\t\t\tConcluído!')

print("Criando teste para bases com categorias...")
for dt, caminhos, exemplos, out in zip(datasets_cat, caminhos_cat, exemplos_cat, out_cat):
        print("\t" + dt + "...")
        items = np.array(pd.read_csv(caminhos['dataset'] + "items.csv", sep = rr.DELIMITER, quotechar= rr.QUOTECHAR, quoting= rr.QUOTING, encoding=rr.ENCODING))
        for metodo in metodos:
            print("\t\t" + metodo + "...")
            embeddings = pd.read_pickle(caminhos['embeddings'] + metodo + "_item_embeddings.pkl")
            sparse = pd.read_pickle(caminhos['embeddings'] + metodo + "_sparse_repr.pkl")
            if dt == 'MovieLens':
                resultado = pd.DataFrame([], columns = ['Label', 'Nome', 'Categorias', 'Tags'])
            else:
                resultado = pd.DataFrame([], columns = ['Label', 'Nome', 'Categorias'])

            print('\t\t\tCalculando...')
            sims = cosine_distances(embeddings)
            np.fill_diagonal(sims, np.inf)

            print("\t\t\tOrdenando...")
            sims_ord = np.argsort(sims, axis=1)[:, 1:3]

            lista_resultado = list()
            lista_resposta = list()

            print("\t\t\tEscrevendo...")
            for alvo_name in exemplos:
                alvo_id = exemplos[alvo_name]
                alvo_sims = sims_ord[sparse.get_idx_of_item(alvo_id)]
                r = random.randint(0, len(embeddings))
                while r == alvo_sims[0] or r == alvo_sims[1] or r == sparse.get_idx_of_item(alvo_id):
                    r = random.randint(0, len(embeddings))
                tupla = [alvo_id, sparse.get_item_of_idx(alvo_sims[0]), sparse.get_item_of_idx(alvo_sims[1]), sparse.get_item_of_idx(r)]
                random.shuffle(tupla)
                lista_resultado.append(tupla)
                lista_resposta.append(tupla.index(alvo_id))
            
            for i_rand in range(N_SAMPLES_RAN):
                alvo_id = random.randint(0, len(embeddings))
                alvo_sims = sims_ord[alvo_id]
                r = random.randint(0, len(embeddings))
                while r == alvo_sims[0] or r == alvo_sims[1] or r == alvo_id:
                    r = random.randint(0, len(embeddings))
                tupla = [sparse.get_item_of_idx(alvo_id), sparse.get_item_of_idx(alvo_sims[0]), sparse.get_item_of_idx(alvo_sims[1]), sparse.get_item_of_idx(r)]
                random.shuffle(tupla)
                lista_resultado.append(tupla)
                lista_resposta.append(tupla.index(sparse.get_item_of_idx(alvo_id))+1)

            #random.shuffle(lista_resultado)

            for exs in lista_resultado:
                for i, ex in enumerate(exs):
                    pos_item = np.where(items[:, 0] == ex)[0][0]
                    linha = out.line_metadata(items[pos_item])
                    if dt == 'MovieLens':
                        df_aux = pd.DataFrame(np.array([[i+1, linha[0], linha[1], linha[2]]]), columns = ['Label', 'Nome', 'Categorias', 'Tags'])
                    else:
                        df_aux = pd.DataFrame(np.array([[i+1, linha[0], linha[1]]]), columns = ['Label', 'Nome', 'Categorias'])
                    resultado = pd.concat([resultado, df_aux])
                if dt == 'MovieLens':
                    df_aux = pd.DataFrame(np.array([[np.NaN, np.NaN, np.NaN, np.NaN]]), columns = ['Label', 'Nome', 'Categorias', 'Tags'])
                else:
                    df_aux = pd.DataFrame(np.array([[np.NaN, np.NaN, np.NaN]]), columns = ['Label', 'Nome', 'Categorias'])
                resultado = pd.concat([resultado, df_aux])
            
            resultado.to_csv('outlier_qualitative/' + "outlier_" + dt + '_' + metodo + '.csv', index=False)
            f=open('outlier_qualitative/' + "resposta_" + dt + '_' + metodo + '.txt','w')
            for ele in lista_resposta:
                f.write(str(ele)+'\n')
            f.close()
            print('\t\t\tConcluído!')
print("Teste Outlier finalizado com sucesso!")




            



