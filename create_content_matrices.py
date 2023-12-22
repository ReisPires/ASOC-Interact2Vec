import pandas as pd
import pickle as pk
from recresearch.content_representation.matrix_BestBuy import Matrix_BestBuy
from recresearch.content_representation.matrix_Book_Crossing import Matrix_Book_Crossing
from recresearch.content_representation.matrix_DeliciousBookmarks import Matrix_DeliciousBookmarks
from recresearch.content_representation.matrix_LastFM_Listened import Matrix_LastFMListened
from recresearch.content_representation.matrix_MovieLens_cat import Matrix_MovieLens_cat
from recresearch.content_representation.matrix_AnimeRecommendations import Matrix_AnimeRecommendations
from recresearch.content_representation.matrix_MovieLens_tag import Matrix_MovieLens_tag
import recresearch as rr
import os

os.makedirs('content_matrices', exist_ok=True)

datasets_tags = ['DeliciousBookmarks',
                'Last.FM - Listened',
                'MovieLens-tag']

datasets_cat = ['Anime Recommendations',
                'BestBuy',
                'Book-Crossing',
                'MovieLens-cat']

caminhos_tags = ["./datasets/DeliciousBookmarks/",
                "./datasets/Last.FM - Listened/",
                "./datasets/MovieLens/"]

caminhos_cat = ["./datasets/Anime Recommendations/",
                "./datasets/BestBuy/",
                "./datasets/Book-Crossing/",
                "./datasets/MovieLens/"]

matriz_bestbuy = Matrix_BestBuy()
matriz_book_crossing = Matrix_Book_Crossing()
matriz_deliciousbookmarks = Matrix_DeliciousBookmarks()
matriz_lastfm = Matrix_LastFMListened()
matriz_movielens_cat = Matrix_MovieLens_cat()
matriz_animerecommendations = Matrix_AnimeRecommendations()
matriz_movielens_tag = Matrix_MovieLens_tag()

classes_matriz_tags = [matriz_deliciousbookmarks, matriz_lastfm, matriz_movielens_tag]
classes_matriz_cat = [matriz_animerecommendations, matriz_bestbuy, matriz_book_crossing, matriz_movielens_cat]

print("Criando matrizes de bases com tags:")
for dt_tag, caminho_tag, classe_tag in zip(datasets_tags, caminhos_tags, classes_matriz_tags):
    items = pd.read_csv(caminho_tag + 'items.csv', sep = rr.DELIMITER, quotechar= rr.QUOTECHAR, quoting= rr.QUOTING, encoding=rr.ENCODING)    

    matriz, le = classe_tag.create_matrix(items)

    with open('content_matrices/' + dt_tag + '_matrix' + '.pkl', 'wb') as f:
        pk.dump(matriz, f)

    with open('content_matrices/' + dt_tag + '_encoder' '.pkl', 'wb') as f:
        pk.dump(le, f) 

print("Criando matrizes de bases com categorias:")
for dt_cat, caminho_cat, classe_cat in zip(datasets_cat, caminhos_cat, classes_matriz_cat):
    items = pd.read_csv(caminho_cat + 'items.csv', sep = rr.DELIMITER, quotechar= rr.QUOTECHAR, quoting= rr.QUOTING, encoding=rr.ENCODING)  

    matriz, le = classe_cat.create_matrix(items)

    with open('content_matrices/' + dt_cat + '_matrix' + '.pkl', 'wb') as f:
        pk.dump(matriz, f)

    with open('content_matrices/' + dt_cat + '_encoder' '.pkl', 'wb') as f:
        pk.dump(le, f) 

print("Matrizes geradas com sucesso!")




