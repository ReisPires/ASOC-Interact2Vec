import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy.sparse import csr_matrix

class Matrix_Book_Crossing(object):
    def create_matrix(self, df):
        print("Criando matriz Book-Crossing...")
        df.loc[~df['author_item'].isna(), 'author_item'] = 'aut_' + df['author_item'].dropna().astype(str)
        df.loc[~df['year_item'].isna(), 'year_item'] = 'aut_' + df['year_item'].dropna().astype(str)
        df.loc[~df['publisher_item'].isna(), 'publisher_item'] = 'aut_' + df['publisher_item'].dropna().astype(str)
        df = df.set_index('id_item')
        authors = df.author_item.dropna().unique()
        years = df.year_item.dropna().unique()
        publishers = df.publisher_item.dropna().unique()
        ids = df.index
        le_cat = LabelEncoder()
        cats = np.concatenate((authors, years, publishers), axis=None)
        le_cat.fit(cats)
        le_id = LabelEncoder()
        le_id.fit(ids)

        print("\tPegando lista de id e categoria...")
        list_column_author = le_cat.transform(df.author_item.dropna())
        list_column_year = le_cat.transform(df.year_item.dropna())
        list_column_publisher = le_cat.transform(df.publisher_item.dropna())
        list_rows_author = le_id.transform(df.author_item.dropna().index.values)
        list_rows_year = le_id.transform(df.year_item.dropna().index.values)
        list_rows_publisher = le_id.transform(df.publisher_item.dropna().index.values)
        column_cat = np.concatenate((list_column_author, list_column_year, list_column_publisher), axis=None)
        column_id = np.concatenate((list_rows_author, list_rows_year, list_rows_publisher), axis=None)
        ones = np.ones(len(column_id))

        print("\tCriando matriz esparsa...")
        matriz_de_coordenadas = np.vstack((column_id, column_cat, ones)).T
        linhas = matriz_de_coordenadas[:, 0]
        colunas = matriz_de_coordenadas[:, 1]
        valores = matriz_de_coordenadas[:, 2]
        matriz_esparsa = csr_matrix((valores, (linhas, colunas)), shape = (len(le_id.classes_), len(le_cat.classes_)))

        print("\tRetornando matriz esparsa e LabelEncoder dos itens...")
        return matriz_esparsa, le_id

