import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy.sparse import csr_matrix
from recresearch.evaluation.content import top_N_tags

class Matrix_LastFMListened(object):
    def create_matrix(self, df):
        print("Criando matriz LastFM - Listened...")
        ids = df.id_item.values
        tags = top_N_tags(df['tags(qt)_item'])
        tags = list(tags)
        le_tag = LabelEncoder()
        le_tag.fit(tags)
        le_id = LabelEncoder()
        le_id.fit(ids)
        
        list_interacts = list()

        print("\tPegando lista de id, tag e qt...")
        for i, j in df.iterrows():
            print("\t\t\tLinha " + str(i+1) + " de " + str(len(df)), end = '\r', flush = True)
            if type(j.values[3]) is str:
                tgs = j.values[3].split("/")
                for tg in tgs:
                    tag, qt = tg.split()
                    if tag in tags:
                        list_interacts.append([le_id.transform([j.values[0]])[0], le_tag.transform([int(tag)])[0], 1])

        print("")
        print("\tCriando matriz esparsa...")
        matriz_de_coordenadas = np.array(list_interacts)
        linhas = matriz_de_coordenadas[:, 0]
        colunas = matriz_de_coordenadas[:, 1]
        valores = matriz_de_coordenadas[:, 2]
        matriz_esparsa = csr_matrix((valores, (linhas, colunas)), shape = (len(le_id.classes_), len(le_tag.classes_)))

        print("\tRetornando matriz esparsa e LabelEncoder dos itens...")
        return matriz_esparsa, le_id
