import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy.sparse import csr_matrix

class Matrix_MovieLens_cat(object):
    def create_matrix(self, df):
        print("Criando matriz MovieLens_cat...")
        list_cat = []
        cat = df.genero.unique()

        print("\tPegando categorias únicas...")
        for cate in cat:
            if "/" in cate:
                s = str(cate).split("/")
                for i in s:
                    list_cat.append(i)
            else:
                list_cat.append(cate)    

        df_aux = pd.DataFrame(list_cat)
        df_aux.columns = ["category_item"]
        cat = df_aux.category_item.unique()
        ids = df.id_item.values
        le_cat = LabelEncoder()
        le_cat.fit(cat)
        le_id = LabelEncoder()
        le_id.fit(ids)

        list_interacts = list()

        print("\tPegando lista de id e categoria...")
        for i, j in df.iterrows():
            print("\t\t\tLinha " + str(i+1) + " de " + str(len(df)), end = '\r', flush = True)
            s = str(j.values[2]). split("/")
            for x in s:
                list_interacts.append([le_id.transform([j.values[0]])[0], le_cat.transform([x])[0], 1])

        print("")
        print("\tCriando matriz esparsa...")
        matriz_de_coordenadas = np.array(list_interacts)
        linhas = matriz_de_coordenadas[:, 0]
        colunas = matriz_de_coordenadas[:, 1]
        valores = matriz_de_coordenadas[:, 2]
        matriz_esparsa = csr_matrix((valores, (linhas, colunas)), shape = (len(le_id.classes_), len(le_cat.classes_)))

        print("\tRetornando matriz esparsa e LabelEncoder dos itens...")
        return matriz_esparsa, le_id
