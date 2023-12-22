import pandas as pd
import numpy as np
from collections import Counter
import re

TOP_N = 50

def top_N_tags(tags_):
    tags = np.array(tags_)
    lista_tags = list()
    for linha in tags:
        tags_qt = str(linha).split('/')
        for tags in tags_qt:
            if re.search("^[0-9]+ [(][0-9]+[)]$", tags):
                tag = tags.split(" ")
                lista_tags.append(tag[0])
            else:
                lista_tags.append(tags)
    lista_tags = list(filter(lambda a: a != 'nan', lista_tags))
    lista_tags = list(filter(lambda a: a != '', lista_tags))
    counter = Counter(lista_tags)
    top_n_list = x = {tag for tag,_ in counter.most_common(TOP_N)}
    return top_n_list