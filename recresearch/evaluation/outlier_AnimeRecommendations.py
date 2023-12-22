import pandas as pd
import numpy as np

class OutlierAnimeRecommendations(object):
    def line_metadata(self, line_):
        line = np.array(line_)
        name = line[1] 
        cat = line[2]
        linha = [name, cat]
        return linha