import pandas as pd
import numpy as np

class OutlierMovieLens(object):
    def line_metadata(self, line_):
        line = np.array(line_)
        name = line[1] 
        cat = line[2]
        tags = line[3]
        linha = [name, cat, tags]
        return linha