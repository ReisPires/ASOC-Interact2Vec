import pandas as pd
import numpy as np

class OutlierBook_Crossing(object):
    def line_metadata(self, line_):
        line = np.array(line_)
        if line.size > 0:
            name = line[0][1] + ': ' + line[0][2] + '/' + str(line[0][3]) + '/' + line[0][4]
            return name
        return line