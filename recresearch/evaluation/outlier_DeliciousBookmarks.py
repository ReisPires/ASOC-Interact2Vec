import numpy as np

class OutlierDeliciousBookmarks(object):
    def line_metadata(self, line_, tags):
        line = np.array(line_)
        if line.size > 0:
            name = line[1]
            name_tag = ""
            if isinstance(line[3], str):
                tok_tag = line[3].split('/')
                for tg in tok_tag:
                    tag = tg.split()
                    if len(tag) > 0:
                        pos_tag = np.where(tags[:, 0] == int(tag[0]))[0]
                        name_tag += tags[pos_tag][0][1] + " " + tag[1] + "/ "
            linha = [name, name_tag]
            return linha
        return line