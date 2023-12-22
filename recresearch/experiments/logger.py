import csv
import os

import recresearch as rr

class BasicLogger(object):

    def __init__(self, filepath, mode='w', verbose=True):
        os.makedirs(rr.DIR_LOG, exist_ok=True)
        self.logger = open(os.path.join(rr.DIR_LOG, filepath), mode)
        self.verbose = verbose
    
    def __del__(self):
        self.logger.close()

    def _force_disk_write(self):
        self.logger.flush()
        os.fsync(self.logger.fileno())

    def log(self, text):
        if self.verbose:
            print('{}\n'.format(text))
        self.logger.write('{}\n'.format(text))
        self._force_disk_write()


class CSVLogger(BasicLogger):

        def __init__(self, filepath, mode='w', verbose=True, header=None):
            super().__init__(filepath, mode=mode, verbose=verbose)
            if header is not None:
                self.log(header)

        def log(self, text):
            super().log(rr.DELIMITER.join(text))


class ResultsLogger(BasicLogger):

    def __init__(self, filepath, mode='w', verbose=True, write_header=True):
        super().__init__(filepath, mode=mode, verbose=verbose)
        self.results_logger = csv.writer(self.logger, delimiter=rr.DELIMITER, quoting=csv.QUOTE_ALL)        
        if write_header:
            self._write_header()
            self._force_disk_write()

    def _write_header(self):
        pass


class ExplicitResultsLogger(ResultsLogger):

    HEADER = ['Dataset', 'Model', 'Params', 'RMSE', 'MAE', 'Time']

    def _write_header(self):
        self.results_logger.writerow(self.HEADER)

    def log(self, dataset='', model='', params='', rmse='', mae='', time=''):
        row = [dataset, model, params, rmse, mae, time]
        if self.verbose:
            print(' | '.join('{}: {}'.format(key, value) for key, value in zip(self.HEADER, row) if value != ''))
        self.results_logger.writerow(row)
        self._force_disk_write()


class ImplicitResultsLogger(ResultsLogger):

    HEADER = ['Dataset', 'Model', 'Params', 'PREC', 'REC', 'NDCG', 'Time']

    def _write_header(self):
        self.results_logger.writerow(self.HEADER)

    def log(self, dataset='', model='', params='', prec='', rec='', ndcg='', time=''):
        row = [dataset, model, params, prec, rec, ndcg, time]
        if self.verbose:
            print(' | '.join('{}: {}'.format(key, value) for key, value in zip(self.HEADER, row) if value != ''))
        self.results_logger.writerow(row)
        self._force_disk_write()


class DatasetLatexTable(BasicLogger):

    HEADER = ['Conjunto de Dados', 'Usuários', 'Itens', 'Interações', 'Int. p/ Usuário', 'Int. p/ Item', 'Esparsidade']
    
    def _write_header(self):
        self.logger.write('\\begin{table}[htpb]')
        self.logger.write('\\centering\n')
        self.logger.write('\\label{table:dados}\n')
        self.logger.write('\\begin{tabular}{l|c|c|c|c|c|c}\n')
        self.logger.write('\\hline \\hline\n')
        self.logger.write(' & '.join(['\\textbf{%s}' % h for h in self.HEADER]))
        self.logger.write('\\\\ \\hline \\hline\n')
        self._force_disk_write()

    def __init__(self, filepath, mode='w', verbose=False):
        super().__init__(filepath, mode=mode, verbose=verbose)
        self._write_header()

    def log(self, dataset, users, items, interactions, int_user_median, int_user_iqr, int_item_median, int_item_iqr, esparsity):
        self.logger.write('\\textbf{%s} & %d & %d & %d & %.1lf \\$pm$ %.1lf & %.1lf \\$pm$ %.1lf & %.2lf\\%% \\\\ \\hline\n' % (
            dataset, users, items, interactions, int_user_median, int_user_iqr, int_item_median, int_item_iqr, esparsity
        ))
        self._force_disk_write()

    def __del__(self):
        self.logger.write('\\end{tabular}\n')
        self.logger.write('\\end{table}\n')
        self._force_disk_write()


def dataframe_to_latex(filepath, df):
    os.makedirs(rr.DIR_LOG, exist_ok=True)
    logger = open(os.path.join(rr.DIR_LOG, filepath), 'w')
    logger.write('\\begin{table}[htpb]')
    logger.write('\\centering\n')
    logger.write('\\label{table:dados}\n')
    logger.write('\\begin{tabular}{l|c|c|c|c|c|c}\n')
    logger.write('\\hline \\hline\n')
    logger.write(' & '.join(['\\textbf{%s}' % h for h in df.columns]))
    logger.write('\\\\ \\hline \\hline\n')
    for _, row in df.iterrows():
        logger.write(' & '.join(['%s' % h for h in row.round(2).astype(str).values]))
        logger.write(' \\\\ \\hline\n')
    logger.write('\\end{tabular}\n')
    logger.write('\\end{table}\n')
    logger.close()