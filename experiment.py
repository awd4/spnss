import cPickle as pickle
import csv
import os


def save_results(rl, filename):
    with open(filename+'.pkl', 'wb') as fp:
        pickle.dump(rl, fp)
    with open(filename+'.csv', 'wb') as fp:
        fields = ['name', 'structure', 't', 'cp', 'time', 'num_nodes', 'num_edges', 'vld_llh', 'tst_llh']
        writer = csv.DictWriter(fp, fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rl)


def csv_path(name, structure, dirname):
    dirname = 'results/'+dirname
    fnl = [os.path.join(dirname, fn) for fn in os.listdir(dirname)]
    fnl = [fn for fn in fnl if name in fn and structure in fn and fn.endswith('.csv')]
    assert len(fnl) == 1, str(fnl)
    return fnl[0]


def read_csv(path):
    reader = csv.DictReader(open(path, 'rb'))
    rows = [row for row in reader]
    return rows

