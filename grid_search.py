import argparse
import numpy as np
import time
import random
import os

import dag
import tree
import spnss.deps as deps
import spnss.network as network
import experiment



def eval_grid_point(learner, name, t, cp, verbosity):
    structure = learner
    learner = {'dag':dag.run, 'tree':tree.run}[structure]
    rs = random.getstate(), np.random.get_state()
    start = time.time()
    net, vld_llh, tst_llh = learner(name, t, cp, verbosity=verbosity)
    elapsed = time.time() - start
    result = {'name':name, 't':t, 'cp':cp, 'time':elapsed, 'num_nodes':len(net.pot), 'num_edges':net.num_edges, \
              'vld_llh':vld_llh, 'tst_llh':tst_llh, 'spn':network.to_data(net), 'random_state':rs, 'structure':structure}
    return result


def search(name, structure, dirname):
    rname = 'results/%s/%s'%(dirname, name)
    ename = 'results/%s/error.%s'%(dirname, name)

    if os.path.isfile(rname+'.' + structure + '.pkl') and os.path.isfile(rname+'.' + structure + '.csv'):
        print 'skipping ', rname
        return

    rlist = []
    try:
        # run grid search
        for t in [0.3, 0.1, 0.03, 0.01, 0.003]:
            for cp in [.1, .3, 1., 3., 10.]:
                print 'ok', name, '\tt: %.5f'%t, '\tcp: ', cp
                r = eval_grid_point(structure, name, t, cp, 1)
                rlist.append(r)
    except Exception as e:
        print 'grid search caught exception: something went wrong at t:%f, cp:%f'%(t, cp)
        experiment.save_results(rlist, ename+'.'+structure)
        import traceback
        traceback.print_exc(file=open(ename+'.'+structure+'.tb', 'w'))
        raise
    else:
        experiment.save_results(rlist, rname+'.'+structure)


def ir(f):
    return int(round(f))

def best_row(rows):
    br = None
    bl = float('-inf')
    for row in rows:
        lh = float(row['vld_llh'])
        if lh > bl:
            bl = lh
            br = row
    return br

def best_grid_point(name, structure, dirname):
    bp = experiment.csv_path(name, structure, dirname)
    rows = experiment.read_csv(bp)
    return best_row(rows)


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser(description='Grid search over SPN structure search parameters')
    parser.add_argument('dataset', help='name of the dataset to use in training, validation, and testing')
    parser.add_argument('structure', help='type of model: tree or dag.')
    parser.add_argument('-o', '--output-directory', default='grid', help='sub-directory to store resuls in (default:grid)')
    args = parser.parse_args()

    search(args.dataset, args.structure, args.output_directory)


if __name__ == '__main__':
    main()

