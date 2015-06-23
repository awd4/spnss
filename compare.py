import grid_search as gs
import experiment
from multiprocessing import Pool
import os

import numpy as np


names = ['mnperm%.2d'%i for i in range(2,10)] + \
        ['uperm%.2d'%i  for i in range(2,10)] + \
        ['accidents', 'ad', 'baudio', 'bbc', 'bnetflix', 'book', 'c20ng', \
         'cr52', 'cwebkb', 'dna', 'jester', 'kdd', 'kosarek', 'msnbc', \
         'msweb', 'nltcs', 'plants', 'pumsb_star', 'tmovie', 'tretail']

dirname = 'compare'

def compare(name, structure):
    #if name == 'c20ng' and structure == 'tree':
    #    return
    rlist = []
    rname = 'results/%s/%s.%s'%(dirname, name, structure)
    ename = 'results/%s/error.%s.%s'%(dirname, name, structure)
    try:
        if os.path.isfile(rname+'.pkl') and os.path.isfile(rname+'.csv'):
            print 'skipping ', rname
            return
        br = gs.best_grid_point(name, structure, 'grid')
        t = float(br['t'])
        cp = float(br['cp'])
        for i in range(10):
            print i, name
            r = gs.eval_grid_point(structure, name, t, cp, 1)
            rlist.append( r )
    except Exception as e:
        print 'compare() caught exception: something went wrong'# at i:%d'%i
        experiment.save_results(rlist, ename)
        import traceback
        traceback.print_exc(file=open(ename+'.tb', 'w'))
        raise
    else:
        experiment.save_results(rlist, rname)


def t_test(name):
    #if name == 'c20ng':
    #    return
    dresults = experiment.read_csv( experiment.csv_path(name, 'dag', 'compare') )
    tresults = experiment.read_csv( experiment.csv_path(name, 'tree', 'compare') )
    dtst = [float(r['tst_llh']) for r in dresults]
    ttst = [float(r['tst_llh']) for r in tresults]
    dsize = [int(r['num_nodes']) for r in dresults]
    tsize = [int(r['num_nodes']) for r in tresults]
    dtime = [float(r['time']) for r in dresults]
    ttime = [float(r['time']) for r in tresults]
    print (name+'       ')[:8], '\t%.4f'%np.mean(dtst), '\t%.4f'%np.mean(ttst), '\t', np.mean(dtst) > np.mean(ttst), '\t\t', '%6d'%int(np.mean(dsize)), '%7d'%int(np.mean(tsize)), '\t', '%.4f'%(np.mean(dsize)/np.mean(tsize)), '\t%8.2f'%np.mean(dtime), '  %8.2f'%np.mean(ttime), np.mean(dtime) / np.mean(ttime),
    from scipy import stats
    print '\t', stats.ttest_ind( dtst, ttst, equal_var=False )[1] < 0.05


def t_tests():
    print 'name\t\tdag_tst\t\t', 'tree_tst\t', 'dag > tree\t', ' dsize', 'tsize\t', '|ds|/|ts|', 'dtime\t\t', 'ttime\t', 'dt/tt'
    for name in names:
        t_test(name)




if False:
    pool = Pool(5)
    structure = 'tree'
    tasks = [(n, structure) for n in names]
    results = [pool.apply_async(compare, t) for t in tasks]
    for r in results:
        r.get()

    pool.close()
    pool.join()
elif True:
    t_tests()
else:
    for structure in ['dag', 'tree']:
        print '\n\t\t\ttime\tvld_llh\t\t\tnum_nodes'
        for name in names:
            if name == 'c20ng': continue
            br = gs.best_grid_point(name, structure, 'grid')
            print (name + '          ')[:8], '\t', structure, '\t', int(float(br['time'])), '\t', br['vld_llh'], '\t', br['num_nodes']




