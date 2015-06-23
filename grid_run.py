from multiprocessing import Pool
import sys

import grid_search as gs

structure = sys.argv[1]

names = ['mnperm%.2d'%i for i in range(2,10)] + \
        ['uperm%.2d'%i  for i in range(2,10)] + \
        ['accidents', 'ad', 'baudio', 'bbc', 'bnetflix', 'book', 'c20ng', \
         'cr52', 'cwebkb', 'dna', 'jester', 'kdd', 'kosarek', 'msnbc', \
         'msweb', 'nltcs', 'plants', 'pumsb_star', 'tmovie', 'tretail']
#names = names[:10]

pool = Pool(3)
tasks = [(n, structure, 'grid') for n in names]
results = [pool.apply_async(gs.search, t) for t in tasks]
for r in results:
    r.get()

pool.close()
pool.join()

