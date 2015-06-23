import os
import os.path

import utils
import loaders


GLOBALS = globals()

gd_names = ['accidents', 'ad', 'baudio', 'bbc', 'bnetflix', 'book', 'c20ng', \
            'cr52', 'cwebkb', 'dna', 'jester', 'kdd', 'kosarek', 'msnbc', \
            'msweb', 'nltcs', 'plants', 'pumsb_star', 'tmovie', 'tretail']


for n in gd_names:
    f = loaders.gd_function_creator(n)
    GLOBALS[n] = f


mnperm   = loaders.load_mnperm
for i in range(2,13):
    f = loaders.mnperm_function_creator(i)
    GLOBALS['mnperm%d'%i] = f
    GLOBALS['mnperm%.2d'%i] = f


floatify = utils.floatify
intify   = utils.intify


