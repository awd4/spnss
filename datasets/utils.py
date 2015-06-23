from math import log, factorial
import numpy as np


def schema_split_helper(data, schema, split, tr_size, va_size, nvals):
    if not (split or schema):
        return data
    retval = [data[:tr_size,:], data[tr_size:tr_size+va_size,:], data[tr_size+va_size:,:]] if split else [data]
    if schema:
        retval.append( nvals )
    return tuple(retval)


def floatify(*args):
    assert len(args) in [1, 4]
    if len(args) == 1:
        return args[0].astype(np.float)
    else:
        trn, vld, tst, schema = args
        trn = trn.astype(np.float)
        vld = vld.astype(np.float)
        tst = tst.astype(np.float)
        return trn, vld, tst, schema


def intify(*args):
    assert len(args) in [1, 4]
    if len(args) == 1:
        return args[0].round().astype(np.int)
    else:
        trn = args[0].round().astype(np.int)
        vld = args[1].round().astype(np.int)
        tst = args[2].round().astype(np.int)
        return trn, vld, tst, args[3]


def mnperm_trn_size(n):
    # The mnperm distribution place a "high" probabilty over a
    # set of instances and a "low" probability over the others.
    # The "high" probability is 10 times the "low" probability.
    # We sample from the distribution enough times so that there
    # is an r% chance of not sampling a particular set of r% of
    # the "high" probability instances.
    #
    # p = 10 / (n**n + 9 * n!) is the "high" probability
    # Let S be a subset of the "high" probability instances. The
    # size of S is r times the number of all high instances.
    # q = 1 - r * n! * p
    # is the probability of not sampling any instance in S using a
    # single draw and
    # q**k
    # is the probability of not having sampled any instance in S
    # after k draws.
    #
    # We set q**k = r and solve for k:
    # k = log(r) / log(q)
#    trn_size = log(0.1) / log( 1.0 - max(1,int(round(0.1*factorial(n)))) * (10. / (n**n + 9.0*factorial(n))) )
    r = 0.03
    trn_size = log(r) / log( 1.0 - r * factorial(n) * (10. / (n**n + 9.0*factorial(n))) )
    trn_size = max(1, int(round(trn_size)))
    return trn_size


