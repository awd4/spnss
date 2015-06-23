import argparse
import logging
import time

import learn
import spnss.knobs as knobs 
import spnss.gens as gens


def run(name, t, cp=None, verbosity=1):
    level = {0:logging.ERROR, 1:logging.WARNING, 2:logging.INFO, 3:logging.DEBUG}[verbosity]
    logging.basicConfig(level=level)

    # learn SPN
    if cp is not None: knobs.cluster_penalty = cp
    knobs.min_instances = 2

    trn, vld, tst, schema = learn.load_data(name)
    start = time.time()

    net = gens.learn_spn(trn, schema, t)

    #learn.smooth_network(net, vld, verbosity>0)
    tst_llh = net.llh(tst)
    vld_llh = net.llh(vld)
    print name, '\tt: %.5f'%t, '\tcp: ', cp, '\ttime:%.1f'%(time.time()-start), '\ttree', len(net.pot), 'va:%8.4f'%vld_llh, 'te:%8.4f'%tst_llh

    return net, vld_llh, tst_llh



def main():
    # parse command line arguments
    parent_parsers = [learn.dataset_parser(), learn.t_cp_parser(), learn.vq_parser(), learn.io_parser()]
    parser = argparse.ArgumentParser(description='Perform SPN structure search', parents=parent_parsers)
    args = parser.parse_args()
    import random
#    random.seed(10)
    verbosity = args.v - args.q
    run(args.dataset, args.threshold, args.cluster_penalty, verbosity=verbosity) 


if __name__ == '__main__':
    main()


