# distutils: language = c++

import numpy as np

cimport spnss.mathutil as mu
from libcpp.vector cimport vector

from spnss.nodes cimport CategoricalNode


'''
    Distribution-node trainers:
        One type of maximum a-posteriori (MAP) trainer:
            categorical with dirichlet prior
    Network trainers (see "Discriminative Learning of Sum-Product Networks" by Gens and Domingos):
        Four types of gradient-descent (gd) trainers:
            hard/soft inference
            discriminative/generative objective function
        Two types of expectation-maximization (em) trainers:
            hard/soft Inference
            always uses a generative objective function

    Another way to classify the network trainers:
        Forward pass    Backward pass   Objective function          Trainer
        soft            soft            likelihood                  soft em  / soft  gradient descent (the difference may be in the way the weights are updated...?)
        soft            hard            likelihood                  mixed em / mixed gradient descent
        hard            soft            likelihood                  ?
        hard            hard            likelihood                  hard em  / hard  gradient descent ( again, what's the difference

        soft            soft            conditional likelihood      soft  gradient descent
        soft            hard            conditional likelihood      mixed gradient descent
        hard            soft            conditional likelihood      ?
        hard            hard            conditional likelihood      hard  gradient descent
'''

def categorical_map_with_dirichlet_prior(num_vals, data, prior_params=None):
    # See pg. 21 of http://cog.brown.edu/~mj/classes/cg168/slides/wk04-Dirichlet.pdf
    # Also see Wikipedia's "Categorical distribution" page
    data = np.asarray(data)
    prior = np.ones(num_vals) if prior_params is None else np.asarray(prior_params, dtype=np.float64)
    assert prior.shape == (num_vals,)
    assert prior.min() >= 1
    assert data.shape[0] == np.prod(data.shape)
    assert 0 <= data.min() and data.max() <= num_vals-1
    assert (data == data.astype(np.int)).all()
    data = data.astype(np.int)
    n = len(data)
    am1 = prior - 1
    return (np.bincount(data.flat, minlength=num_vals) + am1) / (n + am1.sum())

def categorical_node_map_with_dirichlet_prior(CategoricalNode cn, data, prior_params=None):
    masses = categorical_map_with_dirichlet_prior(cn.masses.size(), data, prior_params)
    cn.set_masses(masses)


def categorical_mle(CategoricalNode cn, data):
    nv = cn.masses.size()
    data = np.asarray(data)
    assert data.shape[0] == np.prod(data.shape)
    assert 0 <= data.min() and data.max() <= nv-1
    assert (data == data.astype(np.int)).all()
    cn.set_masses( np.bincount(data.flat, minlength=nv).astype(np.float64) / len(data) )


def initialize_network(network):
    ''' Initialize sum node and categorical node params using (uniform) Dirichlets. '''
    for n in network.pot:
        if   n.is_sum():         n.set_weights(np.random.dirichlet([1]*n.weights.size()))
        elif n.is_categorical(): n.set_masses( np.random.dirichlet([1]*n.masses.size()))


def build_counts(network, reparameterize=False):
    # If reparameterize is true then SPN weight w is equal to exp(w_count). Reparameterizing
    # prevents us from computing negative weights during gradient descent.
    counts = {}
    for n in network.pot:
        if   n.is_sum():         counts[n] = [mu.log(w) if reparameterize else w for w in n.weights]
        elif n.is_categorical(): counts[n] = [mu.log(m) if reparameterize else m for m in n.masses]
    return counts


def backward_max_generative_trainer(network, em=True, forward_max=True, reparameterize=False, eta=1.0, initialize=True, normalize=True):
    ''' Performs either hard or mixed generative training.
        
        If em is True, then performs hard or mixed generative expectation-maximization.
        Otherwise it performs hard or mixed generative gradient descent.
    '''
    # TODO: Generalize. Currently works with SPNs consisting of sum, product, and categorical nodes only.
    assert not (em and (reparameterize or eta != 1.0))
    if initialize:
        initialize_network(network)
    counts = build_counts(network, reparameterize)
    def train(batch):
        for d in batch:
            network.set(d)
            if forward_max:
                network.forward_max_log()
            else:
                network.forward_log()
            max_nodes = network.backward_max_nodes_log()
            for n in max_nodes:     # update counts and params
                if   n.is_sum():         j = [c in max_nodes for c in n.children].index(True)
                elif n.is_categorical(): j = int(n.rv.value) if not mu.isnan(n.rv.value) else np.argmax(n.masses)
                else:
                    continue
                # em updates
                if em:
                    counts[n][j] += 1.0
                # gd updates
                elif reparameterize:
                    counts[n][j] += eta
                else:
                    counts[n][j] += eta / counts[n][j]
                params = [mu.exp(c) for c in counts[n]] if reparameterize else counts[n]
                if   n.is_sum():         n.set_weights(params)
                elif n.is_categorical(): n.set_masses( params)
                if normalize:
                    n.normalize()
    return train


def backward_max_discriminative_gd_trainer(network, forward_max=True, reparameterize=True, eta=1.0, label_index=-1, initialize=True, normalize=True):
    if initialize:
        initialize_network(network)
    counts = build_counts(network, reparameterize)
    label_rv = network.variables[label_index]
    label_dn = [n for n in network.pot if n.is_distribution() and n.rv == label_rv]
    rtopo = network.graph.rtopo(label_dn)
    def train(batch):
        for d in batch:
            network.set(d)
            if forward_max:
                network.forward_max_log()
            else:
                network.forward_log()
            max_nodes = network.backward_max_nodes_log()
            for n in max_nodes:     # update counts and params
                if   n.is_sum():         j = [c in max_nodes for c in n.children].index(True)
                elif n.is_categorical(): j = int(n.rv.value) if not mu.isnan(n.rv.value) else np.argmax(n.masses)
                else:
                    continue
                if reparameterize:
                    counts[n][j] += eta
                else:
                    counts[n][j] += eta / counts[n][j]
            network.forward_nodes(rtopo, forward_max, True)
            max_nodes = network.backward_max_nodes_log()
            for n in max_nodes:     # update counts and params
                if   n.is_sum():         j = [c in max_nodes for c in n.children].index(True)
                elif n.is_categorical(): j = int(n.rv.value)
                else:
                    continue
                if reparameterize:
                    counts[n][j] -= eta
                else:
                    counts[n][j] -= eta / counts[n][j]
                params = [mu.exp(c) for c in counts[n]] if reparameterize else counts[n]
                if   n.is_sum():         n.set_weights(params)
                elif n.is_categorical(): n.set_masses( params)
                if normalize:
                    n.normalize()




def em_trainer(network, inference='hard', initialize=True, normalize=True):
    ''' Return an expectation-maximization trainer. '''
    forward_max = inference == 'hard'
    if inference in ['hard', 'mixed']:
        return backward_max_generative_trainer(network, True, forward_max, False, 1.0, initialize, normalize)
    elif inference == 'soft':
        raise NotImplementedError, 'soft EM not implemented yet'


def gd_trainer(network, inference='hard', discriminative=True, reparameterize=True, eta=1.0, label_index=-1, initialize=True, normalize=True):
    ''' Return a gradient-descent trainer. '''
    forward_max = inference == 'hard'
    if discriminative:
        if inference in ['hard', 'mixed']:
            return backward_max_discriminative_gd_trainer(network, forward_max, reparameterize, eta, label_index, initialize, normalize)
        elif inference == 'soft':
            raise NotImplementedError, 'soft discriminative gradient descent not implemented yet'
    else:
        if inference in ['hard', 'mixed']:
            return backward_max_generative_trainer(network, False, forward_max, reparameterize, eta, initialize, normalize)
        elif inference == 'soft':
            raise NotImplementedError, 'soft generative gradient descent not implemented yet'


# Hard generative em/gd is done (but not tested!)
# TODO: Soft generative em/gd
# Hard discriminative gd (but not tested!)
# TODO: Soft discriminative gd
