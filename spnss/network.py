import cPickle as pickle

import vv
import rdag
import nodes as nodesmod
from fast import forward


def factory(vrs, rdag_builder, *args, **kwargs):
    n = Network()
    n.variables = vv.vvec(vrs)
    n.graph     = rdag_builder(n.variables, *args, **kwargs)
    return n

def independent(vrs):
    return factory(vrs, rdag.independent)

def simple(vrs=None):
    if vrs is None:
        vrs = '2d2'
    return factory(vrs, rdag.simple)

def from_rdag(rdag):
    n = Network()
    n.graph = rdag
    n.variables = vv.VariableVector()
    for dn in [node for node in n.graph.pot() if node.is_distribution()]:
        if dn.rv not in n.variables:
            n.variables.add_variable(dn.rv)
    return n

def from_variables_and_root(variables, root):
    n = Network()
    n.graph = rdag.RootedDAG(root)
    n.variables = variables
    return n
from_vr = from_variables_and_root


def scopes(pot, vrs):
    # TODO: think about moving this somewhere else
    scopes = dict((n,set()) for n in pot)
    for n in pot:
        if n.is_distribution():
            scopes[n].add(vrs.index(n.rv))
            continue
        for c in n.children:
            scopes[n] |= scopes[c]
    return scopes


def to_data(net):
    nodes = net.variables + net.pot
    ndata = [nodesmod.to_data(n) for n in nodes]
    # replace object references with ids (children/rv)
    nids = {n:i for i,n in enumerate(nodes)}
    for i,d in enumerate(ndata):
        n = nodes[i]
        if n.is_sp():
            d[1] = [nids[c] for c in d[1]]
        elif n.is_distribution():
            d[1] = nids[d[1]]
    return ndata


def from_data(data):
    # build objects and replace ids with these objects
    nodes = []
    for i,d in enumerate(data):
        if d[0] in ['sum', 'prod']:
            d[1] = [nodes[c] for c in d[1]]
        elif d[0] in ['gaussian', 'categorical', 'indicator']:
            d[1] = nodes[d[1]]
        n = nodesmod.from_data(d)
        nodes.append(n)
    # build the network
    net = Network()
    net.variables = vv.VariableVector()
    for n in nodes:
        if n.is_rv():
            net.variables.add_variable(n)
    net.graph = rdag.RootedDAG(nodes[-1])
    return net


def copy(net):
    return from_data( to_data(net) )


def save(net, filename):
    ndata = to_data(net)
    with open(filename, 'wb') as fp:
        pickle.dump(ndata, fp)


def load(filename):
    with open(filename, 'rb') as fp:
        ndata = pickle.load(fp)
    return from_data(ndata)



class Network(object):

    def __init__(self):
        self.graph = None
        self.variables = None
        self._pot = None

    @property
    def pot(self):
        if self._pot is None and self.graph is not None:
            self._pot = self.graph.pot()
        return self._pot

    @pot.setter
    def pot(self, value):
        if value is not None:
            raise ValueError, 'Network.pot: cannot set pot to non-None value'
        self._pot = None

    @property
    def num_edges(self):
        ne = 0
        for n in self.pot:
            if n.is_sp():
                ne += len(n.children)
            elif n.is_distribution() and n.rv.is_discrete():
                ne += n.rv.num_values
        return ne

    def set(self, values):
        if values is None:
            return
        self.variables.set(values)

    def forward_nodes(self, nodes, maxspace, logspace):
        return forward(nodes, maxspace, logspace)

    def forward(self, values=None):
        self.set(values)
        return forward(self.pot, False, False)

    def forward_log(self, values=None):
        self.set(values)
        return forward(self.pot, False, True)

    def forward_max(self, values=None):
        self.set(values)
        return forward(self.pot, True, False)

    def forward_max_log(self, values=None):
        self.set(values)
        return forward(self.pot, True, True)

    def forward_rtopo(self, nodes):
        rtopo = self.graph.rtopo(nodes)
        return forward(rtopo, False, False)

    def forward_rtopo_log(self, nodes):
        rtopo = self.graph.rtopo(nodes)
        return forward(rtopo, False, True)

    def forward_rtopo_max(self, nodes):
        rtopo = self.graph.rtopo(nodes)
        return forward(rtopo, True, False)

    def forward_rtopo_max_log(self, nodes):
        rtopo = self.graph.rtopo(nodes)
        return forward(rtopo, True, True)

    def backward_max_nodes(self):
        return self.graph.max_weighted_dc()

    def backward_max_nodes_log(self):
        return self.graph.max_weighted_dc_log()

    def backward_max_edges(self):
        return self.graph.max_weighted_dc_edges()

    def backward_max_edges_log(self):
        return self.graph.max_weighted_dc_edges_log()

    def backward_max_edges_v2(self):
        return self.graph.max_weighted_dc_edges_v2()

    def backward_max_edges_v2_log(self):
        return self.graph.max_weighted_dc_edges_v2_log()

    def scopes(self):
        return scopes(self.pot, self.variables)

    def parents(self):
        p = {}
        for n in self.pot:
            for c in n.children:
                p[c] = [] if c not in p else p[c]
                p[c].append(n)
        return p

    def parents_of(self, node):
        p = []
        i = self.pot.index(node)
        for n in self.pot[i+1:]:
            if node in n.children:
                p.append(n)
        return p

    def llh(self, data):
        # TODO: move this into an inference.py module or somewhere else...
        l = 0.0
        for d in data:
            l += self.forward_log(d)
        return l / len(data)


