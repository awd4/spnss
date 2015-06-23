# distutils: language = c++

import spnss.mathutil as mu

from spnss.nodes cimport SumNode, ProdNode, GaussianNode, CategoricalNode, IndicatorNode


def pc_iter(Node n, dict parents):
    ''' Parent (or child) iterator.

        If parents is None return child iterator; otherwise return parent
        iterator.
    '''
    for an in n.children_iter() if parents is None else parents[n]:
        yield an


def ud_closure(nodes, dict parents=None):
    ''' Finds the upward (or downward) closure of the set of nodes.
        
        For our purposes we define the upward closure as the set of input nodes
        unioned with their ancestors. Similary, the downward closure is the set
        of input nodes unioned with their descendants. These definitions are
        inspired from that given on pgs. 35-36 of 'Probabilistic Graphical Models'
        by Daphne Koller.

        If parents is None the downward closure is returned; otherwise the
        upward closure is returned.
    '''
    cdef set cn = set()     # closure nodes
    cdef set currgen = set(nodes)
    cdef Node n, a
    while len(currgen) > 0:
        n = currgen.pop()
        cn.add(n)
        for a in pc_iter(n, parents):
            if a not in cn:
                currgen.add(a)
    return cn

def upward_closure(nodes, dict parents):
    return ud_closure(nodes, parents)

def downward_closure(nodes):
    return ud_closure(nodes)

def ancestors(nodes, dict parents):
    return upward_closure(nodes, parents).difference(nodes)

def descendants(nodes):
    return downward_closure(nodes).difference(nodes)


def check_topo_seeds(nodes, dict parents=None):
    ''' Check that no node in nodes is a descendant (or ancestor) of any other
        node in nodes.
        
        If parents is None descendants are checked; otherwise acestors are
        checked.
    '''
    def cts_visit(nodes, Node n, set visited, dict pars):
        for a in pc_iter(n, pars):
            if a in nodes:
                return True
            if a in visited:
                continue
            visited.add(a)
            if cts_visit(nodes, a, visited, pars):
                return True
        return False
    cdef set visited = set()
    for n in nodes:
        if cts_visit(nodes, n, visited, parents):
            return False
    return True

def fr_topo_sort(nodes, dict parents=None):
    ''' Return a forward (or reverse) topological sort using the given nodes as
        seed nodes.
        
        If parents is None then a forward topological sort is returned; otherwise
        a reverse topological sort is returned.
    '''
    if not check_topo_seeds(nodes, parents):
        raise ValueError, 'rdag.fr_topo_sort(): invalid seed nodes given.'

    cdef set duc = downward_closure(nodes) if parents is None else upward_closure(nodes, parents)
    cdef dict degree = dict((n,0) for n in duc)
    cdef Node n, a
    for n in duc:
        for a in pc_iter(n, parents):
            degree[a] = degree[a] + 1

    cdef list topo = []
    cdef set curr = set(nodes)
    while len(curr) > 0:
        n = curr.pop()
        topo.append(n)
        for a in pc_iter(n, parents):
            degree[a] = degree[a] - 1
            if degree[a] == 0:
                curr.add(a)

    if sum(degree.values()) != 0:
        raise ValueError, 'rdag.fr_topo_sort(): graph has a cycle'
    return topo

def topo_sort(nodes):
    return fr_topo_sort(nodes)

def rtopo_sort(nodes, dict parents):
    return fr_topo_sort(nodes, parents)


cdef fr_pot_to(Node n, set visited=None, dict parents=None):
    ''' Return the forward (or reverse) post-order traversal to node n. '''
    if visited is None:
        visited = set()
    visited.add(n)
    pot = []
    for a in pc_iter(n, parents):
        if a in visited:
            continue
        apot = fr_pot_to(a, visited, parents)
        pot.extend(apot)
    pot.append(n)
    return pot

cpdef pot_to(Node n):
    return fr_pot_to(n)

cpdef rpot_to(Node n, dict parents):
    return fr_pot_to(n, None, parents)


cpdef max_downward_closure_of(Node n, set mdo=None):
    if mdo is None:
        mdo = set()
    mdo.add(n)
    if not n.has_children_attr():
        return mdo
    children = [n.max_child()] if n.is_sum() else n.children
    for c in children:
        if c in mdo:
            continue
        max_downward_closure_of(c, mdo)
    return mdo


cpdef max_weighted_downward_closure_of(Node n, bint logspace=False, set mdo=None):
    if mdo is None:
        mdo = set()
    mdo.add(n)
    if not n.has_children_attr():
        return mdo
    children = [n.max_weighted_child(logspace)] if n.is_sum() else n.children
    for c in children:
        if c in mdo:
            continue
        max_weighted_downward_closure_of(c, logspace, mdo)
    return mdo

cpdef max_weighted_downward_closure_edges_of(Node n, bint logspace=False, set mdo=None):
    if mdo is None:
        mdo = set()
    if n.is_distribution():
        mdo.add((n, n.rv.value))
        return mdo
    cindexes = [n.max_weighted_child_index(logspace)] if n.is_sum() else range(len(n.children))
    for ci in cindexes:
        if (n, ci) in mdo:
            continue
        mdo.add((n, ci))
        max_weighted_downward_closure_edges_of(n.children[ci], logspace, mdo)
    return mdo

cpdef max_weighted_downward_closure_edges_of_v2(Node n, bint logspace=False, dict mdo=None):
    if mdo is None:
        mdo = {}
    if n in mdo:
        return mdo
    if n.is_distribution():
        mdo[n] = n.rv.value
    elif n.is_sum():
        i = n.max_weighted_child_index(logspace)
        mdo[n] = float(i)
        max_weighted_downward_closure_edges_of_v2(n.children[i], logspace, mdo)
    elif n.is_prod():
        mdo[n] = -1.0
        for ci in range(len(n.children)):
            max_weighted_downward_closure_edges_of_v2(n.children[ci], logspace, mdo)
    return mdo

def has_cycle(Node n, dict marks=None):
    ''' Return true if directed graph, rooted at node n, has a cycle; otherwise return False.

        Inspired from www.cs.berkeley.edu/~kamil/teaching/sp03/041403.pdf
    '''
    if marks is None:
        marks = dict()
    marks[n] = 1
    for c in n.children_iter():
        if c in marks:
            if marks[c] == 1:
                return True
        elif has_cycle(c, marks):
            return True
    marks[n] = 2
    return False




def independent(vrs):
    pn = ProdNode()
    for v in vrs:
        assert v.is_rv()
        # TODO: the following line creates a CategoricalNode over
        # the variable if the variable is discrete and a GaussianNode
        # if it is continuous. This is a problem. It means we do not
        # have the flexibility to add a ChebyshevNode instead of a
        # Gaussian. And if the discrete variable is over all natural
        # numbers or over all integers then putting a CategoricalNode
        # over it will fail since a CategoricalNode requires it to
        # have a finite number of values.
        dn = CategoricalNode() if v.is_discrete() else GaussianNode()
        dn.set_rv(v)
        pn.add_child(dn)
    return RootedDAG(pn)

def simple(vrs):
    # The TODO in the independent() function applies here as well...
    sn = SumNode()
    pns = [ProdNode() for i in range(2)]
    for v in vrs:
        assert v.is_rv()
        for pn in pns:
            dn = CategoricalNode() if v.is_discrete() else GaussianNode()
            dn.set_rv(v)
            pn.add_child(dn)
    for pn in pns:
        sn.add_child(pn)
    return RootedDAG(sn)



cdef class RootedDAG:

    def __cinit__(self):
        self.root = None

    def __init__(self, Node root not None):
        assert not root.is_rv()
        assert not has_cycle(root)
        self.root = root

    def parents(self):
        pars = {}
        for n in self.pot():
            pars[n] = []
            for c in n.children_iter():
                pars[c].append(n)
        return pars

    def topo(self, nodes):
        return topo_sort(nodes)

    def rtopo(self, nodes):
        return rtopo_sort(nodes, self.parents())
    
    def pot(self, n=None):
        return pot_to(self.root if n is None else n)

    def rpot(self, Node n not None):
        return rpot_to(n, self.parents())

    def max_dc(self, n=None):
        ''' Return the max downward closure. '''
        return max_downward_closure_of(self.root if n is None else n)
    
    def max_weighted_dc(self, n=None):
        ''' Return the max weighted downward closure. '''
        return max_weighted_downward_closure_of(self.root if n is None else n)

    def max_weighted_dc_log(self, n=None):
        ''' Return the max weighted downward closure using log values. '''
        return max_weighted_downward_closure_of(self.root if n is None else n, True)

    def max_weighted_dc_edges(self, n=None):
        ''' Return the max weighted downward closure edges. '''
        return max_weighted_downward_closure_edges_of(self.root if n is None else n)

    def max_weighted_dc_edges_log(self, n=None):
        ''' Return the max weighted downward closure edges using log values. '''
        return max_weighted_downward_closure_edges_of(self.root if n is None else n, True)

    def max_weighted_dc_edges_v2(self, n=None):
        ''' Return the max weighted downward closure edges. '''
        return max_weighted_downward_closure_edges_of_v2(self.root if n is None else n)

    def max_weighted_dc_edges_v2_log(self, n=None):
        ''' Return the max weighted downward closure edges using log values. '''
        return max_weighted_downward_closure_edges_of_v2(self.root if n is None else n, True)



