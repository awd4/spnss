# distutils: language = c++
''' Connected components. '''

def components2(ctest, nnodes):
    # ctest - connections test
    # nnodes - number of nodes
    cl = [] # component list
    nodes = set(range(nnodes))
    while len(nodes) > 0:
        curr = set()    # current component
        tmp = set([sorted(list(nodes))[0]])
        nodes.difference_update(tmp)
        while len(tmp) > 0:
            n1 = tmp.pop()
            nbors = [n2 for n2 in nodes if n2 != n1 and ctest(n1, n2)] 
            tmp.update(nbors)
            nodes.difference_update(tmp)
            curr.add(n1)
        cl.append(curr)
    return cl


def components(ctest, nodes):
    # ctest - connections test
    cl = [] # component list
    ni = set(range(len(nodes))) # node indices
    while len(ni) > 0:
        curr = []   # current component
        tmp = set([sorted(list(ni))[0]])    # select a seed node (index)
        ni.difference_update(tmp)
        while len(tmp) > 0:
            i1 = tmp.pop()
            nbors = [i2 for i2 in ni if i2 != i1 and ctest(nodes[i1], nodes[i2])] 
            tmp.update(nbors)
            ni.difference_update(tmp)
            curr.append(nodes[i1])
        cl.append(curr)
    return cl
