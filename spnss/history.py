
class NetHistory:

    def __init__(self, net, vld):
        self.net = net
        self.vld = vld
        self.ni = 0
        self.ops_hist = [None]
        self.vld_hist = [None]
        self.save_vld()

    def add_op_node_list(self, onl):
        assert self.ni == len(self.ops_hist) - 1
        self.ni += 1
        self.vh.append(None)
        self.ops_hist.append( onl )

    def undo(self):
        for op in self.ops_hist[self.ni]:
            op.undo()
        self.ni -= 1

    def redo(self):
        self.ni += 1
        for op in self.ops_hist[self.ni]:
            op.redo()

    def move_to(self, m):
        while self.ni > m:
            self.undo()
        while self.ni < m:
            self.redo()

    @property
    def vh(self):
        return self.vld_hist

    def save_vld(self):
        self.vld_hist[self.ni] = self.net.llh(self.vld)

    def best_net_index(self, i, j, best_v=float('-inf')):
        ''' Use a binary-search-like procedure to find a net with high validation likelihood. '''
        def insert_score(k):
            self.move_to(k)
            self.save_vld()
        return best_index(i, j, lambda k: self.vh[k], insert_score, best_v)


def best_index(i, j, get_score, insert_score, best_score):
    # cases 1-2: si >= sj and (sm >= si or sm < si)
    # cases 3-4: si <  sj and (sm >= sj or sm < sj)
    # case 1: test (i,m) and (m,j)
    # case 2: test (i,m)
    # case 3: test (i,m) and (m,j)
    # case 4: test (m,j)
    sh = get_score
    assert i <= j
    m = int((i + j) / 2)
    for k in [j, m, i]:
        if sh(k) is None:
            insert_score(k)
    if i == j:
        return i
    elif j == i + 1:
        return i if sh(i) >= sh(j) else j
    si, sj, sm = sh(i), sh(j), sh(m)
    best_score = max(si, sj, sm, best_score)
    #print i, m, j, si, sm, sj, '\t', best_score
    if sm > max(si, sj):
        i1 = best_index(i, m, get_score, insert_score, best_score)
        i2 = best_index(m, j, get_score, insert_score, best_score)
        return i1 if sh(i1) > sh(i2) else i2
    elif sm == si and sm >= sj:
        return best_index(i, m, get_score, insert_score, best_score)
    elif sm == sj and sm > si:
        return best_index(i, m, get_score, insert_score, best_score)
    elif best_score >= max(si, sj):
        return i if si > sj else j
    elif si >= sj:
        return best_index(i, m, get_score, insert_score, best_score)
    else:
        return best_index(m, j, get_score, insert_score, best_score)


