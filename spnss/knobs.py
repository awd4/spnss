
max_steps = 1000

min_instances = 5

thresh_ceil = 0.99
thresh_decay = 0.1
thresh_floor = 1e-7

laplace_smooth = 0.01

cluster_penalty = 0.1


def set_knobs(msteps=100, minstances=5, decay=0.5, smooth=0.1, cpenalty=2.0):
    global max_steps, min_instances, thresh_decay, laplace_smooth, cluster_penalty
    max_steps = msteps
    min_instances = minstances
    thresh_decay = decay
    laplace_smooth = smooth
    cluster_penalty = cpenalty

def knobs_string():
    s  = 'ms ' + repr(max_steps)
    s += ' mi ' + repr(min_instances)
    s += ' tc ' + repr(thresh_ceil)
    s += ' td ' + repr(thresh_decay)
    s += ' tf ' + repr(thresh_floor)
    s += ' smooth ' + repr(laplace_smooth)
    s += ' cp ' + repr(cluster_penalty)
    return s

