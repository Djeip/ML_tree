import numpy as np


def show_reservations(places, queue, alg='nn', vis=True):
    """
    :param places: places in each class
    :param queue: list of places queue
    :param alg: 'nn' (NonNested) or 'sn' (StandartNesting)
    :param vis: visualisation
    """
    pl = np.array(places)
    result = [pl.copy()]
    arrow = ['      ' for _ in range(len(places))]

    if alg == 'nn':
        for q in queue:
            if vis:
                tmp = arrow.copy()
                tmp[q - 1] = '  ->  '
                result.append(tmp)
            pl[q - 1] -= 1
            pl[pl < 0] = 0
            result.append(pl.copy())
    elif alg == 'sn':
        for q in queue:
            if vis:
                tmp = arrow.copy()
                tmp[q - 1] = '  ->  '
                result.append(tmp)
            if all(pl != sorted(pl, reverse=True)) or len(set(pl)) == 1:
                pl[:] -= 1
            else:
                pl[:q] -= 1
            pl[pl < 0] = 0
            result.append(pl.copy())
    if vis:
        for line in np.transpose(result):
            print(' '.join(map(str, line)))


show_reservations([2, 3, 5], [3, 1, 2, 3, 2, 2, 1, 2, 1, 2], 'nn')
