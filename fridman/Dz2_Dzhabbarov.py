from scipy import stats
import numpy as np


def littlewood(mu, std, p1, p2):
    dist = stats.norm(mu, std)
    i = 0
    while (1 - dist.cdf(i)) >= p2 / p1:
        i += 1
    return i


littlewood(9, 3, 4000, 1000)


def ESMR(mu_lst, std_lst, p_lst, cap):
    assert len(mu_lst) == len(std_lst) and len(mu_lst) + 1 == len(p_lst), 'Заданные списки имеют неправильную длину'

    res = []
    for i in range(4):
        mu = sum(mu_lst[:i + 1])
        std = np.sqrt(sum(map(lambda x: pow(x, 2), std_lst[:i + 1])))
        if i > 0:
            p2 = np.average(p_lst[:i + 1], weights=mu_lst[:i + 1])
        else:
            p2 = p_lst[i + 1]
        print(mu, std, p_lst[i], p2)
        d = littlewood(mu, std, p_lst[i], p2)
        print(d)
        res.append(d)
    return res


ESMR([5, 7, 13, 15],[3, 2, 4, 5], [7000, 5000, 3000, 1000, 500], [1])
