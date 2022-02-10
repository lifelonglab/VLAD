from rpy2 import robjects
from rpy2.robjects import numpy2ri

r = robjects.r
x = r['source']('models/our/cpds/lifewatch/wasserstein_test.R')
numpy2ri.activate()

r_wasserstein_dist = robjects.r['WassersteinTest']


def wassertein(sample, dist):
    try:
        return abs(r_wasserstein_dist(sample, dist)[0])
    except:
        exit()
