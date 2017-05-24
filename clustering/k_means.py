# k-means clustering for 2D board datasciencelab.wordpress.com
import numpy as np
import random

def cluster_points(X, mu):
    clusters = {}
    for x in X:
        bestmukey = min([(i[0], abs(5*(x[0]-mu[i[0]][0])**2 + 5*(x[1]-mu[i[0]][1])**2)) \
    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(x)
        except KeyError:
            clusters[bestmukey] = [x]
    return clusters

def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu

def has_converged(mu, oldmu):
    return (set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu]))

def find_centers(X, K):
    # Initialize to K random centers
    oldmu = random.sample(X, K)
    mu = random.sample(X, K)

    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return(mu, clusters)

def init_board(N):
    X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(N)])
    return X

def gauss(x,mean=0.0,stddev=0.02,peak=1.0):
    return peak*np.exp(-(x-mean)**2/(2*stddev**2))
    
def init_dots(N,n_bar):
    bar_centers = np.random.rand(n_bar) 
    bar_heights = np.random.rand(n_bar)
    x = np.linspace(0,1,N)
    X = np.zeros([N,2])
    for j in range(n_bar):
        X += np.array([[x[i],gauss(x[i]-bar_centers[j],peak=bar_heights[j])] for i in range(N)])
    return X

