import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import os

def PrecisionAdj(
        data : np.ndarray,
        total : float = 1e-5,
        perplexity : int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function is used to choose beta and the probability.
    shape: (n,d) -> (n,n), (n,1)
    """
    # initialize
    betas = np.ones(data.shape[0])
    u = np.log(perplexity)
    probs = np.zeros((data.shape[0], data.shape[0]))

    
    for i in range(data.shape[0]):
        # initialize
        min_beta = -np.inf
        max_beta = np.inf
        epilson = 1e-5
        ncount = 0
        # calculate the probability
        p = np.exp(-np.sum((data - data[i]) ** 2, axis=1) * betas[i])
        p[i] = 0
        p = p / np.sum(p)
        # calculate the entropy
        h = -np.sum(p * np.log(p + epilson))
        # calculate the difference
        diff = h - u
        while np.abs(diff) > total and ncount < 50:
            if diff > 0:
                min_beta = betas[i]
                if max_beta == np.inf or max_beta == -np.inf:
                    betas[i] = betas[i] * 2
                else:
                    betas[i] = (betas[i] + max_beta) / 2
            else:
                max_beta = betas[i]
                if min_beta == np.inf or min_beta == -np.inf:
                    betas[i] = betas[i] / 2
                else:
                    betas[i] = (betas[i] + min_beta) / 2
            # Recpmpute the probability
            p = np.exp(-np.sum((data - data[i]) ** 2, axis=1) * betas[i])
            p[i] = 0
            p = p / np.sum(p)
            # calculate the entropy
            h = -np.sum(p * np.log(p + epilson))
            # calculate the difference
            diff = h - u
            ncount += 1
        probs[i] = p

    return probs, betas

def pca(X, no_dims=50):
    """
    Runs PCA on the nxd array X in order to reduce its dimensionality to
    no_dims dimensions.

    Parameters
    ----------
    X : numpy.ndarray
        data input array with dimension (n,d)
    no_dims : int
        number of dimensions that PCA reduce to

    Returns
    -------
    Y : numpy.ndarray
        low-dimensional representation of input X
    """
    n, d = X.shape
    X = X - X.mean(axis=0)[None, :]
    _, M = np.linalg.eig(np.dot(X.T, X))
    Y = np.real(np.dot(X, M[:, :no_dims]))
    return Y

def tsne(
        data : np.ndarray,
        no_dims : int = 2,
        perplexity : int = 30,
        total : float = 1e-5,
        initial_momentum : float = 0.5,
        final_momentum : float = 0.8,
        eita : float = 500,
        min_gain : float = 0.01,
        iters : int = 1000
) -> np.ndarray:
    """
    This function is used to implement the t-SNE algorithm.
    shape: (n,d) -> (n,2)
    """

    # initialize
    ## probs: the probability of the data -> (n,n)
    ## betas: the beta of the data -> (n,1)
    probs, betas = PrecisionAdj(data, perplexity=perplexity,total=total)

    ## compute the pairwise distance
    P = probs + probs.T
    P = P / np.sum(P)

    # Early exaggeration
    MultiplyFactor = 4
    P = P * MultiplyFactor
    P = np.maximum(P, 1e-12)

    # initialize with PCA
    Y = pca(data, no_dims)[:, :no_dims]
    deltas = np.zeros((data.shape[0], no_dims))
    gains = np.ones((data.shape[0], no_dims))

    # start iteration
    for i in range(iters):
        # compute low-dimensional affinity
        q = 1 / (1 + np.sum((Y[:, None, :] - Y[None, :, :]) ** 2, axis=2))
        # make diagnoal to be zero
        np.fill_diagonal(q, 0)
        q_norm = q / np.sum(q)

        # compute the gradient
        dY = np.sum((P - q_norm)[:,:,None] * (Y[:, None, :] - Y[None, :, :]) * q[: ,: ,None] , axis=1)
        if i < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((dY > 0) != (deltas > 0)) + (gains * 0.8) * ((dY > 0) == (deltas > 0))
        gains[gains < min_gain] = min_gain
        deltas = momentum * deltas - eita * (gains * dY)
        Y += deltas
        if i == 100:
            P = P / MultiplyFactor
    return Y
        
if __name__ == "__main__":
    print("Run Y = tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    working_dir = os.getcwd() + "/Block4_yifei/tsne"
    os.chdir(working_dir)
    X = np.loadtxt("./mnist2500/mnist2500_X.txt")
    labels = np.loadtxt("./mnist2500/mnist2500_labels.txt")
    Y = tsne(X)
    plt.scatter(Y[:, 0], Y[:, 1], 20, labels)
    plt.savefig("./mnist_tsne.png")
        


    

    
    