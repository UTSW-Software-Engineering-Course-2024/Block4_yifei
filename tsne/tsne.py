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
    Choose beta and the probability based on input data.
    
    Parameters
    ----------
    data : numpy.ndarray
        input data with shape (n,d)
    total : float
        the total difference between the entropy and the perplexity
    perplexity : int
        the perplexity of the data

    Returns
    -------
    probs : numpy.ndarray
        the probability of the data with shape (n,n)
    betas : numpy.ndarray
        the beta of the data with shape (n,1)
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
    Implement the t-SNE algorithm.
    
    Parameters
    ----------
    data : numpy.ndarray
        input data with shape (n,d)
    no_dims : int
        the dimension of the low-dimensional representation
    perplexity : int
        the perplexity of the data
    total : float
        the total difference between the entropy and the perplexity
    initial_momentum : float
        the initial momentum of the gradient descent
    final_momentum : float
        the final momentum of the gradient descent
    eita : float
        the learning rate of the gradient descent
    min_gain : float
        the minimum gain of the gradient descent
    iters : int 
        the number of iterations

    Returns
    -------
    Y : numpy.ndarray
        the low-dimensional representation of the data with shape (n,no_dims)
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
    import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import os
import argparse

# ... (rest of your code remains the same)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform t-SNE on a dataset.')
    parser.add_argument('--data', type=str, required=True, help='Path to the input data file')
    parser.add_argument('--labels', type=str, help='Path to the labels file (optional)')
    parser.add_argument('--output', type=str, default='tsne_output.png', help='Path to save the output image')
    parser.add_argument('--no_dims', type=int, default=2, help='Number of dimensions for t-SNE')
    parser.add_argument('--perplexity', type=int, default=30, help='Perplexity parameter for t-SNE')
    args = parser.parse_args()

    X = np.loadtxt(args.data)
    labels = None
    if args.labels:
        labels = np.loadtxt(args.labels)

    Y = tsne(X, no_dims=args.no_dims, perplexity=args.perplexity)

    plt.figure(figsize=(8, 8))
    if labels is not None:
        plt.scatter(Y[:, 0], Y[:, 1], c=labels, cmap='viridis')
        plt.colorbar(label='Labels')
    else:
        plt.scatter(Y[:, 0], Y[:, 1])
    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"t-SNE visualization saved as {args.output}")
        


    

    
    