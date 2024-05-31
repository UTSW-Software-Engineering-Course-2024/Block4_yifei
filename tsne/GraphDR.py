import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph, eye, issparse,csr_matrix
from scipy.sparse.linalg import spsolve

def graphdr_simple(
    X: np.ndarray, 
    lambda_ : float ,
    no_rotation : bool = False,
    n_component : int = None) -> np.ndarray:
    """
    Implement the GraphDR algorithm for small dataset.
    
    Parameters
    ----------
    X : numpy.ndarray
        input data with shape (n,d)
    lambda_ : float
        the regularization parameter
    no_rotation : bool
        whether to perform rotation, default is False
    n_component : int
        the number of components to keep
        
    Returns
    -------
    Z : numpy.ndarray
        the low-dimensional representation of the data with shape (n,no_dims)
    """
    # Compute the k-nearest neighbors graph
    graph = kneighbors_graph(X, n_neighbors=105, mode='distance',metric="euclidean", include_self=False)
    graph = 0.5 * (graph + graph.T)
    # Compute the graph Laplacian
    graphL = csgraph.laplacian(graph)
    G = eye(X.shape[0]) + lambda_ * graphL
    if issparse(G):
        Ginv = np.linalg.inv(G.todense())
    else:
        Ginv = np.linalg.inv(G)
    if no_rotation:
        if n_component:
            try:
                X = X[:, :n_component]
            except:
                raise ValueError("n_component should be less than the number of features")
        Z = np.asarray(np.dot(X.T, Ginv).T)
    else:
        C = np.dot(np.dot(X.T, Ginv),X)
        _,W = np.linalg.eigh(C)
        W = np.array(W)
        W = W[:,::-1]
        if n_component:
            try:
                W = W[:, :n_component]
            except:
                raise ValueError("n_component should be less than the number of features")
            
        Z =  np.asarray(np.dot(np.dot(W.T, X.T), Ginv).T)
        
    return Z

def graphdr(
    X: np.ndarray, 
    lambda_ : float ,
    no_rotation : bool = False,
    n_component : int = None) -> np.ndarray:
    """
    Implement the GraphDR algorithm for large dataset.
    
    Parameters
    ----------
    X : numpy.ndarray
        input data with shape (n,d)
    lambda_ : float
        the regularization parameter
    no_rotation : bool
        whether to perform rotation, default is False
    n_component : int
        the number of components to keep
        
    Returns
    -------
    Z : numpy.ndarray
        the low-dimensional representation of the data with shape (n,no_dims)
    """
    # Compute the k-nearest neighbors graph
    graph = kneighbors_graph(X, n_neighbors=105, mode='distance',metric="euclidean", include_self=False)
    graph = 0.5 * (graph + graph.T)
    # Compute the graph Laplacian
    graphL = csgraph.laplacian(graph)
    G = eye(X.shape[0]) + lambda_ * graphL
    
    if not issparse(G):
        G = csr_matrix(G)
    if no_rotation:
        if n_component:
            try:
                X = X[:, :n_component]
            except:
                raise ValueError("n_component should be less than the number of features")
        # solve GZ = X
        Z = spsolve(G,X)
        
    else:
        # solve G G^-1 X = X
        G_inv_X = spsolve(G,X)
        C = np.dot(G_inv_X.T,X)
        _,W = np.linalg.eigh(C)
        W = np.array(W)
        W = W[:,::-1]
        if n_component:
            try:
                W = W[:, :n_component]
            except:
                raise ValueError("n_component should be less than the number of features")
        Z = np.dot(W.T,G_inv_X.T).T
    return Z
        
