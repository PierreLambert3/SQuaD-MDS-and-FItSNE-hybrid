import numpy as np
from sklearn.manifold import _barnes_hut_tsne


'''
All the functions from this file where taken from sklearn's implementation of tSNE
'''


def KL_divergeance_BH(flat_X_LD, P, degrees_of_freedom, n_samples, n_components,
                      skip_num_points, compute_error, grad,
                      angle=0.75, verbose=False,  num_threads=1):
    grad.fill(0.)
    flat_X_LD = flat_X_LD.astype(np.float32, copy=False)
    X_embedded = flat_X_LD.reshape(n_samples, n_components)

    val_P = P.data.astype(np.float32, copy=False)
    neighbors = P.indices.astype(np.int64, copy=False)
    indptr = P.indptr.astype(np.int64, copy=False)
    error = _barnes_hut_tsne.gradient(val_P, X_embedded, neighbors, indptr,
                                      grad, angle, n_components, verbose,
                                      dof=degrees_of_freedom,
                                      compute_error=compute_error,
                                      num_threads=num_threads)
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad = grad.ravel()
    grad *= c
    return grad.reshape((n_samples, n_components))

def joint_P(X, PP, N_jobs=4):
    from sklearn.neighbors import NearestNeighbors
    N, dim = X.shape
    n_neighbors = min(N - 1, int(3.*PP + 1))
    knn = NearestNeighbors(algorithm='auto', n_jobs=N_jobs, n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(X)
    D_nn = knn.kneighbors_graph(mode='distance')
    D_nn.data **= 2
    del knn
    P = joint_probabilities_nn(D_nn, PP)
    return P

def joint_probabilities_nn(D, target_PP):
    from sklearn.manifold._utils import _binary_search_perplexity
    from scipy.sparse import csr_matrix
    D.sort_indices()
    N = D.shape[0]
    D_data = D.data.reshape(N, -1)
    D_data = D_data.astype(np.float32, copy=False)
    conditional_P = _binary_search_perplexity(D_data, target_PP, 0)
    assert np.all(np.isfinite(conditional_P))
    P = csr_matrix((conditional_P.ravel(), D.indices, D.indptr), shape=(N, N))
    P = P + P.T
    sum_P = np.maximum(P.sum(), np.finfo(np.double).eps)
    P /= sum_P
    assert np.all(np.abs(P.data) <= 1.0)
    return P
