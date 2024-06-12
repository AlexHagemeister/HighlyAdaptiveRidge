import numpy as np

# computing min_matrix

def compute_K(X_train, X_prime):
    K = np.empty((
        X_train.shape[0], # j: training observation
        X_prime.shape[0], # k: test observation
    ), dtype=int)
    for tr in range(X_train.shape[0]):
        for te in range(X_prime.shape[0]):
                K[tr, te] = sum(
                    2**(np.sum(
                        X_train[knot,:] <= np.minimum(X_train[tr, :], X_prime[te, :])
                    )) for knot in range(X_train.shape[0])
                )
    return K

def compute_K_half_np(X_train, X_prime):
    min_matrix = np.minimum(X_train[:, np.newaxis, :], X_prime[np.newaxis, :, :])
    
    return sum(
        2 ** np.sum(X_train[knot,:] <= min_matrix[np.newaxis,:,:,:], axis=-1)
        for knot in range(X_train.shape[0])
    )

def compute_K_np(X_train, X_prime):
    min_matrix = np.minimum(X_train[:, np.newaxis, :], X_prime[np.newaxis, :, :])
    
    comparison_sum = (X_train[:, np.newaxis, np.newaxis, :] <= min_matrix).sum(axis=-1)
    return np.sum(2**comparison_sum, axis=0)

# computing mins_train and mins_prime

def compute_K_(X_train, X_prime):
    mins_train, mins_prime = (X_train[:, np.newaxis, :] <= X for X in (X_train, X_prime))

    K = np.empty((
        X_train.shape[0], # j: training observation
        X_prime.shape[0], # k: test observation
    ), dtype=int)
    for tr in range(X_train.shape[0]):
        for te in range(X_prime.shape[0]):
                K[tr, te] = sum(
                    2**(np.sum(mins_train[knot, tr, :] & mins_prime[knot, te, :])) 
                    for knot in range(X_train.shape[0])
                )
    return K

def compute_K_half_np_(X_train, X_prime):
    mins_train, mins_prime = (X_train[:, np.newaxis, :] <= X for X in (X_train, X_prime))
    return sum(
        2 ** np.sum(mins_train[knot, :, np.newaxis, :] & mins_prime[knot, np.newaxis, :, :], axis=-1)
        for knot in range(X_train.shape[0])
    )

def compute_K_np_(X_train, X_prime):
    mins_train, mins_prime = (X_train[:, np.newaxis, :] <= X for X in (X_train, X_prime))
    
    comparison_sum = (mins_train[:, :, np.newaxis, :] & mins_prime[:, np.newaxis, :, :]).sum(axis=-1)
    return np.sum(2**comparison_sum, axis=0)
