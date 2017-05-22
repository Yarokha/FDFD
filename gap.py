import numpy as np
import scipy as sp
import scipy.sparse.linalg as linalg
import matplotlib.pyplot as plt

a = 1
n_grid = 101
dz = a/n_grid
ratio = 0.5
eps1 = 13
eps2 = 13
eps = np.ones(n_grid)
eps[:int(ratio*n_grid)] = 1/eps1
eps[int(ratio*n_grid):] = 1/eps2
epsm = sp.sparse.dia_matrix(([eps], [0]), [n_grid, n_grid])

one = np.ones(n_grid)/dz+0j
Fd = sp.sparse.dia_matrix(([-one,one], [0,1]), [n_grid, n_grid])
Bd= sp.sparse.dia_matrix(([-one,one], [-1,0]), [n_grid, n_grid])
F = sp.sparse.lil_matrix(Fd)
B = sp.sparse.lil_matrix(Bd)

num_modes = 6

beta = 2*np.pi*np.linspace(-0.5, 0.5, 301)+0j
k = np.zeros((num_modes, beta.size), dtype=complex)

for i in range(beta.size):
    F[n_grid-1, 0] = np.exp(1j*beta[i]*a)/dz
    B[0, n_grid-1] = -np.conj(F[n_grid-1, 0])
    # EM = -np.dot(np.dot(epsv, B),F)
    EM = -epsm * B * F
    k_t = beta[i] / np.sqrt((eps1 * ratio + eps2 * (1 - ratio)))
    k2, V = linalg.eigs(EM, k=num_modes, M=None, sigma=k_t ** 2)
    k[:,i] = np.sqrt(k2)
    # k[:, i] = k2
for i in range(num_modes):
    plt.plot(beta / (2 * np.pi), np.real(k[i,:] / (2 * np.pi)), '-')
plt.xlabel("k vector")
plt.ylabel("frequency")
plt.xlim([-0.5, 0.5])
plt.ylim([0, 0.5])
plt.show()

#                   numpy
##############################################################
# epsv = np.zeros((n_grid, n_grid))
# np.fill_diagonal(epsv, eps)
# Matrix F
# F = np.zeros((n_grid, n_grid), dtype=complex)
# i, j = np.indices(F.shape)
# np.fill_diagonal(F, -1)
# F[i == j-1] = 1
# F *= 1/dz

#Matrix B
# B = np.zeros((n_grid, n_grid), dtype=complex)
# i, j = np.indices(B.shape)
# np.fill_diagonal(B, 1)
# B[i == j+1] = -1
# B *= 1/dz