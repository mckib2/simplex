'''Python port of Revised Simplex Method from MATLAB.'''

import numpy as np
from scipy.optimize import OptimizeResult

def rsm(c, A, b, bfs, eps1=10e-5):
    '''Revised Simplex Method.

    Solves: minimize c @ x subject to A @ x == b and x >= 0

    Parameters
    ----------
    c : 1-D array_like
        Objective coefficients.
    A : 2-D array_like
        Equality constraint matrix.
    bfs : 1-D array_like
        Basic feasible solution.
    eps1 : float, optional
        Tolerance.

    Returns
    -------
    res : OptimizeResult
        Result object with the following fields:

        - ``x``: the optinal solution
        - ``nit``: number of iterations

    Raises
    ------
    ValueError
        If solution is unbounded.

    Notes
    -----
    Implements algorithm from [1]_.
    Python port of MATLAB script found at [2]_ from [3]_.

    References
    ----------
    .. [1] Richard B. Darst, "Introduction to Linear Programming,
           Applications and Extenstions", page 101
    .. [2] https://web.archive.org/web/20120501220155/http://www.cise.ufl.edu/research/sparse/Morgan/appendix.htm
    .. [3] Morgan, Steven S. A comparison of simplex method algorithms.
           Diss. University of Florida, 1997.
    '''

    A = np.array(A)
    c = np.array(c)
    bfs = np.array(bfs)
    b = np.array(b)

    # m : number of rows in A
    # n : number of columns in A
    # B_indices : vector of columns in A comprising the solution basis
    # V_indices : vector of columns in A not in solution basis

    _m, n = A.shape
    print(A.shape)
    B_indices = np.argwhere(bfs).flatten()
    # V_indices = np.argwhere(
    #     np.ones(n) - np.abs(np.sign(bfs))).flatten()
    V_indices = np.argwhere(np.abs(bfs) == 0).flatten()
    print(B_indices.shape)
    print(V_indices.shape)

    # rsm_nnz = zeros(5000,2);

    # Simplex method loops continuously until solution is found or
    # discovered to be impossible.

    iters = 0
    while True:
        iters += 1

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #	Step 1
        #	compute B^-1

        #	Binv		inverse of the basis (directly computed)

        Binv = np.linalg.pinv(A[:, B_indices])

        # rsm_nnz(iters,1) = nnz(A(:,B_indices));
        # rsm_nnz(iters,2) = nnz(Binv);

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #	Step 2
        #	compute d = B^-1 * b

        #	d		current solution vector

        d = Binv @ b

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Step 3/Step 4/Step 5
        # compute c_tilde = c_V - c_B * B^-1 * V

        # c_tilde : modified cost vector

        c_tilde = np.zeros(n)
        c_tilde[V_indices] = c[V_indices] - (
            c[B_indices] @ Binv @ A[:, V_indices])

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Step 6
        # compute j s.t. c_tilde[j] <= c_tilde[k] for all k in
        # V_indices
        # cj minimum cost value (negative) of non-basic columns
        # j column in A corresponding to minimum cost value

        j = np.argmin(c_tilde)
        cj = c_tilde[j]

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Step 7
        # if cj >= 0 , then we're done --
        #              return solution which is optimal

        if cj >= -eps1:
            solution = np.zeros(n)
            solution[B_indices] = d
            return OptimizeResult({
                'x': solution,
                'nit': iters,
            })

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Step 8
        # compute w = B^-1 * a[j]

        # w : relative weight (vector) of column entering the basis

        w = Binv @ A[:, j]

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Step 9
        # compute i s.t. w[i]>0 and d[i]/w[i] is a smallest positive
        # ratio
        # swap column j into basis and swap column i out of basis

        # mn : minimum of d[i]/w[i] when w[i] > 0
        # i : row corresponding to mn -- determines outgoing column
        # k : temporary storage variable

        # zz = find (w > eps1)'
        zz = np.argwhere(w > eps1).flatten()
        if not zz.size:
            raise ValueError('System is unbounded.')
        ii = np.argmin(d[zz]/w[zz])
        i = zz[ii]

        k = B_indices[i]
        B_indices[i] = j
        V_indices[j == V_indices] = k

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Step 10
        # REPEAT

if __name__ == '__main__':

    m = 3
    n = 8
    c = [0, 1, 1, 1, -2, 0, 0, 0]
    A = [
        [3, 1, 0, 0, -1, 1, 0, 0],
        [1, 1, 1, 1, 0, 0, 1, 0],
        [-3, 0, 2, 1, 5, 0, 0, 1],
    ]
    b = [1, 2, 6]
    bfs = [0, 0, 0, 0, 0, 1, 2, 6]

    res = rsm(c, A, b, bfs)
    print(res)
