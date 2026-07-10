"""
A script that generates a set of rotation matrices for orders 1-7. These matrices can then be loaded
and hardswapped to make the rotation faster and make realtime computation achievable.
"""

import numpy as np
import shroom.utils.rotation_utils as rot_utils
from pathlib import Path
from utils import resolve_path

class RotationMatrix:
    """
    A class prividing functionality to calculate Wigner-D rotation matrices. The class gives functionality to precalculate a set of
    Wigner small-d matrices, save them to a file, load that file and calculate Wigner-D matrices based on the rpecalculated small-d matrices.
    
    Copyright (c) 2026 Kylan Klein Lenderink

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files 
    (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, 
    publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do 
    so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
    OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
    LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
    IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
    """

    def __init__(self, path='resources/rotation_matrices/', file='small_d_matrices_0.5deg.npz'):
        # our resources folder
        p = Path(path)
        # filename
        f = p / file

        # load the small d dictionary
        self.small_d = self._load_small_d(resolve_path(f))

        # build all unitary transofrmation matrices as well as their Hermitian transpose for query later
        self.unitary = {}
        self.unitary_H = {}
        for order in range(8):
            self.unitary[order] = self.build_complex_to_real_transform(order)
            # conjugate transpose -> Hermitian transpose
            self.unitary_H[order] = self.unitary[order].conj().T


    
    def _load_small_d(self, file: Path):
        """
        Load a .npz file containing precalculated wigner small-d matrices.

        Parameters
        ----------
        file : Path
            The Path to the file to be loaded.

        Returns
        ----------
        d : dict
            A dictionary containing all preloaded small-d matrices.
            Key (int) corresponds to SH order.
            Value is the small-d Matrix.
        """
        # load file
        loaded = np.load(file=file)

        # convert NpzFile to Dictionary
        d = {}
        for order in loaded.keys():
            d[int(order)] = loaded[order]

        # close file
        loaded.close()

        return d
    
    def real_wigner_d_matrix(self, N: int, alpha: float, beta: float, gamma: float) -> np.ndarray:
        """
        Compute the real-valued Wigner-D matrix for real Spherical Harmonics.

        Parameters
        ----------
        N : int
            Maximum SH order.
        alpha, beta, gamma : float
            Euler angles in radians (Z-Y-Z convention).
            Rotation R = Rz(alpha) * Ry(beta) * Rz(gamma).

        Returns
        -------
        D : np.ndarray
            real Wigner-D matrix of shape ((N+1)^2, (N+1)^2).
            Block diagonal structure with blocks of size (2n+1)x(2n+1).
        """
        # Compute the complex matrix 
        D_complex = self.wigner_d_matrix(N, alpha, beta, gamma) 
        
        # Transform to the real SH basis
        # D_real = U @ D_complex @ U^H
        D_real = self.unitary[N] @ D_complex @ self.unitary_H[N]
        
        # Clean up numerical artifacts (should be purely real)
        return np.real(D_real)

    def wigner_d_matrix(self, N: int, alpha: float, beta: float, gamma: float):
        """
        Compute the Wigner-D matrix for Spherical Harmonics rotation.
        This implementation is taken from @Yhonatangayers implementation
        in the pyshroom package. It has been adapted for use with precalculated
        small_d matrices as per the terms of the MIT Licence, which the pyshroom
        package is licensed udner.


        The matrix D rotates SH coefficients such that:
        f_rot(omega) = f(R^-1 omega)
        c_rot = D(R) @ c

        Parameters
        ----------
        N : int
            Maximum SH order.
        alpha, beta, gamma : float
            Euler angles in radians (Z-Y-Z convention).
            Rotation R = Rz(alpha) * Ry(beta) * Rz(gamma).

        Returns
        -------
        D : np.ndarray
            Wigner-D matrix of shape ((N+1)^2, (N+1)^2).
            Block diagonal structure with blocks of size (2n+1)x(2n+1).
        """
        # Total number of coefficients
        L = (N + 1) ** 2
        D = np.zeros((L, L), dtype=np.complex128)

        # Compute for each order n
        for n in range(N + 1):
            # Get the small-d matrix for this order
            d_n = self._query_small_d(n, beta)

            # Construct the full D matrix for this order
            # D^n_{m',m} = e^{-i m' alpha} * d^n_{m',m}(beta) * e^{-i m gamma}

            m_range = np.arange(-n, n + 1)

            # Phase terms
            # exp(-i * m' * alpha)  [rows]
            phase_left = np.exp(-1j * m_range * alpha)

            # exp(-i * m * gamma)   [cols]
            phase_right = np.exp(-1j * m_range * gamma)

            # Combine: D = diag(phase_left) @ d @ diag(phase_right)
            # Broadcasting: (2n+1, 1) * (2n+1, 2n+1) * (1, 2n+1)
            D_n = phase_left[:, np.newaxis] * d_n * phase_right[np.newaxis, :]

            # Place in the big matrix
            start_idx = n**2
            end_idx = (n + 1) ** 2
            D[start_idx:end_idx, start_idx:end_idx] = D_n

        return D

    def build_complex_to_real_transform(self, N: int) -> np.ndarray:
        """
        Build the unitary transformation matrix U that maps complex SH coefficients
        (ordered -n,...,n) to real SH coefficients (ordered -n,...,n).

        For a given order n and degree m > 0:
            Y_real^{+m} = 1/sqrt(2) * (Y^{-m} + (-1)^m Y^m)
            Y_real^{-m} = i/sqrt(2) * (Y^{-m} - (-1)^m Y^m)
            Y_real^{0}   = Y^0

        Returns
        -------
        U : np.ndarray
            Shape ((N+1)^2, (N+1)^2), complex unitary.
            c_real = U @ c_complex
        """
        L = (N + 1) ** 2
        U = np.zeros((L, L), dtype=np.complex128)

        # tracks the starting index for each order block (0, 1, 4, 9, ...)
        index = 0  
        for n in range(N + 1):
            block_size = 2 * n + 1
            U_block = np.zeros((block_size, block_size), dtype=np.complex128)
            
            for m in range(-n, n + 1):
                # column index in the complex block (0 to 2n)
                column = m + n
                
                if m == 0:
                    # row index for m=0 in the real block (center)
                    row = n
                    U_block[row, column] = 1.0
                elif m > 0:
                    # Map complex Y^{-m} (col idx c_minus) and Y^{m} (col idx c_plus)
                    c_minus = (-m) + n
                    c_plus = m + n
                    
                    # Real Y^{+m}
                    r_plus = n + m
                    factor = 1.0 / np.sqrt(2.0)
                    U_block[r_plus, c_minus] = factor
                    U_block[r_plus, c_plus] = factor * ((-1) ** m)
                    
                    # Real Y^{-m}
                    r_minus = n - m
                    U_block[r_minus, c_minus] = 1j * factor
                    U_block[r_minus, c_plus] = -1j * factor * ((-1) ** m)
            
            # Place the block into the full matrix
            U[index:index + block_size, index:index + block_size] = U_block
            index += block_size
        
        return U

    def _query_small_d(self, N: int, beta: float) -> np.ndarray:
        """
        Gets the small_d value from the precalculated and loaded file. The beta value closest to the 
        next available precalculated beta is used.

        Parameters
        ----------
        N : int
            The ambisonics/SH order for the queried wigner small-d matrix.
        beta : float
            The Euler beta angle in radians.

        Returns
        ---------
        d : np.NDArray 
            The wigner small-d array approximately corresponding to the given beta value.
        """
        # local copy of the small_d matrix corresponding to this order
        sd = self.small_d[N]

        # Extract the keys (shape (N,))
        keys = sd[:, 0, 0]
        # Binary search for insertion point
        pos = keys.searchsorted(beta)

        # take care of the edge cases
        n = len(keys)
        if pos == 0:
            return sd[pos]
        if pos == n:
            return sd[n - 1]

        # Compare distances to left and right neighbours and return index of the closest
        if beta - keys[pos - 1] <= keys[pos] - beta:
            return sd[pos - 1]
        else:
            return sd[pos]

    @staticmethod
    def create_small_d_matrices_file(file: Path, step=0.5, sh_order=7):
        """
        This function creates a set of Wigner small-d matrices for all possible beta values (0-360°) per order up to 
        the given order sh_order (7 is default). The stepsize of the betavalue can be changed. A valid filepath must be given.

        Parameters
        ----------
        file : Path
            Path to the file to write the small-d amtrices to
        step : float, optional
            Stepsize in degrees of the betavalues. Default is 0.5
        sh_order : int, optional
            The maximum SH/Ambisonics order to calculate small-d matrices for. Default is 7.
        """
        # create angles with specific step size
        # step can never be smaller than 1 degree
        if step < 0.1:
            step = 0.1
        angles = np.arange(0, 360.1, step)
        angles_rad = np.deg2rad(angles)
        length = len(angles)

        # dictionary that stores key-value pair (order-small_d)
        results = {}

        for order in range(sh_order+1):
            # size of small D matrix
            dim = 2 * order + 1
            # prepare empty array with shape (order, len(beta), dim, dim). small d is real valued!
            d = np.empty((length, dim, dim), dtype=np.float64)

            for i, beta in enumerate(angles_rad):
                d[i] = rot_utils._wigner_small_d(order, beta)

            results[str(order)] = d

        # save the file
        np.savez_compressed(file, allow_pickle=False, **results)
