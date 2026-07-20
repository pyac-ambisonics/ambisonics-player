"""
A class providing utility to generate a set of rotation matrices for orders 1-7. These matrices can then be loaded
and hardswapped to make the rotation faster and make realtime computation achievable.
"""

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp

from spharpy.transforms import wigner_d_function

from playamb.utils.utils import resolve_path

class RotationMatrix:
    """
    A class prividing functionality to calculate Wigner-D rotation matrices. The class gives functionality to precalculate a set of
    Wigner small-d matrices, save them to a file, load that file and calculate Wigner-D matrices based on the precalculated small-d matrices.
    
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
        self.small_d = self._load_small_d_npz(resolve_path(f))
    
    def _load_small_d_npz(self, file: Path):
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
        self.betas_rad = loaded['betas_rad']

        # convert NpzFile to Dictionary
        d = {}
        for order in loaded.keys():
            # ignore the betas key
            if order != 'betas_rad':
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
        total = (N + 1) ** 2
        D_real = np.zeros((total, total), dtype=np.float64)

        for l in range(N + 1):
            # shape (2l+1, 2l+1)
            d_l = self._query_small_d(l, -beta)
            # m indices
            m_vals = np.arange(-l, l + 1) 

            # Create meshgrid: rows = m', cols = m
            # (size, 1)
            m_row = m_vals[:, None]
            # (1, size)
            m_col = m_vals[None, :]

            # Compute Phi(m, alpha) and Phi(m', gamma) for all m, m'
            Phi_m_alpha = self._phi(m_col, alpha)
            Phi_md_gamma = self._phi(m_row, gamma)
            Phi_neg_m_alpha = self._phi(-m_col, alpha)
            Phi_neg_md_gamma = self._phi(-m_row, gamma)

            # Absolute values for indices
            md_abs = np.abs(m_row)
            m_abs = np.abs(m_col)

            # Access small-d entries using index = value + l
            # d(l, |m|, |m'|, beta)
            d1 = d_l[md_abs + l, m_abs + l]
            # d(l, -|m'|, |m|, beta)
            d2 = d_l[m_abs + l, -md_abs + l]

            # Signs and parity
            sign_m = np.where(m_col < 0, -1, 1)
            sign_md = np.where(m_row < 0, -1, 1)
            # (-1)^m
            parity_m = np.where(m_col % 2 == 0, 1, -1)

            # Blanco formula (adapted to +beta)
            term1 = d1 + parity_m * d2
            term2 = d1 - parity_m * d2

            R_l = 0.5 * ( sign_md * Phi_m_alpha * Phi_md_gamma * term1
                        - sign_m  * Phi_neg_m_alpha * Phi_neg_md_gamma * term2 )

            # Imaginary parts should cancel; take real part to clean up
            R_l_real = np.real(R_l)

            # Place block into global matrix (order: l=0,1,2,...)
            start = l ** 2
            end = (l + 1) ** 2
            D_real[start:end, start:end] = R_l_real

        return D_real
    
    # Helper: Phi(m, angle) as defined in spharpy
    # possible to precompute!!!
    def _phi(self, m, angle):
        # vectorized version for 1D arrays
        out = np.zeros_like(m, dtype=np.float64)
        mask_pos = m > 0
        mask_zero = m == 0
        mask_neg = m < 0
        out[mask_pos] = np.sqrt(2) * np.cos(m[mask_pos] * angle)
        out[mask_zero] = 1.0
        out[mask_neg] = -np.sqrt(2) * np.sin(-m[mask_neg] * angle)
        return out

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

        # Binary search for insertion point
        pos = self.betas_rad.searchsorted(beta)

        # take care of the edge cases
        n = len(self.betas_rad)
        if pos == 0:
            return sd[pos]
        if pos == n:
            return sd[n - 1]

        # Compare distances to left and right neighbours and return index of the closest
        if beta - self.betas_rad[pos - 1] <= self.betas_rad[pos] - beta:
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
        angles = np.arange(-180, 180.1, step)
        angles_rad = np.deg2rad(angles)
        length = len(angles)

        # dictionary that stores key-value pair (order-small_d)
        results = {}
        results['betas_rad'] = angles_rad

        for order in range(sh_order+1):
            # size of small D matrix
            dim = 2 * order + 1
            # prepare empty array with shape (order, len(beta), dim, dim). small d is real valued!
            d = np.empty((length, dim, dim), dtype=np.float64)

            for i, beta in enumerate(angles_rad):
                #d[i] = rot_utils.wigner_d_function(order, beta)

                m_vals = np.arange(-order, order+1)
                #d = np.zeros((dim, dim))
                for j, m_dash in enumerate(m_vals):
                    for k, m in enumerate(m_vals):
                        d[i, j, k] = wigner_d_function(order, m_dash, m, beta)

            results[str(order)] = d

        # save the file
        np.savez_compressed(file, allow_pickle=False, **results)

    def interpolate_rotation_matrices(
        self,
        angles_old,
        angles_new,
        order,
        n_steps: int = 16,
        convention: str = "zyx",
    ) -> np.ndarray:
        """
        Interpolate between two head orientations and return a stack of
        real Wigner-D matrices for the intermediate rotations.

        Parameters
        ----------
        angles_old : sequence of float
            Previous Euler angles in degrees.
        angles_new : sequence of float
            Target Euler angles in degrees.
        n_steps : int
            Number of interpolation steps. Use this as a short fade length.
        convention : str
            Euler convention for the input angles.

        Returns
        -------
        np.ndarray
            Array with shape (n_steps, n_sh, n_sh).
        """
        if n_steps < 2:
            raise ValueError("n_steps must be >= 2")

        key_rots = Rotation.from_euler(
            convention,
            np.asarray([angles_old, angles_new], dtype=float),
            degrees=True,
        )
        slerp = Slerp([0.0, 1.0], key_rots)
        samples = slerp(np.linspace(0.0, 1.0, n_steps))

        n_sh = (order + 1) ** 2
        out = np.empty((n_steps, n_sh, n_sh), dtype=float)

        for i, rot in enumerate(samples):
            alpha, beta, gamma = rot.as_euler("zyz")
            out[i] = self.real_wigner_d_matrix(order, alpha, beta, gamma)

        return out
