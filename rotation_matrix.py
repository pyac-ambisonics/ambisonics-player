"""
A script that generates a set of rotation matrices for orders 1-7. These matrices can then be loaded
and hardswapped to make the rotation faster and make realtime computation achievable.
"""

import numpy as np
import shroom.utils.rotation_utils as rot_utils
from scipy.spatial.transform import Rotation
from pathlib import Path

class RotationMatrix:

    def __init__(self, D):
        self.D = D

    def from_angles(self, angles, order, convention='zyx'):
        return RotationMatrix(compute_wigner_D(angles, order, convention))

def compute_wigner_D(angles, order, convention='zyx'):
    """
    Directly Compute and update the internal Wigner-D matrix based on thie given Euler angles in degrees (zyx). This will block the thread!
    The updating is done in a thread-safe manner.

    Parameters
    ----------
    angles : sequence of float
        Euler angles in degrees as (z, y, x).
    convention : str
        A string identifying the used convention/order of the angles. 'zyx' is default.
    """

    rotation = Rotation.from_euler(convention, angles, degrees=True)
    alpha, beta, gamma = rotation.as_euler("zyz")
    return rot_utils.wigner_d_matrix(order, alpha, beta, gamma)

# def create_matrices(step=2):
#     # create angles with specific step size
#     # step can never be smaller than 1 degree
#     if step < 1:
#         step = 1
#     angles = np.arange(0, 361, step)
#     length = len(angles)
#     # dimension of the wigner D matrix is 64 for 7th order ((7+1)**2)
#     dim = 64
#     # prepare empty array with shape (z, y, x, dim, dim)
#     D = np.empty((length, length, length, dim, dim), dtype=np.complex128)

#     # nested loop for building all matrices
#     for i, z in enumerate(angles):
#         for j, y in enumerate(angles):
#             for k, x in enumerate(angles):
#                 D[i, j, k] = compute_wigner_D((z, y, x), 7)

#     return D

def create_small_d_matrices(step=1):
    # create angles with specific step size
    # step can never be smaller than 1 degree
    if step < 0.1:
        step = 0.1
    angles = np.arange(0, 360.1, step)
    length = len(angles)
    
    d = np.empty((8, length, dim, dim), dtype=np.float64)

    # dictionary that stores key-value pair (order-small_d)
    results = {}

    for order in range(8):
        # size of small D matrix
        dim = 2 * order + 1
        # prepare empty array with shape (order, len(beta), dim, dim). small d is real valued!
        d = np.empty((length, dim, dim), dtype=np.float64)

        for i, beta in enumerate(angles):
            d[order, i] = rot_utils._wigner_small_d(order, beta)

        results[order] = d

    return results


def main():
    # create matrices
    matrices = create_matrices()
    
    # save to resources folder
    path = Path('/resources/rotation_matrices/')
    file = path / 'matrices_7th_order_2deg.npz'

    # save that shit
    np.savez_compressed(file, **matrices)



if __name__ == "__main__":
    main()