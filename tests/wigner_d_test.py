import sys
from pathlib import Path

# Add the 'src' directory to sys.path
src_path = Path.cwd().parent / 'src'
sys.path.append(str(src_path))

from playamb import RotationMatrix
from spharpy.transforms import wigner_d_rotation_real
from scipy.spatial.transform import Rotation
import numpy as np

rot_mat = RotationMatrix()

def validate_wigner_d(angles, order):
    """
    Validates the playamb Wigner D calculation against the Spharpy Wigner D calculation
    """
    rotation = Rotation.from_euler('zyx', angles, degrees=True)
    # compute wigner D matrix
    alpha, beta, gamma = rotation.as_euler("zyz")

    playamb_D = rot_mat.real_wigner_d_matrix(order, alpha, beta, gamma)

    spharpy_D = wigner_d_rotation_real(order, alpha, beta, gamma)

    return np.allclose(playamb_D, spharpy_D)


def main():
    max_order = 7
    rand_z = np.random.randint(0, 180, 30)
    rand_y = np.random.randint(0, 180, 30)
    rand_x = np.random.randint(0, 180, 30)

    for z in rand_z:
        for y in rand_y:
            for x in rand_x:
                for order in range(max_order + 1):
                    if not validate_wigner_d((z,y,x), order):
                        print(f"Validation failed for ({z},{y},{x}), {order}!")
                        return
    
    print("Validation successfull!")


if __name__ == "__main__":
    main()