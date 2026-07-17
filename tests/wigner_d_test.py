import sys
from pathlib import Path

# Add the 'src' directory to sys.path
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

from playamb import RotationMatrix
from spharpy.transforms import wigner_d_rotation_real
from spharpy.transforms import wigner_d_function
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

    return playamb_D, spharpy_D, np.allclose(playamb_D, spharpy_D, atol=0.03)

def main():
    max_order = 7
    rand_z = np.random.randint(0, 180, 10)
    rand_y = np.random.randint(0, 180, 10)
    rand_x = np.random.randint(0, 180, 10)

    for z in rand_z:
        for y in rand_y:
            for x in rand_x:
                order = 7
                # for order in range(max_order + 1):
                playmb_d, spharpy_d, validate = validate_wigner_d((z,y,x), order)
                if not validate:
                    print(f"Validation failed for ({z},{y},{x}), {order}!")
                    return
    
    print("Validation successfull!")

if __name__ == "__main__":
    main()