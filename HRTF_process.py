import numpy as np
import pyfar as pf
import spharpy
from scipy.optimize import minimize

# a class making different HRTF preprocessing algorithms available

class HRTF_process:
    
    def __init__(self):
        # the different algorithms available
        self.algorithms = ['LS', 'MagLS', 'TA', 'BiMagLS']
        # the current algorithm
        self.current = 'LS'
        # the number of frequency bins we use for doing FFT
        self.n_bins = 2048

    # apply the chosen preprocessing algorithm. use MagLS as default
    def apply_preprocessing(self, hrirs, sh, algorithm='LS'):
        match algorithm:
            case 'LS':
                print("Using standard spherical harmonics processing (Least Squares)")
                return self.__ls(hrirs, sh)
            case 'MagLS':
                print("Using MagLS HRTF Preprocessing")
                return self.__mag_ls(hrirs, sh)
            case 'TA':
                print("TA not implemented yet. Using MagLS instead")
                return self.__mag_ls(hrirs, sh)
            case 'BiMagLS':
                print("BiMagLS not implemented yet. Using MagLS instead")
                return self.__mag_ls(hrirs, sh)
            case _:
                print("Using default spherical harmonics processing (Least Squares)")
                return self.__ls(hrirs, sh)

    # solves the Least Squares Problem. This means just applying the spherical Harmonics to the HRTF
    def __ls(self, hrirs: pf.Signal, sh: spharpy.SphericalHarmonics):
        print("Using LS method: Simple matrix multiplication.")
        return (sh.basis_inv @ hrirs).T

    # apply the magnitude least squares algorithm for better results for low order ambisonics
    # ramp = 0 -> no ramp. ramp = 1 -> default ramp (cutoff * sqrt(2)). ramp > 1 -> specific ramp in freq
    def __mag_ls(self, hrirs: pf.Signal, sh: spharpy.SphericalHarmonics, cutoff=3000, ramp=1):
        print("Using MagLS method.")
        # make our data frequency data
        hrirs_freq = hrirs.freq_raw.copy()
        # find corresponding frequency bin
        freq_axis = hrirs.frequencies
        freq_idx = hrirs.find_nearest_frequency(cutoff)

        # create lambda value for each frequency: possibly with a ramp up? or just 0/1?
        # with just 0/1 we might have to smooth the phase later
        alpha = np.zeros_like(freq_axis, dtype=float)
        alpha[freq_idx:] = 1

        
        # make a ramp up if needed. Use Hanning for smoothness
        if ramp > 0:
            # use default ramp of sqrt(2)
            if ramp == 1:
                ramp = cutoff * np.sqrt(2)
            # find start and stop indices. make sure they are not below 0 or above fs
            half = ramp / 2
            start = hrirs.find_nearest_frequency(cutoff - half)
            # probably unnecessary, since find_nearest_freq should always return valid indices
            # start = max(start, 0) 
            stop = hrirs.find_nearest_frequency(cutoff + half)
            # probably unnecessary, since find_nearest_freq should always return valid indices
            # stop = min(hrirs_freq.size, stop)
            # length
            length = max(stop - start, 0)
            # create a Hanning window, use the half that goes from 0 -> 1
            win = np.hanning(2*length)[length:]
            alpha[start:start+length] = win
        print("Created ramp successfully")

        # below cutoff: do simple least squares
        # do regular  least squares -> simple matrix mult
        hrirs_sh = (sh.basis_inv @ hrirs).T.freq_raw
        nm_magls = hrirs_sh.copy()
        
        sh_basis = sh.basis.copy()
        print(f"{hrirs_freq.shape=}")
        print(f"{nm_magls.shape=}")
        print(f"{sh_basis.shape=}")
        # for each ear, for each frequency, starting at the cutoff bin
        print("Solving Magnitude Least Squares for each frequency bin.")
        for ear in range(2):
            for f in range(start, freq_axis.size):
                # for each ear, all directional data, and the current frequency: find magls
                # solve the magnitude only optimization
                # above cutoff: do least squares, magnitude from original, phase from min-phase
                nm_magls[ear, :, f] = (
                    alpha[f] * self.__mag_ls_solver(Y=sh_basis, 
                                                    target=hrirs_freq[:, ear, f], 
                                                    x_prev=nm_magls[ear, :, f-1]
                                                    )
                    + (1 - alpha[f]) * hrirs_sh[ear, :, f]
                )
        
        print("Creating pyfar signal from calculated Magnitude Least Squares frequencies.")
        # create time domain by creating a pyfar Signal from frequency data
        hrirs_sh_time = pf.Signal(nm_magls, hrirs.sampling_rate, n_samples=hrirs.n_samples, domain='freq')
        return hrirs_sh_time
    
    # the core of the algorithm, this produces the actual iterative least squares evaluation
    # but its suuuuuuuuuper slow. 8 minutes for 140 frequency bins
    def __mag_ls_solver(self, Y, target, x_prev):

        x_prev = np.asarray(x_prev, dtype=complex)
        n = x_prev.size

        # stack real and imaginary parts into a signle real vector
        real_array = np.concatenate([x_prev.real, x_prev.imag])

        # loss determines the squared difference between predicted magnitude and target magnitude
        def loss(r):
            # recreate our complex array
            x = r[:n] + 1j * r[n:]
            predicted_mag = np.abs(Y @ x)
            target_mag = np.abs(target)
            return np.sum((predicted_mag - target_mag)**2)
        
        # with minimize we find the least loss in precision. this is relatively slow though
        result = minimize(loss, real_array, method='L-BFGS-B')
        result_x = result.x
        # recreate our complex values
        x_min = result_x[:n] + 1j * result_x[n:]

        # to do: implement phase alignment

        return x_min
    
    # AI generated MAG_LS solver. look at this some other time
    def mag_ls_gs(Y, target, x0=None, n_iter=50, tol=1e-8):
        """
        Alternating-projection solver for min || |Y x| - |target| ||^2
        Y: (m, n) complex matrix
        target: (m,) complex vector (we use its magnitude)
        x0: initial complex x (n,) or None
        Returns complex x (n,)
        """
        m, n = Y.shape
        if x0 is None:
            x = np.linalg.lstsq(Y, target, rcond=None)[0]    # init with LS (uses phase of target)
        else:
            x = x0.copy().astype(complex)

        mag = np.abs(target)
        # build real-augmented matrix for complex least squares:
        # [Re(Y) -Im(Y)] [Re(x)] = [Re(y)]
        # [Im(Y)  Re(Y)] [Im(x)]   [Im(y)]
        A_top = np.hstack([Y.real, -Y.imag])
        A_bot = np.hstack([Y.imag,  Y.real])
        A = np.vstack([A_top, A_bot])        # shape (2m, 2n)

        for k in range(n_iter):
            u = Y @ x                         # current complex measurements (m,)
            phases = np.exp(1j * np.angle(u)) # keep phase
            y_desired = mag * phases          # target with current phase

            b = np.concatenate([y_desired.real, y_desired.imag])
            # solve real least-squares for [Re(x); Im(x)]
            rx, *_ = np.linalg.lstsq(A, b, rcond=None)
            x_new = rx[:n] + 1j * rx[n:]

            if np.linalg.norm(x_new - x) < tol:
                return x_new
            x = x_new
        return x