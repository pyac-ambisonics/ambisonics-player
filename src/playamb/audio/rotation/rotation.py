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
    Load and manage Head-Related Transfer Function (HRTF) data with preprocessing.

    This class handles complete HRTF dataset management: loads HRIR/HRTF data 
    from SOFA files (with fallback to web download), supports resampling, 
    applies headphone compensation filters, and provides diffuse-field 
    equalization. Maintains both original unmodified HRIRs and a working 
    copy that can be modified by filters.

    Parameters
    ----------
    path : str, pathlib.Path, or None, optional
        Path to a SOFA file containing HRTF/HRIR data. If None, uses the
        default FABIAN HRTF dataset (specified by DEFAULT_HRTF_FILE).
        If loading fails locally, attempts web download. Default is None.

    Attributes
    ----------
    path : pathlib.Path or None
        Resolved filesystem path to the loaded SOFA file. None if path
        could not be resolved.
    hrirs : pf.Signal
        Working HRIR signals (may be modified by filters).
        Shape: (n_directions, n_channels).
    hrirs_linear : pf.Signal
        Original unmodified HRIR signals (never modified).
        Use reset_hrirs() to restore hrirs to this state.
    sources : np.ndarray
        Spatial source coordinates (direction/elevation) of HRIR measurements.
    fs : int
        Current sampling rate in Hz.
    hp_list : list of str
        Available headphone filter names (FABIAN dataset only).
        Always includes "Diffuse Field Equalization".
    current_filter : str or None
        Name of currently applied headphone filter (None if none active).
    resources : pathlib.Path
        Path to the application's resources directory.
    hp_dir : pathlib.Path
        Path to the Headphones subdirectory containing filter definitions.
    app_dir : pathlib.Path
        Root application directory.

    Methods
    -------
    resample(fs, truncate=True)
        Resample HRIRs to a new sampling rate.
    load_hp_filter(name, min_phase=True, n_samples=512)
        Load and apply a headphone compensation filter.
    reset_hrirs()
        Restore HRIRs to unmodified state.
    apply_dfe()
        Apply Diffuse Field Equalization.
    get_IR_length(linear=False)
        Get impulse-response length in samples.

    Notes
    -----
    - FABIAN HRTF dataset is used by default
    - Headphone filters are only available if using FABIAN dataset
    - self.hrirs_linear is always preserved; modifications affect only self.hrirs
    - Resampling with truncate=True is recommended for real-time performance
    - DFE and minimum-phase conversions are available for perceptual optimization

    Examples
    --------
    >>> hrtf = HRTF()  # Load default FABIAN
    >>> hrtf.resample(48000)
    >>> hrtf.load_hp_filter("AKG K701")
    >>> ir_length = hrtf.get_IR_length()
    """

    def __init__(self, path='resources/rotation_matrices/', file='small_d_matrices_0.5deg.npz'):
        """
        Initialize rotation matrix calculator by loading pre-computed Wigner small-d matrices.
        
        Loads a .npz file containing Wigner small-d matrices for SH orders 0–7 and
        associated beta angle values. These matrices are queried during real Wigner-D
        matrix computation to accelerate rotations.

        Parameters
        ----------
        path : str or pathlib.Path, optional
            Directory containing the .npz file. Default is 'resources/rotation_matrices/'.
        file : str, optional
            Filename of the .npz archive containing pre-computed matrices.
            Default is 'small_d_matrices_0.5deg.npz'.

        Attributes
        ----------
        small_d : dict
            Dictionary mapping SH order (int) to small-d matrix arrays.
        betas_rad : np.ndarray
            Beta angle values (radians) for which small-d matrices were pre-computed.

        Raises
        ------
        FileNotFoundError
            If the .npz file cannot be found or loaded.

        Notes
        -----
        - Pre-computed matrices use 0.5° resolution by default
        - Coverage: beta ∈ [-180°, 180°]
        - Orders 0–7 support up to 3rd-order Ambisonics and beyond
        """
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
        dict
            Dictionary mapping SH order (int) to small-d matrix arrays.
            Each value has shape (n_beta_angles, 2*order+1, 2*order+1).
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
        """
        Compute the phase factor Phi(m, angle) for real Wigner-D matrix construction.
        
        Vectorized computation of the phase function used in real SH rotation:
        - Phi(m=0, angle) = 1
        - Phi(m>0, angle) = sqrt(2) * cos(m * angle)
        - Phi(m<0, angle) = -sqrt(2) * sin(|m| * angle)

        Parameters
        ----------
        m : np.ndarray
            Magnetic quantum number(s), can be positive, negative, or zero.
        angle : float
            Angle in radians (alpha or gamma in Euler Z-Y-Z convention).

        Returns
        -------
        np.ndarray
            Phase factor values matching the shape of m.

        Notes
        -----
        - Vectorized for efficient computation over all m values at once
        - Used internally by real_wigner_d_matrix()
        - Part of Blanco's real Wigner-D formula implementation
        """
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
        Query the nearest pre-computed small-d matrix for a given order and angle.
        
        Uses binary search to find the pre-computed beta value closest to the
        requested angle, then returns the corresponding small-d matrix. This
        avoids expensive re-computation during rotation.

        Parameters
        ----------
        N : int
            SH order (0–7) for which to retrieve the small-d matrix.
        beta : float
            Rotation angle (radians) for which the nearest pre-computed 
            matrix is desired.

        Returns
        -------
        np.ndarray
            Small-d matrix with shape (2*N+1, 2*N+1).

        Notes
        -----
        - Uses binary search for O(log n) lookup time
        - Selects the nearest pre-computed beta (no interpolation)
        - Handles edge cases: beta < min_beta or beta > max_beta
        - Default pre-computation uses 0.5° resolution
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
        Pre-compute and cache Wigner small-d matrices for all beta angles and SH orders.

        Generates a comprehensive .npz cache file containing small-d matrices at
        regularly-spaced beta angles for orders 0 through sh_order. This is a
        one-time computation; the resulting file enables O(1) matrix lookups
        during real-time rotation operations.

        Parameters
        ----------
        file : pathlib.Path
            Output filepath for the compressed .npz cache. Must have write permissions.
        step : float, optional
            Angular resolution in degrees of pre-computed beta values.
            Minimum 0.1°. Default is 0.5° (721 angles from -180° to 180°).
        sh_order : int, optional
            Maximum SH order to compute matrices for (0 to sh_order inclusive).
            Default is 7 (supports 3rd-order Ambisonics and higher).

        Returns
        -------
        None
            Writes compressed .npz file to disk.

        Notes
        -----
        - Computation may take 1–10 seconds depending on step size and order
        - Output file size: ~50–100 MB for 0.5° resolution, order 7
        - Beta angle range: [-180°, 180°] (equivalent to [0°, 360°])
        - Call this method once; load result via __init__() thereafter
        - Pre-computed matrices are exact; no interpolation used during lookup

        Examples
        --------
        >>> from pathlib import Path
        >>> cache_file = Path('small_d_matrices_0.5deg.npz')
        >>> RotationMatrix.create_small_d_matrices_file(cache_file, step=0.5, sh_order=7)
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
