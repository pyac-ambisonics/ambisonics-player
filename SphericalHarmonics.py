# a class that handles the conversion of HRTF data into spherical harmonics

import numpy as np
import pyfar as pf
import spharpy as sh
import pooch

class SphericalHarmonics:

    # Constructor
    def __init__(self, hrtf=None, ambi_file=None):
        # this will not work yet, work out the kinks
        # # set the hrtf. this should be a pyfar signal!
        # self.hrtf = hrtf
        # # get sampling rate from HRTF files
        # self.sampling_rate = hrtf.sampling_rate
        # # set the ambisonics file
        # self.ambi_file = ambi_file
        # # set the ambisonics order we are calcualting for
        # self.ambi_order = ambi_file.get_ambi_order()
        # # set up the ambisonics matrix
        # self.ambi_matrix = self.matrix_from_ambi(ambi_file)

        # load FABIAN from the web
        self.hrirs, sources = self.load_hrtf_from_web()
        # store the sources in a Sampling Sphere
        self.sources = sh.SamplingSphere.from_coordinates(sources)

        # create a spherical harmonics definition
        self.n_max = 16
        self.sh_definition = sh.SphericalHarmonicDefinition(self.n_max, normalization="SN3D")

        # create the spherical harmonics object from definition and sampling sphere
        self.spherical_harmonics = sh.SphericalHarmonics.from_definition(self.sh_definition, self.sources, inverse_method="pseudo_inverse")

        # create h_nm matrix 
        hrirs_nm = (self.spherical_harmonics.basis_inv @ self.hrirs).T
        self.hrirs_nm = sh.SphericalHarmonicSignal.from_definition(self.sh_definition, hrirs_nm.time, hrirs_nm.sampling_rate)

        # prepare a rotation matrix, with all angles 0 currently
        angles = [0, 0, 0]
        self.rotation = sh.transforms.SphericalHarmonicRotation.from_euler('xyz', np.deg2rad(angles))

    # does nothing as of yet
    def matrix_from_ambi(self, ambi_file):
        # TODO: get the shape from the ambisoncis file
        matrix = np.zeros(shape=(ambi_file.shape))
        return matrix
    
    # loads HRTFs from the internet
    def load_hrtf_from_web(self):
        # Leave this as it is: This is the URL from which the data will be downloaded
        # and a hash for checking if the download worked.
        url = 'https://github.com/pyfar/files/raw/refs/heads/main/education/VAR_TUB/FABIAN_HRIR_measured_HATO_0.sofa?download='
        hash = '83ebbcd9a09d17679b95d201c9775438c0bb1199d565c3fc7a25448a905cdc3c'

        file = pooch.retrieve(
            url, hash, fname='FABIAN_HRIR_measured_HATO_0.sofa', path=None)

        # load HRIRs and source positions
        hrirs, sources, _ = pf.io.read_sofa(file)
        return hrirs, sources
    
    # set the current rotation angle
    def set_rotation(self, angles):
        self.rotation = sh.transforms.SphericalHarmonicRotation.from_euler('xyz', np.deg2rad(angles))

    # apply a rotation and return them as signal
    def apply_rotation(self):
        # creates rotated HRIRs matrix
        hrirs_nm_rotated = self.rotation.apply(self.hrirs_nm)

        # interpolate the HRIRs by matrix mulitplication
        hrirs_interpolated_rotated = pf.matrix_multiplication(
            (self.spherical_harmonics.basis, hrirs_nm_rotated), domain='time', axes=[(0, 1), (1, 0), (0, 1)])
        hrirs_interpolated_rotated = pf.Signal(hrirs_interpolated_rotated, self.hrirs.sampling_rate)
        return hrirs_interpolated_rotated
