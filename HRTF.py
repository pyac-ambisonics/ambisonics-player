from pathlib import Path
import pyfar as pf
import pooch

class HRTF:
    def __init__(self, path):
        # set the path for this HRTF
        try:
            self.path = Path(path)
        except Exception as e:
            print(f"Couldn't load from path: {path}. {e}")
            self.path = None

        # load HRTF from files
        self.hrirs, self.sources = self.load_HRTF(self.path)

    def load_HRTF(self, path=None):
        try:
            # load HRIRS from file
            pass
        except Exception as e:
            print(f"Couldn't load the HRTF from file. Attempting to load from web instead. Exception: {e}")
        # try to load HRTF from web
        try:
            return self.load_hrtf_from_web()
        except Exception as e:
            print(f"Couldn't load the HRTF from the web. No HRTFs are loaded. Exception: {e}") 

    # loads HRTFs from the internet
    def load_hrtf_from_web(self):
        """
        Download and load the FABIAN HRTF SOFA file.

        Returns
        -------
        hrirs : pyfar.Signal
            Loaded HRIR signals.
        sources : ndarray
            Source coordinate array associated with `hrirs`.
        """

        # Leave this as it is: This is the URL from which the data will be downloaded
        # and a hash for checking if the download worked.
        url = 'https://github.com/pyfar/files/raw/refs/heads/main/education/VAR_TUB/FABIAN_HRIR_measured_HATO_0.sofa?download='
        hash = '83ebbcd9a09d17679b95d201c9775438c0bb1199d565c3fc7a25448a905cdc3c'

        file = pooch.retrieve(
            url, hash, fname='FABIAN_HRIR_measured_HATO_0.sofa', path=None)

        # load HRIRs and source positions
        hrirs, sources, _ = pf.io.read_sofa(file)
        print("Loaded HRTF from web")
        return hrirs, sources