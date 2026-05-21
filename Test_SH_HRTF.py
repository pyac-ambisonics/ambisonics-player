from SphericalHarmonics import SphericalHarmonics
import pyfar as pf
import numpy as np
import time

def main():
    print("Loading ambi file")
    start = time.time()
    ambi_file, ambi_order = load_ambi_file()
    print(f"Loading ambi file took {time.time() - start:.4f} seconds\n")

    print("Creating Spherical Harmonics")
    start = time.time()
    harmonics = SphericalHarmonics(sampling_rate=ambi_file.sampling_rate, ambi_order=ambi_order)
    print(f"Creating SH took {time.time() - start:.4f} seconds\n")

    print("Apply HRTF")
    start = time.time()
    stereo = harmonics.apply_hrtf(ambi_signal=ambi_file, pre_gain=0.5)
    print(f"Applying HRTF took {time.time() - start:.4f} seconds\n")

    print("Write binaural audio file")
    start = time.time()
    pf.io.write_audio(stereo, "written_binaural.wav")
    print(f"Writing file took {time.time() - start:.2f} seconds\n")

def load_ambi_file():
    """
    Load the 'Ambisonics_Noise_3rd_order_noise_dir.wav' test file used for HRTF application.

    Returns
    -------
    ambi_signal : pyfar.Signal
        Loaded Ambisonic signal.
    ambi_order : int
        The Ambisonic order used by the signal (must match HRTF spherical-harmonic order).
    """
    # load audio data
    ambi_signal = pf.io.read_audio("Ambisonics_Noise_3rd_order_noise_dir.wav")
    channels, *_ = ambi_signal.cshape
    order = np.sqrt(channels) - 1
    if order != int(order):
        raise ImportError("The loaded Ambisonics file does not have a valid channel number, corresponding to its order. " \
                            "Probably a faulty file was loaded.")
    print(f'{order = }')
    # we know that this file is 3rd order
    return ambi_signal, int(order)

if __name__ == "__main__":
    main()