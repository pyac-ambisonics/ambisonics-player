from SphericalHarmonics import SphericalHarmonics
import pyfar as pf
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
    stereo = harmonics.apply_hrtf(ambi_signal=ambi_file)
    print(f"Applying HRTF took {time.time() - start:.4f} seconds\n")

    print("Write binaural audio file")
    start = time.time()
    pf.io.write_audio(stereo, "written_binaural.wav")
    print(f"Writing file took {time.time() - start:.2f} seconds\n")

def load_ambi_file():
    # load audio data
    ambi_signal = pf.io.read_audio("Ambisonics_Noise_3rd_order_noise_dir.wav")
    # we know that this file is 3rd order
    return ambi_signal, 3

if __name__ == "__main__":
    main()