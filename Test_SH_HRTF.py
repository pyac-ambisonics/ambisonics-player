from SphericalHarmonics import SphericalHarmonics
import pyfar as pf

def main():
    ambi_file, ambi_order = load_ambi_file()
    print("loaded ambi file")
    harmonics = SphericalHarmonics(sampling_rate=ambi_file.sampling_rate, ambi_order=ambi_order)
    print("created harmonics")
    stereo = harmonics.apply_hrtf(ambi_signal=ambi_file)
    print("applied hrtf")
    import time
    start = time.time()
    # creates rotated HRIRs matrix
    #hrirs_nm_rotated = self.rotation_matrix @ self.hrirs_nm
    pf.io.write_audio(stereo, "written_binaural.wav")
    print(f"Writing took {time.time() - start:.2f} seconds")
    #sf.write("written_binaural.wav", (left.time, right.time), int(ambi_file.sampling_rate), 'PCM_24')
    print("wrote file")

def load_ambi_file():
    # load audio data
    ambi_signal = pf.io.read_audio("Ambisonics_Noise_3rd_order_noise_dir.wav")
    print(f'{ambi_signal.cshape = }')
    print(f'{ambi_signal.time.shape = }')
    # we know that this file is 3rd order
    return ambi_signal, 3

if __name__ == "__main__":
    main()