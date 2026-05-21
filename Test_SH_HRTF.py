from SphericalHarmonics import SphericalHarmonics
import sounddevice as sd
import soundfile as sf
import pyfar as pf
import numpy as np

def main():
    ambi_file, ambi_order = load_ambi_file()
    print("loaded ambi file")
    harmonics = SphericalHarmonics(sampling_rate=ambi_file.sampling_rate, ambi_order=ambi_order)
    print("created harmonics")
    stereo = apply_hrtf(ambi_signal=ambi_file, sh_hrir=harmonics.apply_rotation(), sampling_rate=ambi_file.sampling_rate)
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

# apply the hrtf data to an ambisonics file
def apply_hrtf(ambi_signal, sh_hrir, sampling_rate):
    # Check channel count by comparing the channel shape
    # we know the channel shape for ambi_signal is (channels,)
    ambi_ch, *_ = ambi_signal.cshape
    # we know the channel shape for sh_hrir is (2, channels)
    *_, sh_hrir_ch = sh_hrir.cshape
    if ambi_ch != sh_hrir_ch:
        raise ValueError("Channel counts must match (16 for 3rd order).")

    # sh_hrir should have the shape (2, ambi_order)
    # Convolve each channel separately for left and right
    # instantly store it as time data
    left_conv = pf.dsp.convolve(
        ambi_signal,
        pf.Signal(sh_hrir.time[0, :, :], sampling_rate, domain='time'), # need to get the left channel here
        mode='full'
    ).time
    right_conv = pf.dsp.convolve(
        ambi_signal,
        pf.Signal(sh_hrir.time[1, :, :], sampling_rate, domain='time'), # need to get the right channel here
        mode='full'
    ).time
    
    # Sum over channels -> single‑channel binaural signals
    left_signal = np.sum(left_conv, axis=0)
    right_signal = np.sum(right_conv, axis=0)

    # do gain staging
    # Find the maximum absolute value across both channels
    peak = max(np.abs(left_signal).max(), np.abs(right_signal).max())
    # Avoid division by zero
    if peak > 0:
        gain = 0.99 / peak   # 0.99 leaves a tiny headroom
        left_signal *= gain
        right_signal *= gain


    # create stereo signal by stacking the time data horizontally
    stereo_time = np.vstack((left_signal, right_signal))
    stereo = pf.Signal(stereo_time, sampling_rate=sampling_rate, domain='time')

    return stereo

if __name__ == "__main__":
    main()