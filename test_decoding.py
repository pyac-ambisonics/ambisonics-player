from spherical import SphericalHarmonics
from hrtf import HRTF
from ambisonics_file_English import AmbisonicsFile
import pyfar as pf
import numpy as np
import time

def main():
    """
    Run an end-to-end demo: load an Ambisonics file, load HRTFs, build spherical harmonics, render binaural audio, and write output.
    This entry-point performs file I/O, timing prints and invokes the rendering pipeline.
    Exceptions from underlying I/O and processing functions are propagated to the caller.
    """

    print("Loading ambi file")
    start = time.time()
    # ambi_file, ambi_order = load_ambi_file("Ambisonics_Noise_3rd_order_noise_dir.wav")
    ambi_file = AmbisonicsFile("Ambisonics_Noise_3rd_order_noise_dir.wav", chunk_size=512)
    ambi_order = ambi_file.get_order()
    print(f"Loading ambi file took {time.time() - start:.4f} seconds\n")

    print("Load HRTFs")
    start = time.time()
    hrtf = HRTF("FABIAN_HRIR_measured_HATO_0.sofa")
    hrtf.load_hp_filter("Audio-technica ATH M50x")
    print(f"Loading HRTF took {time.time() - start:.4f} seconds\n")

    print("Creating Spherical Harmonics")
    start = time.time()
    harmonics = SphericalHarmonics(hrtf=hrtf, sampling_rate=ambi_file.get_samplerate(), ambi_order=ambi_order)
    print(f"Creating SH took {time.time() - start:.4f} seconds\n")

    print(f"{hrtf.hrirs_linear.time.shape=}")
    print(f"{ambi_order=}")
    print(f"{ambi_file.get_num_channels()=}")
    print(f"{harmonics.ambi_order=}")
    print(f"{harmonics.hrirs_nm.shape=}\n")

    # print("Apply HRTF pyfar")
    # start = time.time()
    # ambi_file2, ambi_order2 = load_ambi_file()
    # stereo2 = harmonics.apply_hrtf(ambi_signal=ambi_file2, gain=0.5)
    # print(f"Applying HRTF took {time.time() - start:.4f} seconds\n")

    
    print("Apply HRTF numpy")
    start = time.time()
    stereo = overlap_add_fast(ambi_file, harmonics, gain=0.5)
    #stereo = harmonics.apply_hrtf(ambi_signal=ambi_file, gain=0.5)
    print(f"Applying HRTF took {time.time() - start:.4f} seconds\n")

    print("Write binaural audio file")
    start = time.time()
    pf.io.write_audio(stereo, "written_binaural_magls.wav")
    print(f"Writing file took {time.time() - start:.2f} seconds\n")

def overlap_add_fast(ambi_file: AmbisonicsFile, sh: SphericalHarmonics, gain=1.):
    """
    Render an Ambisonic stream to binaural using FFT-based overlap-add (frequency-domain).
    
    Parameters
    ----------
    ambi_file : AmbisonicsFile
        Reader object providing the loaded Ambisonics file.
    sh : SphericalHarmonics
        Spherical-harmonic HRTF processor.
    gain : float, optional
        Linear output gain in [0., 1.]. Default 1.0.

    Returns
    -----------
    pyfar.Signal
        Stereo time-domain pyfar.Signal containing the rendered binaural result.

    Raises
    -------------
    ValueError
        If Ambisonic channel counts do not match the HRTF channel count.
    AttributeError
        If gain is outside [0., 1.].

    Notes
    -------------
    This function updates HRIR FFTs once per block-size and performs convolution
    in the frequency domain for improved speed.
    """


    # Check channel count by comparing the channel shape
    # we know the channel shape for sh_hrir is (2, channels)
    _, sh_hrir_ch, _ = sh.hrirs_nm.shape
    if ambi_file.get_num_channels() != sh_hrir_ch:
        raise ValueError("Channel counts must match (16 for 3rd order).")
    
    if gain > 1. or gain < 0:
        raise AttributeError("The gain must be in range [0., 1.].")
    
    # current impulse response length M
    sh_length = sh.get_IR_length()
    # block size L
    block_size = ambi_file.get_chunk_size()
    # possibly validate that chunk size is a power of 2!!

    # compute N >= M + L - 1
    N = next_power_of_two(sh_length + block_size - 1)

    # update our sh fft coefficients once (and on each rotation update)
    sh.update_hrirs_fft(N)

    # allocate buffers
    overlap_buffer = np.zeros((2, sh_length - 1), dtype=np.float32)
    output_length = ambi_file.total_frames + sh_length - 1
    output_buffer = np.zeros((2, output_length), dtype=np.float32)
    pos = 0

    # core loop to process each chunk
    while True:
        chunk, end_of_file = ambi_file.get_next_chunk()

        # update our block size
        block_size, *_ = chunk.shape
        # break if our chunk is None
        if chunk is None:
            break

        # apply hrtf
        stereo = sh.apply_hrtf_fast(chunk, N, gain)
        # add overlap to stereo output
        stereo[:,:sh_length-1] += overlap_buffer
        # add stereo to output buffer
        output_buffer[:, pos:pos + block_size] = stereo[:,:block_size]
        # save new overlap buffer
        overlap_buffer = stereo[:, block_size:block_size + sh_length - 1]
        #update position
        pos += block_size
        
        # escape the loop if we reacehd the end of the file
        if end_of_file:
            # write the last overlap_buffer to our output
            output_buffer[:, pos:pos + sh_length - 1] = overlap_buffer
            break


    # create final pyfar signal
    stereo = pf.Signal(output_buffer, 
                       sampling_rate=ambi_file.get_samplerate(), 
                       domain='time'
                       )

    return stereo  

def next_power_of_two(n: int) -> int:
    """
    Return the smallest power of two greater than or equal to n.

    Parameters
    --------------
    n : int
        Input integer (n >= 1).

    Returns
    ------------
    int
        Smallest power of two >= n.
    """
    # make use of bitshifts to quickly calculate the enxt power of two
    return 1 << (n - 1).bit_length()

def load_ambi_file(file: str):
    """
    Load the 'Ambisonics_Noise_3rd_order_noise_dir.wav' test file used for HRTF application.

    Parameters
    --------------
    file : str
        Filename to load. File must be in the working directory

    Returns
    -------
    ambi_signal : pyfar.Signal
        Loaded Ambisonic signal.
    ambi_order : int
        The Ambisonic order used by the signal (must match HRTF spherical-harmonic order).

    Raises
    ----------------
    ImportError
        If the file cannot be read or the channel count does not correspond to a valid
        Ambisonic order.
    """
    # load audio data
    ambi_signal = pf.io.read_audio(file)
    if ambi_signal is None:
        raise ImportError("Couldn't load ambisonics file.")
    # infer ambisonics order
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