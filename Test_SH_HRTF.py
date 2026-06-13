from SphericalHarmonics import SphericalHarmonics
from HRTF import HRTF
from ambisonics_file_English import AmbisonicsFile
import pyfar as pf
import numpy as np
import time
import shroom as ps

def main():
    print("Loading ambi file")
    start = time.time()
    # ambi_file, ambi_order = load_ambi_file()
    ambi_file = AmbisonicsFile("Ambisonics_Noise_3rd_order_noise_dir.wav", chunk_size=256)
    ambi_order = ambi_file.get_order()
    print(f"Loading ambi file took {time.time() - start:.4f} seconds\n")

    print("Load HRTFs")
    start = time.time()
    hrtf = HRTF("FABIAN_HRIR_measured_HATO_0.sofa")
    print(f"Loading HRTF took {time.time() - start:.4f} seconds\n")

    print("Creating Spherical Harmonics")
    start = time.time()
    harmonics = SphericalHarmonics(hrtf=hrtf, sampling_rate=ambi_file.get_samplerate(), ambi_order=ambi_order)
    print(f"{harmonics.ambi_order=}")
    print(f"{harmonics.hrirs_nm.shape=}")
    print(f"Creating SH took {time.time() - start:.4f} seconds\n")


    # print("Apply HRTF pyfar")
    # start = time.time()
    # stereo2 = overlap_add(ambi_file, harmonics, gain=0.5)
    # #stereo = harmonics.apply_hrtf(ambi_signal=ambi_file, gain=0.5)
    # print(f"Applying HRTF took {time.time() - start:.4f} seconds\n")

    
    print("Apply HRTF numpy")
    start = time.time()
    stereo = overlap_add_fast(ambi_file, harmonics, gain=0.5)
    #stereo = harmonics.apply_hrtf(ambi_signal=ambi_file, gain=0.5)
    print(f"Applying HRTF took {time.time() - start:.4f} seconds\n")

    print("Write binaural audio file")
    start = time.time()
    pf.io.write_audio(stereo, "written_binaural_fast_oa.wav")
    print(f"Writing file took {time.time() - start:.2f} seconds\n")

def overlap_add(ambi_file: AmbisonicsFile, sh: SphericalHarmonics, gain=1.):

    # Check channel count by comparing the channel shape
    # we know the channel shape for sh_hrir is (2, channels)
    *_, sh_hrir_ch = sh.hrirs_nm.cshape
    if ambi_file.get_num_channels() != sh_hrir_ch:
        raise ValueError("Channel counts must match (16 for 3rd order).")
    
    
    if gain > 1. or gain < 0:
        raise AttributeError("The gain must be in range [0., 1.].")
    
    # current impulse response length M
    sh_length = sh.get_IR_length()
    # block size L
    block_size = ambi_file.get_chunk_size()
    # compute N >= M + L - 1
    N = next_power_of_two(sh_length + block_size - 1)
    # amount we need to pad our HRTF to
    required_pad = block_size - sh_length

    # allocate buffers
    overlap_buffer = np.zeros((2, sh_length - 1), dtype=np.float32)
    output_length = ambi_file.total_frames + block_size
    output_buffer = np.zeros((2, output_length), dtype=np.float32)
    pos = 0

    # core loop to process each chunk
    while True:
        chunk, end_of_file = ambi_file.get_next_chunk()

        # break if our chunk is None
        if chunk is None:
            break

        # zero pad and make a signal
        pad_width = N - len(chunk[0])
        chunk_pad = np.pad(chunk, ((0,pad_width),(0, 0)))
        chunk_sig = pf.Signal(chunk_pad.T, ambi_file.get_samplerate())

        # apply hrtf
        stereo = sh.apply_hrtf(chunk_sig, gain, pad=True, pad_length=required_pad)
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


def overlap_add_fast(ambi_file: AmbisonicsFile, sh: SphericalHarmonics, gain=1.):

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
    # amount we need to pad our HRTF to
    required_pad = block_size - sh_length

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
    # make use of bitshifts to quickly calculate the enxt power of two
    return 1 << (n - 1).bit_length()

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