from audio_player import AudioPlayer
from gui_player_v2 import AudioPlayerGUI
from hrtf import HRTF
from spherical import SphericalHarmonics
from ambisonics_file_English import AmbisonicsFile
from player import BinauralPlayer
import pyfar as pf
import numpy as np
import shroom as ps
import threading
from pathlib import Path
import os
import time

def main():
    #gui = AudioPlayerGUI(ambix_handler=)
    
    # start gui
    #gui.run()

    block_size = 512#gui.get_block_size()

    # wait for user input

    # input: file path

    # load file

    # raise error if not valid

    # wait for user input

    # on user input: start offset, paly, pause, stop

    print("Loading ambi file")
    start = time.time()
    ambi_file = AmbisonicsFile("Ambisonics_Noise_3rd_order_noise_dir.wav", chunk_size=block_size)
    ambi_order = ambi_file.get_order()
    print(f"Loading ambi file took {time.time() - start:.4f} seconds\n")

    print("Load HRTFs")
    start = time.time()
    hrtf = HRTF("FABIAN_HRIR_measured_HATO_0.sofa")
    #hrtf.load_hp_filter("Audio-technica ATH M50x")
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


    print("Start binaural player")
    start = time.time()
    player = BinauralPlayer(ambi_file=ambi_file, sh=harmonics, gain=0.5, block_size=block_size)
    player.play()

    # After a while, pause
    time.sleep(3)
    player.pause()
    time.sleep(1)
    player.resume()

    time.sleep(3)
    # Seek to 8 seconds
    player.seek_to_time(12)

    time.sleep(3)

    # Stop
    player.stop()


if __name__ == "__main__":
    main()