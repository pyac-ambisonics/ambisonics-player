from gui_player_v2 import AudioPlayerGUI
from hrtf import HRTF
from spherical import SphericalHarmonics
from ambisonics_file_English import AmbisonicsFile
import time

def main():
    gui = AudioPlayerGUI()
    
    # start gui
    gui.run()


if __name__ == "__main__":
    main()