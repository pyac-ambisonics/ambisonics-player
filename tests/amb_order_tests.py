
import sys
sys.path.append('../')

from src.playamb import HRTF, SphericalHarmonics, AmbisonicsFile

import time
import soundfile as sf

def write_test_files(filepath, orders=[1, 3, 5, 7], hp_filters=["None", "Diffuse Field Equalization", "Audio-technica ATH M50x"], hp_names=None, preprocessing=["MagLS"]):

    if len(hp_filters) != len (hp_names):
        raise Exception("Invalid Input! HP Filters and HP names must be of same length!")

    ambix = AmbisonicsFile(
                    filepath,
                    chunk_size=1024,
                    order=7,
                    trim_extra_channels=True,
                    format="ambix",
                    normalization="SN3D",
                )
    
    fs = ambix.get_samplerate()

    for pp in preprocessing:
        # for each filter
        for i in range(len(hp_filters)):
            # for each order:
            for order in orders:

                print(f"Rendering File for order {order} with filter: {hp_filters[i]}")
                start = time.time()

                # set order
                ambix.set_order(order)

                # hrtf
                hrtf = HRTF()
                hrtf.load_hp_filter(hp_filters[i])

                # Sh
                sh = SphericalHarmonics(
                            hrtf=hrtf,
                            sampling_rate=fs,
                            ambi_order=order,
                            preprocess=pp
                        )
                
                # render file
                stereo = sh.apply_hrtf(ambix.get_signal_chunk(0, ambix.total_frames)).T

                # write soundfile
                if hp_names is None:
                    file = "filter_" + hp_filters[i] + "_process_" + str(pp) + "_order_" + str(order) + ".wav"
                else:
                    file =  "filter_" + hp_names[i] + "_process_" + str(pp) + "_order_" + str(order) + ".wav"
                sf.write(file, stereo, fs)

                print(f"Finished writing {file}! Time: {time.time() - start}\n")

def main():
    file = "D:/Bibliotheken/Dokumente/_Uni/02_Semester 2/Python & Akustik/Ambisonics Player/Bechet_AmbiX_7th.wav"
    filter = ["None", "Diffuse Field Equalization", "Audio-technica ATH M50x"]
    names = ["None", "DFE", "ATH-M50x"]
    orders = [0, 1, 5, 7]
    preprocess = ["LS", "MagLS"]
    write_test_files(file, orders=orders, 
                     hp_filters=filter, 
                     hp_names=names, 
                     preprocessing=preprocess
                     )

if __name__ == "__main__":
    main()
