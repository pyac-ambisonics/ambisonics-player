from gui_player_v2_new import AudioPlayerGUI
import cProfile
import pstats


def main():
    gui = AudioPlayerGUI()
    gui.run()


if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.sort_stats("cumtime").print_stats(30)
