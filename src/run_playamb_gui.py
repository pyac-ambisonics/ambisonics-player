from playamb import AudioPlayerGUI
import os

def main():
    print(f"{os.getcwd()=}")
    gui = AudioPlayerGUI()
    gui.run()


if __name__ == "__main__":
    main()
