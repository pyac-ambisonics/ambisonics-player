from audio_player import AudioPlayer


def print_help():
    """
    Print available command-line commands.
    """

    print("\nAvailable commands:")
    print("  play              Start playback")
    print("  stop              Stop playback and reset to beginning")
    print("  volume 0.5        Set volume, range: 0.0 to 1.0")
    print("  offset 10         Set start offset in seconds")
    print("  info              Show audio information")
    print("  help              Show this help menu")
    print("  quit              Exit the program")


def main():
    player = AudioPlayer()

    wav_path = "test.wav"
    player.load_wav(wav_path)

    print_help()

    while True:
        command = input("\nEnter command: ").strip().lower()

        if command == "play":
            player.play()

        elif command == "stop":
            player.stop()

        elif command.startswith("volume"):
            parts = command.split()

            if len(parts) != 2:
                print("Usage: volume 0.5")
                continue

            try:
                volume = float(parts[1])
                player.set_volume(volume)
            except ValueError:
                print("Invalid volume value. Example: volume 0.5")

        elif command.startswith("offset"):
            parts = command.split()

            if len(parts) != 2:
                print("Usage: offset 10")
                continue

            try:
                seconds = float(parts[1])
                player.set_start_offset(seconds)
            except ValueError:
                print("Invalid offset value. Example: offset 10")

        elif command == "info":
            player.print_info()

        elif command == "help":
            print_help()

        elif command == "quit":
            player.stop()
            print("Exit program.")
            break

        else:
            print("Unknown command. Type 'help' to see available commands.")


if __name__ == "__main__":
    main()