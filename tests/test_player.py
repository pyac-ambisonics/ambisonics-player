"""Device-free regression tests for AudioPlayer (is_loaded + queue producer)."""

import queue
import threading
import time

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("sounddevice")

from playamb.audio.data.ambifile import AmbisonicsFile
from playamb.audio.engine.hrtf import HRTF
from playamb.audio.engine.player import AudioPlayer
from playamb.audio.engine.spherical import SphericalHarmonics

FRAMES = 48037  # deliberately not divisible by chunk size


@pytest.fixture(scope="module")
def sh_first_order():
    """SphericalHarmonics, order 1, LS preprocessing (expensive, build once)."""
    hrtf = HRTF()
    sh = SphericalHarmonics(
        hrtf=hrtf, sampling_rate=48000, ambi_order=1, preprocess="LS",
    )
    yield sh
    if hasattr(sh, "close"):
        sh.close()


@pytest.fixture
def synthetic_ambix_path(tmp_path):
    """1st-order AmbiX WAV, 48037 frames, 4 ch, float32, seeded noise."""
    rng = np.random.default_rng(42)
    data = (0.3 * rng.standard_normal((FRAMES, 4))).astype(np.float32)
    path = tmp_path / "synthetic_ambix.wav"
    sf.write(str(path), data, 48000, subtype="FLOAT")
    return path


@pytest.fixture
def player(synthetic_ambix_path, sh_first_order):
    ambi = AmbisonicsFile(str(synthetic_ambix_path), chunk_size=2048)
    p = AudioPlayer(ambi, sh_first_order)
    yield p
    p.stop_event.set()


class TestIsLoaded:

    def test_init_sets_is_loaded(self, player):
        """Constructing with a valid file must make the player playable."""
        assert player.is_loaded is True

    def test_get_duration(self, player):
        """get_duration must return the file duration, not the 0.0 guard."""
        assert player.get_duration() == pytest.approx(FRAMES / 48000)

    def test_channel_mismatch_raises(self, tmp_path, sh_first_order):
        rng = np.random.default_rng(42)
        data = (0.3 * rng.standard_normal((1000, 9))).astype(np.float32)
        path9 = tmp_path / "order2.wav"
        sf.write(str(path9), data, 48000, subtype="FLOAT")
        ambi = AmbisonicsFile(str(path9), chunk_size=2048)
        with pytest.raises(ValueError, match="(?i)channel"):
            AudioPlayer(ambi, sh_first_order)

    def test_gain_out_of_range_raises(self, synthetic_ambix_path,
                                      sh_first_order):
        ambi = AmbisonicsFile(str(synthetic_ambix_path), chunk_size=2048)
        with pytest.raises(AttributeError, match="(?i)gain"):
            AudioPlayer(ambi, sh_first_order, gain=1.5)
        with pytest.raises(AttributeError, match="(?i)gain"):
            AudioPlayer(ambi, sh_first_order, gain=-0.1)


class TestPutStopAware:

    @staticmethod
    def _fill_queue(player):
        while True:
            try:
                player.audio_queue.put_nowait(np.zeros((4, 2), np.float32))
            except queue.Full:
                return

    def test_true_when_space(self, player):
        assert player._put_stop_aware(np.zeros((4, 2), np.float32)) is True
        assert player.audio_queue.qsize() == 1

    def test_immediate_false_when_stopped(self, player):
        self._fill_queue(player)
        player.stop_event.set()
        start = time.perf_counter()
        result = player._put_stop_aware(np.zeros((4, 2), np.float32))
        elapsed = time.perf_counter() - start
        assert result is False
        assert player.audio_queue.qsize() == 10
        assert elapsed < 0.5

    def test_unblocks_on_stop(self, player):
        self._fill_queue(player)
        threading.Timer(0.3, player.stop_event.set).start()
        start = time.perf_counter()
        result = player._put_stop_aware(np.zeros((4, 2), np.float32))
        elapsed = time.perf_counter() - start
        assert result is False
        assert 0.25 <= elapsed < 1.5


class TestProcessLoop:

    def test_stop_drains_stale_queue(self, player):
        for _ in range(3):
            player.audio_queue.put_nowait(np.zeros((4, 2), np.float32))
        player.audio_queue.put_nowait(None)
        player.stop()
        assert player.audio_queue.empty()
        assert not player.play_event.is_set()

    def test_process_loop_stop_leaves_queue_clean(self, player, capsys):
        """Core regression: stop must kill the producer promptly, drain the
        queue, and never drop blocks (old code printed a drop warning and
        left stale items behind)."""
        player._reset_process_variables()
        thread = threading.Thread(target=player._process_loop, daemon=True)
        # register like play() does, so stop() joins before draining
        player.processing_thread = thread
        thread.start()
        deadline = time.perf_counter() + 10
        while player.audio_queue.qsize() < 10:
            assert time.perf_counter() < deadline, "queue never filled"
            time.sleep(0.01)
        time.sleep(0.3)  # > 2x the 0.1s put timeout: producer keeps retrying
        player.stop()
        thread.join(2)
        assert not thread.is_alive()
        assert player.audio_queue.empty()
        assert "Dropping this block" not in capsys.readouterr().out

    def test_process_loop_natural_eof(self, player):
        """Producer must exit by itself after emitting the None sentinel."""
        player._reset_process_variables()
        thread = threading.Thread(target=player._process_loop, daemon=True)
        thread.start()
        deadline = time.perf_counter() + 15
        while True:
            assert time.perf_counter() < deadline, "never saw EOF sentinel"
            item = player.audio_queue.get(timeout=5)
            if item is None:
                break
        player.stop_event.set()  # let the loop's next iteration exit
        thread.join(2)
        assert not thread.is_alive()

    def test_stop_while_paused_exits(self, player, capsys):
        player._reset_process_variables()
        player.pause_event.set()
        thread = threading.Thread(target=player._process_loop, daemon=True)
        thread.start()
        time.sleep(0.3)
        player.stop()
        thread.join(1.5)
        assert not thread.is_alive()
        assert "did not terminate cleanly" not in capsys.readouterr().out
