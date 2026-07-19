"""Tests for chunked binaural rendering via render_to_binaural_file."""

import numpy as np
import pytest
import soundfile as sf

from playamb.audio.data.ambifile import AmbisonicsFile
from playamb.audio.engine.hrtf import HRTF
from playamb.audio.engine.spherical import SphericalHarmonics
from playamb.audio.engine.render import render_to_binaural_file


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sh_first_order():
    """SphericalHarmonics, order 1, LS preprocessing (~3 s to build once)."""
    hrtf = HRTF()
    sh = SphericalHarmonics(
        hrtf=hrtf,
        sampling_rate=48000,
        ambi_order=1,
        preprocess="LS",
    )
    yield sh
    if hasattr(sh, "close"):
        sh.close()


@pytest.fixture
def synthetic_ambix_path(tmp_path):
    """1st-order AmbiX WAV, 48037 frames, 4 ch, float32, seeded noise."""
    rng = np.random.default_rng(42)
    data = (0.3 * rng.standard_normal((48037, 4))).astype(np.float32)
    path = tmp_path / "synthetic_ambix.wav"
    sf.write(str(path), data, 48000, subtype="FLOAT")
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRenderToBinauralFile:

    def test_chunked_render_matches_full_decode(
        self, synthetic_ambix_path, sh_first_order, tmp_path,
    ):
        """Chunked output must be numerically equivalent to apply_hrtf."""
        # -- full-file reference --------------------------------------------------
        ambi = AmbisonicsFile(str(synthetic_ambix_path), chunk_size=2048)
        full_signal = ambi.get_signal_chunk(0, ambi.total_frames)
        full = sh_first_order.apply_hrtf(full_signal).T  # (frames, 2)

        # -- chunked render -------------------------------------------------------
        out = tmp_path / "chunked.wav"
        ambi2 = AmbisonicsFile(str(synthetic_ambix_path), chunk_size=2048)
        render_to_binaural_file(ambi2, sh_first_order, str(out))

        chunked, _ = sf.read(str(out), dtype="float64")
        max_diff = np.max(np.abs(chunked - full))

        assert chunked.shape == full.shape, (
            f"Shape {chunked.shape} != {full.shape}"
        )
        # Tolerance rationale: the fast path accumulates in complex64 and
        # carries a float32 overlap buffer, FLOAT storage adds ~1.2e-7
        # relative error; signal amplitude is ~0.03 after pre_gain, so
        # atol=1e-5 leaves two orders of magnitude of margin.
        assert np.allclose(chunked, full, rtol=1e-4, atol=1e-5), (
            f"Numerical mismatch: max abs diff = {max_diff:.3e}"
        )

    def test_gain_scales_output(
        self, synthetic_ambix_path, sh_first_order, tmp_path,
    ):
        """gain=0.5 output equals exactly half of the gain=1.0 output."""
        ambi = AmbisonicsFile(str(synthetic_ambix_path), chunk_size=2048)
        ref_out = tmp_path / "gain_ref.wav"
        render_to_binaural_file(ambi, sh_first_order, str(ref_out))

        ambi2 = AmbisonicsFile(str(synthetic_ambix_path), chunk_size=2048)
        half_out = tmp_path / "gain_half.wav"
        render_to_binaural_file(ambi2, sh_first_order, str(half_out), gain=0.5)

        ref, _ = sf.read(str(ref_out), dtype="float64")
        half, _ = sf.read(str(half_out), dtype="float64")
        # multiplying by 0.5 is exact in binary floating point
        assert np.allclose(half, 0.5 * ref, rtol=1e-6, atol=1e-7), (
            "gain=0.5 output is not half of gain=1.0 output"
        )

    def test_output_wav_metadata(self, synthetic_ambix_path, sh_first_order, tmp_path):
        """Output WAV: stereo, 48 kHz, correct length, FLOAT subtype."""
        ir_len = sh_first_order.get_IR_length()
        ambi = AmbisonicsFile(str(synthetic_ambix_path), chunk_size=2048)

        out = tmp_path / "meta.wav"
        render_to_binaural_file(ambi, sh_first_order, str(out))

        info = sf.info(str(out))
        assert info.channels == 2
        assert info.samplerate == 48000
        assert info.frames == 48037 + ir_len - 1, (
            f"Expected {48037 + ir_len - 1}, got {info.frames}"
        )
        assert info.subtype == "FLOAT"

        # Second render with PCM_16
        ambi2 = AmbisonicsFile(str(synthetic_ambix_path), chunk_size=2048)
        out2 = tmp_path / "meta_pcm16.wav"
        render_to_binaural_file(ambi2, sh_first_order, str(out2), subtype="PCM_16")
        assert sf.info(str(out2)).subtype == "PCM_16"

    def test_chunk_size_larger_than_file(self, tmp_path, sh_first_order):
        """500-frame file with chunk_size=2048 must still produce correct output."""
        rng = np.random.default_rng(42)
        data = (0.3 * rng.standard_normal((500, 4))).astype(np.float32)
        small_path = tmp_path / "small.wav"
        sf.write(str(small_path), data, 48000, subtype="FLOAT")

        ambi = AmbisonicsFile(str(small_path), chunk_size=2048)
        full_signal = ambi.get_signal_chunk(0, ambi.total_frames)
        full = sh_first_order.apply_hrtf(full_signal).T

        out = tmp_path / "small_out.wav"
        ambi2 = AmbisonicsFile(str(small_path), chunk_size=2048)
        render_to_binaural_file(ambi2, sh_first_order, str(out))

        chunked, _ = sf.read(str(out), dtype="float64")
        max_diff = np.max(np.abs(chunked - full))
        assert chunked.shape == full.shape
        assert np.allclose(chunked, full, rtol=1e-4, atol=1e-5), (
            f"Numerical mismatch: max abs diff = {max_diff:.3e}"
        )

    def test_empty_file(self, tmp_path, sh_first_order):
        """0-frame WAV returns 0, writes valid 0-frame stereo WAV."""
        empty_path = tmp_path / "empty.wav"
        sf.write(str(empty_path), np.zeros((0, 4), np.float32), 48000,
                 subtype="FLOAT")

        ambi = AmbisonicsFile(str(empty_path), chunk_size=2048)
        out = tmp_path / "empty_out.wav"
        result = render_to_binaural_file(ambi, sh_first_order, str(out))

        assert result == 0
        info = sf.info(str(out))
        assert info.frames == 0
        assert info.channels == 2

    def test_position_saved_and_restored(
        self, synthetic_ambix_path, sh_first_order, tmp_path,
    ):
        """Read position saved on entry, restored on exit."""
        ambi = AmbisonicsFile(str(synthetic_ambix_path), chunk_size=2048)
        ambi.seek_to_position(1000)

        out = tmp_path / "pos.wav"
        render_to_binaural_file(ambi, sh_first_order, str(out))

        # Position restored
        assert ambi.get_current_position() == 1000, (
            f"Position {ambi.get_current_position()} != 1000"
        )

        # Output equal to render from fresh file
        fresh = AmbisonicsFile(str(synthetic_ambix_path), chunk_size=2048)
        ref_out = tmp_path / "pos_ref.wav"
        render_to_binaural_file(fresh, sh_first_order, str(ref_out))

        result, _ = sf.read(str(out), dtype="float64")
        ref, _ = sf.read(str(ref_out), dtype="float64")
        assert np.array_equal(result, ref), "Output differs from fresh-file render"

    def test_progress_callback(self, synthetic_ambix_path, sh_first_order, tmp_path):
        """Progress callback receives strictly increasing done, correct total."""
        ambi = AmbisonicsFile(str(synthetic_ambix_path), chunk_size=2048)
        out = tmp_path / "progress.wav"
        collected = []

        def cb(done, total):
            collected.append((done, total))

        render_to_binaural_file(
            ambi, sh_first_order, str(out), progress_callback=cb,
        )

        assert len(collected) > 0, "Callback never called"
        dones = [c[0] for c in collected]
        totals = [c[1] for c in collected]
        assert all(dones[i] < dones[i + 1] for i in range(len(dones) - 1)), (
            f"done not strictly increasing: {dones}"
        )
        assert dones[-1] == 48037, f"Last done {dones[-1]} != 48037"
        assert all(t == 48037 for t in totals), (
            f"Not all totals=48037: {totals}"
        )

    def test_headphone_filter_longer_tail(self, synthetic_ambix_path, tmp_path):
        """Render must handle longer IR from headphone-compensated HRIRs."""
        hrtf = HRTF()
        hrtf.load_hp_filter("Diffuse Field Equalization")
        sh2 = SphericalHarmonics(
            hrtf=hrtf, sampling_rate=48000, ambi_order=1, preprocess="LS",
        )
        try:
            ambi = AmbisonicsFile(str(synthetic_ambix_path), chunk_size=2048)
            full_signal = ambi.get_signal_chunk(0, ambi.total_frames)
            full = sh2.apply_hrtf(full_signal).T

            out = tmp_path / "dfe.wav"
            ambi2 = AmbisonicsFile(str(synthetic_ambix_path), chunk_size=2048)
            render_to_binaural_file(ambi2, sh2, str(out))

            ir_len = sh2.get_IR_length()
            info = sf.info(str(out))
            assert info.frames == 48037 + ir_len - 1, (
                f"Expected {48037 + ir_len - 1}, got {info.frames}"
            )

            chunked, _ = sf.read(str(out), dtype="float64")
            max_diff = np.max(np.abs(chunked - full))
            assert chunked.shape == full.shape
            assert np.allclose(chunked, full, rtol=1e-4, atol=1e-5), (
                f"Numerical mismatch with DFE: max diff = {max_diff:.3e}"
            )
        finally:
            sh2.close()

    def test_channel_mismatch_raises(self, tmp_path, sh_first_order):
        """ValueError when file channels != SH channels."""
        rng = np.random.default_rng(42)
        data = (0.3 * rng.standard_normal((1000, 9))).astype(np.float32)
        path9 = tmp_path / "order2.wav"
        sf.write(str(path9), data, 48000, subtype="FLOAT")

        ambi = AmbisonicsFile(str(path9), chunk_size=2048)
        out = tmp_path / "mismatch.wav"
        with pytest.raises(ValueError, match="(?i)channel"):
            render_to_binaural_file(ambi, sh_first_order, str(out))

    def test_gain_too_high_raises(
        self, synthetic_ambix_path, sh_first_order, tmp_path,
    ):
        """AttributeError when gain > 1.0."""
        ambi = AmbisonicsFile(str(synthetic_ambix_path), chunk_size=2048)
        with pytest.raises(AttributeError, match="(?i)gain"):
            render_to_binaural_file(
                ambi, sh_first_order, tmp_path / "gain_high.wav", gain=1.5,
            )

    def test_gain_too_low_raises(
        self, synthetic_ambix_path, sh_first_order, tmp_path,
    ):
        """AttributeError when gain < 0."""
        ambi = AmbisonicsFile(str(synthetic_ambix_path), chunk_size=2048)
        with pytest.raises(AttributeError, match="(?i)gain"):
            render_to_binaural_file(
                ambi, sh_first_order, tmp_path / "gain_low.wav", gain=-0.1,
            )
