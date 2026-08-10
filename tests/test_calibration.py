from pathlib import Path

import numpy as np

from astro_stack.calibration import calibrate_lights
from astro_stack.loader import Frame


def test_calibration_removes_known_dark_offset():
    rng = np.random.default_rng(0)
    signal = rng.uniform(0.05, 0.5, size=(24, 24)).astype(np.float32)
    dark_offset = np.full((24, 24), 0.07, dtype=np.float32)

    light = Frame(data=signal + dark_offset, path=Path("Light_001.fits"), frame_type="light")
    # Three identical dark frames -> median-combined master dark equals the offset exactly.
    darks = [
        Frame(data=dark_offset.copy(), path=Path(f"Dark_{i}.fits"), frame_type="dark")
        for i in range(3)
    ]

    buckets = {"light": [light], "dark": darks, "flat": [], "bias": []}
    calibrated = calibrate_lights(buckets)

    assert len(calibrated) == 1
    assert np.allclose(calibrated[0].data, signal, atol=1e-5)


def test_calibration_skipped_cleanly_when_no_cal_frames():
    rng = np.random.default_rng(1)
    signal = rng.uniform(0.05, 0.5, size=(16, 16)).astype(np.float32)
    light = Frame(data=signal.copy(), path=Path("IMG_0001.jpg"), frame_type="light")

    buckets = {"light": [light], "dark": [], "flat": [], "bias": []}
    result = calibrate_lights(buckets)

    assert len(result) == 1
    assert np.array_equal(result[0].data, signal)


def test_calibration_applies_flat_field_correction():
    rng = np.random.default_rng(2)
    signal = rng.uniform(0.2, 0.4, size=(20, 20)).astype(np.float32)

    # Simulate vignetting: a radial dimming that peaks at 1.0 in the center.
    yy, xx = np.mgrid[0:20, 0:20]
    center = 9.5
    radius = np.sqrt((yy - center) ** 2 + (xx - center) ** 2)
    vignette = (1.0 - 0.4 * (radius / radius.max())).astype(np.float32)

    light = Frame(data=signal * vignette, path=Path("Light_001.fits"), frame_type="light")
    flats = [
        Frame(data=vignette.copy(), path=Path(f"Flat_{i}.fits"), frame_type="flat")
        for i in range(3)
    ]

    buckets = {"light": [light], "dark": [], "flat": flats, "bias": []}
    calibrated = calibrate_lights(buckets)

    # Flat is normalized to mean=1 before dividing, so the vignetting cancels
    # out and what's left is the flat signal pattern scaled by mean(vignette).
    scale = float(np.mean(vignette))
    assert np.allclose(calibrated[0].data, signal * scale, atol=1e-3)
