"""load a folder of astro images and sort into light/dark/flat/bias frames.
sorted by filename prefix. see readme for what each frame type means."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger("astro_stack.loader")

FITS_EXTS = {".fits", ".fit", ".fts"}
RAW_EXTS = {".cr2", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2"}
STANDARD_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
SUPPORTED_EXTS = FITS_EXTS | RAW_EXTS | STANDARD_EXTS

FRAME_TYPES = ("light", "dark", "flat", "bias")

# split on non-alphanumeric chars so tokens match exactly (regex \b fails on '_')
_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9]+")

# some dslr sets use a single leading letter instead of the full word.
# only trusted as the first token.
_SINGLE_LETTER_TYPES = {"l": "light", "d": "dark", "f": "flat", "b": "bias"}


@dataclass
class Frame:
    """one decoded image plus its metadata."""

    data: np.ndarray  # float32, shape (H, W) mono or (H, W, C) color
    path: Path
    frame_type: str  # 'light' | 'dark' | 'flat' | 'bias'
    exposure_s: Optional[float] = None
    header: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.data.shape


def classify_filename(name: str) -> str:
    """guess a frame's type from its filename. defaults to 'light'."""
    tokens = [t for t in _TOKEN_SPLIT.split(name) if t]
    if tokens and tokens[0].lower() in _SINGLE_LETTER_TYPES:
        return _SINGLE_LETTER_TYPES[tokens[0].lower()]
    for token in tokens:
        low = token.lower()
        if low in FRAME_TYPES:
            return low
    return "light"


def _load_fits(path: Path) -> Frame:
    from astropy.io import fits

    with fits.open(path) as hdul:
        # first hdu with actual pixel data (primary hdu is sometimes empty)
        hdu = next(h for h in hdul if h.data is not None)
        data = np.asarray(hdu.data, dtype=np.float32)
        header = dict(hdu.header)

    if data.ndim == 3 and data.shape[0] in (3, 4):
        # (channels, h, w) -> (h, w, channels)
        data = np.moveaxis(data, 0, -1)

    exposure = header.get("EXPTIME") or header.get("EXPOSURE")
    return Frame(data=data, path=path, frame_type="", exposure_s=exposure, header=header)


def _load_raw(path: Path) -> Frame:
    import rawpy

    with rawpy.imread(str(path)) as raw:
        # linear-ish demosaic (no gamma, no auto-brightness) so calibration
        # math stays valid. see readme for the tradeoff vs raw bayer calibration.
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            output_bps=16,
            gamma=(1, 1),
        )
    data = np.asarray(rgb, dtype=np.float32) / 65535.0
    return Frame(data=data, path=path, frame_type="", header={})


def _load_standard(path: Path) -> Frame:
    import imageio.v3 as iio

    img = iio.imread(path)
    arr = np.asarray(img)
    max_val = 255.0 if arr.dtype == np.uint8 else 65535.0
    data = arr.astype(np.float32) / max_val
    if data.ndim == 3 and data.shape[2] == 4:
        data = data[:, :, :3]  # drop alpha channel
    return Frame(data=data, path=path, frame_type="", header={})


def load_frame(path: Path) -> Frame:
    ext = path.suffix.lower()
    if ext in FITS_EXTS:
        frame = _load_fits(path)
    elif ext in RAW_EXTS:
        frame = _load_raw(path)
    elif ext in STANDARD_EXTS:
        frame = _load_standard(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext} ({path})")

    frame.frame_type = classify_filename(path.stem)
    return frame


def load_folder(input_dir: Path, recursive: bool = False) -> dict[str, list[Frame]]:
    """read every supported image in a folder and bucket it by frame type."""
    input_dir = Path(input_dir)
    pattern = "**/*" if recursive else "*"
    paths = sorted(
        p for p in input_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )

    buckets: dict[str, list[Frame]] = {t: [] for t in FRAME_TYPES}
    for path in paths:
        try:
            frame = load_frame(path)
        except Exception as exc:  # noqa: BLE001 - log and keep going
            logger.warning("Skipping %s: %s", path.name, exc)
            continue

        buckets[frame.frame_type].append(frame)
        label = {
            "light": "LIGHT frame (target + sky signal + noise)",
            "dark": "DARK frame (thermal + read noise, no signal)",
            "flat": "FLAT frame (vignetting + dust, uniform illumination)",
            "bias": "BIAS frame (pure read noise, zero-length exposure)",
        }[frame.frame_type]
        logger.info("Loaded %s as %s -- shape %s", path.name, label, frame.data.shape)

    counts = {t: len(v) for t, v in buckets.items()}
    logger.info("Folder scan complete: %s", counts)
    if counts["light"] == 0:
        raise ValueError(f"No light frames found in {input_dir}")

    return buckets


def scan_folder(input_dir: Path, recursive: bool = False) -> dict[str, list[Path]]:
    """classify files by filename only, no decoding. used to estimate memory
    footprint before picking in-memory vs streaming mode (pipeline.py)."""
    input_dir = Path(input_dir)
    pattern = "**/*" if recursive else "*"
    paths = sorted(
        p for p in input_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    )

    buckets: dict[str, list[Path]] = {t: [] for t in FRAME_TYPES}
    for path in paths:
        buckets[classify_filename(path.stem)].append(path)

    if not buckets["light"]:
        raise ValueError(f"No light frames found in {input_dir}")

    return buckets


def estimate_decoded_bytes(path: Path) -> int:
    """cheap estimate of decoded size in bytes, without decoding pixel data."""
    ext = path.suffix.lower()
    try:
        if ext in FITS_EXTS:
            from astropy.io import fits

            with fits.open(path) as hdul:
                for hdu in hdul:
                    naxis = hdu.header.get("NAXIS", 0)
                    if naxis >= 2:
                        dims = [hdu.header.get(f"NAXIS{i + 1}", 0) for i in range(naxis)]
                        n = 1
                        for d in dims:
                            n *= max(int(d), 1)
                        return n * 4
            return path.stat().st_size * 4

        if ext in RAW_EXTS:
            import rawpy

            with rawpy.imread(str(path)) as raw:
                sizes = raw.sizes
                return sizes.width * sizes.height * 3 * 4  # demosaiced rgb, float32

        if ext in STANDARD_EXTS:
            from PIL import Image

            with Image.open(path) as im:
                w, h = im.size
                channels = max(len(im.getbands()), 1)
                return w * h * channels * 4
    except Exception as exc:  # noqa: BLE001 - fall through to file-size guess
        logger.debug("Could not peek dimensions of %s (%s); estimating from file size", path.name, exc)

    return path.stat().st_size * 4
