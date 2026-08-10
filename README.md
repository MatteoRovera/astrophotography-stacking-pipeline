# Astrophotography Stacking Pipeline

A from-scratch Python pipeline that turns a folder of night-sky exposures
into one clean, noise-reduced image: it optionally calibrates out sensor
defects, aligns the frames to a common reference, stacks them, and applies
a display stretch. Built as a learning project -- every stage is written to
be read, not just run, with the astronomy and image-processing reasoning
behind it explained in code comments and below.

It's built to handle two very different kinds of input with the *same code
path*:

- **Phone photos** -- JPEG or RAW/DNG, usually just light frames, no
  calibration set. Calibration is skipped automatically.
- **DSLR practice datasets** -- Canon CR2 or FITS, with matching dark/flat/
  bias calibration frames. Calibration runs automatically when those frames
  are present.

## Result

![Stacked and calibrated Cocoon Nebula field](cocoon_result_preview.png)

25 real light frames (240s each, Canon DSLR) of the Cocoon Nebula (IC 5146)
region, calibrated against real dark/flat/bias frames and stacked with this
pipeline. Stars are sharp with no doubling or trailing, which confirms the
calibration and alignment stages are working correctly on real hardware
data -- not just synthetic test cases.

This is linear data with only a brightness stretch applied, no color
calibration -- see [Next steps](#next-steps) for why it looks like this and
what closing that gap involves.

## The story

This pipeline was built stage by stage -- loading, then calibration, then
alignment, then stacking, then post-processing -- each one covered by unit
tests against synthetic data with known, exactly-checkable answers (a
known injected dark offset that should subtract out to zero, a known
injected pixel shift that alignment should recover, N identical frames
that any combine method should return unchanged). All of that passed
before it ever touched a real photo.

Then it got pointed at a real 80-file DSLR dataset (the Cocoon Nebula set
above) and immediately surfaced two bugs synthetic tests hadn't caught:

1. **The filename-based frame sorter silently misclassified everything.**
   `\bbias\b`-style regex matching doesn't work on `Bias_000.fits` because
   `_` counts as a word character in regex, so there's no boundary there --
   every calibration frame was quietly falling through to "light". Worse,
   this real dataset used a single-letter convention (`L_`/`D_`/`F_`/`B_`)
   that the fixed classifier *still* didn't handle. Both are fixed now,
   with regression tests using the exact real filenames that broke it.
2. **The pipeline didn't scale to full-resolution DSLR data.** The
   straightforward approach -- decode every file into RAM and hold it all
   at once -- needed roughly 17 GB for this 80-file session, more than the
   16 GB machine it was running on had. Rather than just documenting that
   limit, the pipeline now estimates the decoded size of a folder up front
   and automatically switches to a memory-bounded streaming mode (temp
   memory-mapped files on disk, processed one frame and one row-band at a
   time) when a session is too big to fit in RAM. See
   [In-memory vs. streaming mode](#6-in-memory-vs-streaming-mode).

With both fixed, the full 80-file set (25 lights, 9 darks, 23 flats, 23
bias) ran end to end and produced the image above: background noise
dropped 2.0x and SNR improved ~1.9x after stacking (below the theoretical
sqrt(25)=5.0x, for real reasons -- alignment resampling correlates
neighboring pixels, the dark frames were shot across a 16-20C range rather
than one fixed temperature, and the last 5 frames appear to be a separate
sub-session with fewer matched alignment stars). Real data, real numbers,
real bugs found and fixed -- not just a pipeline that works on toy inputs.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS/Linux
pip install -r requirements.txt
```

## Usage

```bash
python stack.py --input ./my_photos --output result.tiff --stack sigma
```

Drop every file for one session into a single input folder -- lights,
and optionally darks/flats/bias -- and point `--input` at it. The pipeline
sorts them automatically (see [Loading](#1-loading--sorting) below) and
either calibrates or skips calibration depending on what it finds.

### CLI options

| Flag | Default | Meaning |
|---|---|---|
| `--input` | *(required)* | Folder of input images |
| `--output` | *(required)* | Output path; `.tiff` saves 16-bit, anything else saves 8-bit |
| `--stack` | `sigma` | Combine method: `mean`, `median`, or `sigma` |
| `--sigma-thresh` | `3.0` | Sigma-clip rejection threshold (only used by `--stack sigma`) |
| `--sigma-maxiters` | `5` | Sigma-clip max iterations |
| `--stretch` | `asinh` | Display stretch: `percentile`, `asinh`, or `none` |
| `--reference` | `0` | Index of the light frame used as the alignment reference |
| `--recursive` | off | Scan `--input` recursively |
| `--bit-depth` | `16` | Output bit depth for TIFF (`8` or `16`) |
| `--memory-budget-mb` | `3072` | Above this estimated data size, switch to memory-bounded streaming mode. `0` forces streaming always. |
| `-v`, `--verbose` | off | Debug-level logging |

The CLI prints a before/after noise and SNR report for every run -- that's
the quantified "did stacking actually help" answer, not just a picture.

## How it works

### 1. Loading & sorting

Every file in the input folder is decoded and bucketed into **light**,
**dark**, **flat**, or **bias** by filename: an explicit prefix like
`Light_`, `Dark_`, `Flat_`, `Bias_` is honored (as is the single-letter
`L_`/`D_`/`F_`/`B_` convention some capture tools use), and any file with
no recognizable keyword (a phone photo, or a descriptive name like
`M42_300s_001.fits`) is assumed to be a light frame -- since that's the
only frame type you'd capture without labeling it.

What each frame type physically is:

- **Light**: the real exposure -- target signal plus every noise source
  (photon shot noise, sensor read noise, thermal "dark current").
- **Dark**: shot with the sensor capped, same exposure time/temperature as
  the lights. No light signal, only the thermal/read-noise pattern, so it
  can be subtracted back out of a light.
- **Flat**: a picture of a uniformly lit target (twilight sky, light
  panel). No astronomical signal, but it records *multiplicative* optical
  defects -- vignetting, dust shadows -- so it's used as a divisor.
- **Bias**: a zero-length exposure. Isolates the sensor's read-noise
  offset with no thermal component at all.

Supported formats: JPEG/PNG/TIFF (via `imageio`), Canon/Nikon/etc. RAW
(`.cr2`, `.nef`, `.dng`, ...) via `rawpy`/LibRaw, and FITS via `astropy`.

RAW files are demosaiced with gamma disabled and auto-brightness off
(`rawpy.postprocess(gamma=(1,1), no_auto_bright=True)`), so pixel values
stay roughly proportional to captured light. That keeps calibration
arithmetic valid, but it's a simplification: the more rigorous approach
calibrates the raw Bayer mosaic *before* demosaicing (since flat-fielding
corrects per-pixel sensor response, and demosaicing interpolates between
pixels). This pipeline demosaics first and calibrates the RGB result, which
is simpler and still removes the bulk of vignetting/dark current, at the
cost of some fine per-pixel accuracy.

### 2. Calibration (optional)

If no dark/flat/bias frames are found, this stage is skipped entirely and
logged -- lights pass straight to alignment. This is what makes the phone-
photo path work.

If calibration frames exist, each set is combined into a **master frame**
with a per-pixel **median** (not mean) across the set -- a median rejects
one-off outliers (a cosmic ray hit, a hot pixel that spiked in a single
sub-frame) that a mean would blend into the result.

The correction applied to every light frame:

```
calibrated = (light - master_dark) / normalized_master_flat
```

- Subtracting the master dark removes the additive thermal/read-noise
  pattern.
- Dividing by the flat (normalized to mean = 1 first) removes vignetting
  and dust shadows *without* changing overall brightness.
- Bias is **not** subtracted from the light directly when a matching dark
  exists -- a dark already contains that same read-noise offset (a dark
  exposure is "bias + thermal noise"), so subtracting both would remove it
  twice. Bias is used instead to clean up the flat, or as a fallback
  stand-in for dark subtraction when no matching dark is available (this
  only removes the fixed offset, not thermal buildup, so it's an
  approximation -- fine for short exposures).

### 3. Alignment (registration)

Between exposures the target drifts across the sensor (tracking error,
polar misalignment, or handheld drift), so every frame needs to be
resampled onto the same pixel grid as a reference frame before stacking.
Three approaches were considered:

- **astroalign (asterism/triangle matching)** -- detects star-like point
  sources, forms triangles from triplets of them, and matches triangles
  between frames by their shape (side-length ratios stay the same
  regardless of rotation/scale/translation). Robust and handles field
  rotation, but needs a handful of well-detected stars -- it has nothing
  to grab onto with 1-2 stars or a diffuse target like the Moon.
- **Generic feature matching (ORB/SIFT + RANSAC)** -- works on textured,
  non-stellar content (lunar craters), but stars are small, nearly
  identical blobs, so feature descriptors barely distinguish one star from
  another. Not a good fit for star fields, which is the common case here.
- **Phase correlation (FFT cross-correlation)** -- computes the pixel
  shift directly from the phase of the cross-power spectrum, no point
  detection needed. Fast and robust, and it's exactly what still works
  when there's only one dominant feature (a Moon shot, a couple of stars).
  Its limitation is that it only recovers translation, not rotation.

**Decision**: astroalign is tried first (it's the only one of the three
suited to dense star fields). If it can't find enough matched stars --
raises an error, or isn't installed -- the frame falls back automatically
to phase correlation. That one fallback rule is what lets the same code
path handle both a rich star field and a sparse Moon shot.

### 4. Stacking

Why stacking reduces noise at all: assuming each pixel's noise (shot noise
+ read noise) is independent from frame to frame, averaging N frames makes
the *signal* add linearly (N x) while independent noise adds in quadrature
(sqrt(N) x). Signal-to-noise therefore improves by sqrt(N) -- the entire
reason deep-sky imaging stacks many sub-exposures instead of one long one.

Three combine methods:

- **mean** -- statistically optimal for independent Gaussian noise (gets
  the full sqrt(N) improvement), but has no concept of "this pixel is
  wrong": a satellite trail or cosmic ray hit in even one frame gets
  blended into the result.
- **median** -- immune to a minority of contaminated frames (a trail or
  hit vanishes completely, as long as fewer than half the frames are bad
  at that pixel), at the cost of statistical efficiency: for large N, its
  noise is about 1.25x the mean's, so it needs ~1.57x as many frames to
  match the mean's noise reduction.
- **sigma-clipped mean** (default) -- per pixel, compute the mean/std
  across the stack, mask values more than `--sigma-thresh` std devs away,
  then average what's left. This rejects outliers like the median does
  while staying close to the mean's noise performance. It's what tools
  like DeepSkyStacker and PixInsight use by default, and needs a handful
  of frames (5+) for the per-pixel statistics to be meaningful.

The CLI reports background noise and SNR for the reference frame vs. the
final stack, plus the theoretical sqrt(N) prediction, so you get a real
number instead of just a prettier picture.

### 5. Post-processing

Stacked data is linear with an enormous dynamic range -- the sky background
and faint nebulosity sit just above zero while stars are orders of
magnitude brighter. Displayed linearly, that's a black frame with a few
white dots. Two stretch options remap intensity so faint detail becomes
visible:

- **percentile** -- pick a low/high percentile as black/white point and
  linearly rescale between them. Simple, but a linear map can't reveal
  faint detail and protect bright cores at the same time when the dynamic
  range is large.
- **asinh** (default) -- `asinh(x)` behaves like `x` for small values and
  like `log(x)` for large ones. That's exactly the shape needed: faint,
  near-background signal is stretched close to proportionally, while
  bright signal (stars) is compressed logarithmically instead of clipping
  to solid white. This is the same idea behind the "Lupton stretch" used
  for SDSS survey images.

Both stretches compute black/white points across all channels together
(not per-channel), so color balance isn't skewed -- but nothing in this
stage *corrects* color balance either, which is exactly the gap the next
section is about.

### 6. In-memory vs. streaming mode

There are two ways the pipeline actually executes, and it picks between
them automatically. The straightforward approach -- decode every file into
a normal in-RAM array and keep all of them around at once -- is simple and
fast, but N frames at H x W x channels x 4 bytes (float32) each means N x
that much RAM simultaneously. A session of full-resolution DSLR frames adds
up fast: 80 frames at ~24 MP demosaiced to RGB is on the order of 17 GB,
more than most machines have free.

Before doing any real work, the pipeline cheaply estimates the total
decoded size of everything in the input folder -- RAW files are only
header-parsed (no demosaic), FITS files are sized from header keywords,
standard images are opened lazily -- none of that requires actually
decoding pixel data. If the estimate exceeds `--memory-budget-mb` (default
3072 MB / 3 GB), it switches to a streaming mode instead:

- Every file is decoded one at a time and immediately written to a
  temporary memory-mapped file on disk, dropping the in-RAM copy. Peak RAM
  for loading is one frame, not N.
- The two operations that genuinely need to see every frame at a given
  pixel -- median-combining the calibration masters, and the final stack
  -- are done in horizontal row bands: read a band of rows from every
  frame's memmap, combine just that band (with the exact same
  `stack_mean`/`stack_median`/`stack_sigma_clip` functions used everywhere
  else), write it out, move to the next band. Peak RAM for combining is
  tuned to a fixed budget regardless of how many frames there are or how
  large they are.
- Calibration and alignment, which only ever need one frame plus the
  (already small) master arrays, happen in that same one-file-at-a-time
  pass, and intermediate temp files are deleted as soon as they're no
  longer needed rather than piling up for the whole run.

The streaming path is slower (disk I/O instead of RAM, and each row-band
combine has some fixed overhead), so it's only used when the estimate says
it's needed. Force it with `--memory-budget-mb 0` if you'd rather not rely
on the estimate.

## Next steps

**Color calibration.** The result above has a strong green cast, which is
expected: Bayer sensors are inherently green-heavy (twice as many green
photosites as red or blue per the RGGB pattern), and this pipeline's
post-processing stage only stretches *brightness* -- nothing currently
corrects *color balance*. The planned fix, roughly in order of how it'd
actually get built:

1. **Background neutralization (first step).** The pipeline already
   computes sigma-clipped background statistics per image for the noise/SNR
   report (`stacking.measure_noise_snr`) -- extending that to run
   per-channel would give the background level of R, G, and B separately.
   Scaling each channel so those three background levels match (a "gray
   world on the background" assumption -- the night sky itself should be
   neutral, not green) is a simple, cheap correction that would remove
   most of the cast directly, applied before the stretch.
2. **SCNR-style green suppression (complementary).** A common finishing
   touch in astro tools specifically for residual green: clamp each pixel's
   green channel to no more than the average of its red and blue, which
   targets exactly the kind of broad green cast visible here without
   touching genuinely green astronomical signal (which is rare -- most
   real nebula color is red/blue/magenta).
3. **Photometric color calibration (the rigorous version, later).**
   Real tools calibrate color by comparing measured star colors in the
   image against catalog color indices (e.g. Gaia). That needs star
   detection and a catalog cross-match -- meaningfully more work than 1-2,
   and probably not worth it until the simpler passes prove insufficient.
4. **A mild saturation boost after calibration.** Nebula color is usually
   quite desaturated in linear data even once neutrally balanced; a small
   saturation lift after step 1/2 is what makes color "pop" the way
   processed astrophotos typically look, without it being a color-accuracy
   step per se.

Two smaller items also worth doing:

- **Crop to common overlap.** The result has a faint rectangular seam near
  one edge -- the "stacking footprint" effect, where frames registered
  with slightly different sub-pixel shifts don't all cover exactly the
  same area, so the border gets thinner coverage than the center. Cropping
  the final stack to the intersection of every frame's footprint (or
  feathering that edge) is standard practice and not yet implemented here.
- **A stronger default stretch**, or an easy way to re-run just the
  post-processing stage with different stretch parameters without
  recalibrating/re-aligning/re-stacking from scratch -- useful for taste
  before committing to a full re-run.

## Project layout

```
stack.py                   # CLI entry point
astro_stack/
    loader.py               # read a folder, decode formats, sort by type
    calibration.py           # master dark/flat/bias + calibration math
    alignment.py              # astroalign + phase-correlation registration
    stacking.py                # mean/median/sigma-clip + noise/SNR measurement
    postprocess.py              # percentile/asinh stretch + save
    streaming.py                 # memory-bounded pipeline for large sessions
    pipeline.py                   # picks streaming vs. in-memory, wires stages together
tests/
    test_loader.py               # filename classification
    test_calibration.py           # synthetic dark/flat removal
    test_alignment.py              # recovers a known injected shift
    test_stacking.py                # N identical frames -> same frame
    test_streaming.py                # chunked combine matches eager combine
```

## Tests

```bash
pytest tests/ -v
```

- `test_stacking.py` -- stacking N identical frames (mean/median/sigma,
  mono and color) returns that same frame.
- `test_alignment.py` -- injects a known sub-pixel shift into a synthetic
  star field and a synthetic sharp-edged "Moon" disk, and checks the
  aligned output has near-zero residual shift against the reference.
- `test_calibration.py` -- a synthetic dark frame with a known constant
  offset is subtracted out exactly; a synthetic flat with a known
  vignetting pattern is divided out; and calibration is confirmed to be a
  clean no-op when no calibration frames are present.
- `test_loader.py` -- filename-based frame-type classification, including
  the underscore-prefix case (`Bias_000.fits`, `Dark_005s_000.fits`, ...)
  and the single-letter DSLR convention (`L_`/`D_`/`F_`/`B_`) that broke on
  first contact with real data.
- `test_streaming.py` -- the chunked, memmap-backed combine used by
  streaming mode produces bit-for-bit the same result as the plain
  in-memory combine, and the full streaming pipeline agrees with the
  in-memory pipeline on the same synthetic dataset.
