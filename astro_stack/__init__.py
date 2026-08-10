"""Astrophotography stacking pipeline.

Modules:
    loader        - read a folder, decode images, sort into light/dark/flat/bias
    calibration   - build master calibration frames and calibrate lights
    alignment     - register (align) light frames to a common reference
    stacking      - combine aligned frames and measure noise/SNR
    postprocess   - stretch the stacked image and save it
    pipeline      - wires the stages together for the CLI
"""

__version__ = "0.1.0"
