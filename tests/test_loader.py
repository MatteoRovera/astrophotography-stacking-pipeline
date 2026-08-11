from astro_stack.loader import classify_filename


def test_classify_filename_prefix_with_underscore():
    """regression test: \\bbias\\b never matches "Bias_000" since '_' counts
    as a word character in regex, so there's no boundary there."""
    assert classify_filename("Bias_000.fits") == "bias"
    assert classify_filename("Dark_005s_000.fits") == "dark"
    assert classify_filename("Flat_001.fits") == "flat"
    assert classify_filename("Light_M42_005s_000.fits") == "light"


def test_classify_filename_defaults_to_light_for_descriptive_names():
    """no dark/flat/bias keyword in the name means it's a light frame."""
    assert classify_filename("M42_300s_001.fits") == "light"
    assert classify_filename("IMG_0001.jpg") == "light"
    assert classify_filename("moon_shot.dng") == "light"


def test_classify_filename_single_letter_prefix_convention():
    """some dslr sets abbreviate to one leading letter, e.g. "L_0020...", "B_0085...CR2"."""
    assert classify_filename("B_0085_ISO800___20C.CR2") == "bias"
    assert classify_filename("D_0006_ISO800_240s__16C.CR2") == "dark"
    assert classify_filename("F_0062_ISO800_1-1000s__20C.CR2") == "flat"
    assert classify_filename("L_0020_ISO800_240s__18C.CR2") == "light"
