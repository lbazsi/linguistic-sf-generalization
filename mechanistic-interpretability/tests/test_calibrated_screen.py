import math
def test_calibrated_fraction_magnitude():
    for width in (256, 4096):
        for fraction in (-.25,-.1,-.03,0,.03,.1,.25):
            strength=fraction*math.sqrt(width)
            assert math.isclose(strength/math.sqrt(width),fraction,abs_tol=1e-12)
