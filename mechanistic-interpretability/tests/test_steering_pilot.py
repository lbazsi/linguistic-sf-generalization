from mechinterp.steering_pilot import relative_to_strength
def test_relative_strength():
    assert abs(relative_to_strength(.1,4096)/64-.1)<1e-10
