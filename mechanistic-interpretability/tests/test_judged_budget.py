import math
from mechinterp.judged import selected_conditions

def test_request_accounting():
    # Across four model families, one configuration per each of 14 features x 2 adapters.
    features=14
    models=4
    n_per_model=features*2*100
    requests_per_model=features*2*math.ceil(100/10)*2
    assert n_per_model*models==11200
    assert requests_per_model*models==2240
