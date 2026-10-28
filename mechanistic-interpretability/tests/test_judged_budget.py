import math
from mechinterp.judged import selected_conditions

def test_request_accounting():
    # Across four model families, one configuration per each of 14 features x 2 adapters.
    features=14
    models=4
    matched_pairs_per_model=features*2*100
    generations_per_model=matched_pairs_per_model*2
    judgments_per_model=generations_per_model*2
    requests_per_model=features*2*math.ceil(100/10)*2*2
    assert matched_pairs_per_model*models==11200
    assert generations_per_model*models==22400
    assert judgments_per_model*models==44800
    assert requests_per_model*models==4480
