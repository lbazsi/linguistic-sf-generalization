import pytest
from mechinterp.judged import verified_batch_judgments

def test_id_mapping_does_not_assume_order():
    batch=[{"scenario_id":7},{"scenario_id":9}]
    rows=[{"scenario_id":9,"scores":{}},{"scenario_id":7,"scores":{}}]
    result=verified_batch_judgments(batch,rows)
    assert list(result)==[9,7]
    assert set(result)=={7,9}

def test_id_mapping_rejects_duplicates_and_missing():
    batch=[{"scenario_id":7},{"scenario_id":9}]
    with pytest.raises(ValueError):
        verified_batch_judgments(batch,[{"scenario_id":7},{"scenario_id":7}])
    with pytest.raises(ValueError):
        verified_batch_judgments(batch,[{"scenario_id":7}])
    with pytest.raises(ValueError):
        verified_batch_judgments(batch,[{"scenario_id":7},{"scenario_id":3}])
