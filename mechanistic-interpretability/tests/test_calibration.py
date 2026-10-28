from collections import Counter
from mechinterp.calibration import stratified_general

def test_stratified_calibration_covers_all_general_domains():
    rows=[{"id":i+domain*50,"domain":f"d{domain}","prompt":"x"}
          for domain in range(20) for i in range(50)]
    chosen=stratified_general(rows,200,42)
    assert len(chosen)==200
    assert set(Counter(r["domain"] for r in chosen).values())=={10}
    assert chosen==stratified_general(rows,200,42)
