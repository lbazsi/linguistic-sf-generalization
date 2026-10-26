from mechinterp.classifier_controls import controls

def test_lexical_control_detects_easy_surface_cues():
    rows=[]
    for i in range(90):
        part="train" if i<60 else "development" if i<75 else "test"
        rows.append({"text":f"The unmistakable azure token number {i}", "label":1,"partition":part})
        rows.append({"text":f"The unmistakable ochre token number {i}", "label":0,"partition":part})
    result=controls(rows)
    assert result["lexical_tfidf"]["test"]["roc_auc"]>.9
    assert result["constant_baseline"]["roc_auc"]==.5
