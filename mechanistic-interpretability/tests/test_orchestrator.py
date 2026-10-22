from run_all import plan

def test_complete_steering_coverage():
    tasks=plan("gemma2","steering")
    assert len(tasks)==14*2*2
    assert set(x["policy"] for x in tasks)=={"prompt_last","generation"}
    assert all(x["model"]=="gemma2" for x in tasks)
    assert len({(t["feature"],t["condition"],t["policy"]) for t in tasks})==len(tasks)

def test_nonjudge_task_counts():
    assert len(plan("gemma2","difference"))==16
    assert len(plan("gemma2","lora"))==2
    assert len(plan("gemma2","probes"))==17*(2+14)
