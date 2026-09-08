import pytest
from llm_lab.data.miner import classify_task

def test_classify_task():
    assert classify_task(4, 4) == "solved"
    assert classify_task(3, 4) == "solved"
    assert classify_task(2, 4) == "borderline"
    assert classify_task(1, 4) == "hard"
    assert classify_task(0, 4) == "impossible"
    
    # Test arbitrary n
    assert classify_task(8, 10) == "solved"       # 0.8
    assert classify_task(7, 10) == "borderline"   # 0.7
    assert classify_task(5, 10) == "borderline"   # 0.5
    assert classify_task(2, 10) == "hard"         # 0.2
    assert classify_task(0, 10) == "impossible"   # 0.0
