
from autonomy.deconfliction import detect_conflicts


def test_detects_close_drones():
    proposed = [(0, 0, 10), (1, 0, 10)]
    other = [(0, 1, 10), (1, 1, 10)]

    conflicts = detect_conflicts(proposed, [other], 2)
    assert len(conflicts) == 2


def test_no_conflict_when_far_apart():
    proposed = [(0, 0, 10), (1, 0, 10)]
    other = [(0, 0, 30), (1, 0, 30)]

    conflicts = detect_conflicts(proposed, [other], 5)
    assert conflicts == []
