import time

from eae_runtime import Profiler


def test_record_and_report_basic_stats():
    p = Profiler()
    p.record("op", 0.1)
    p.record("op", 0.3)
    report = p.report()
    assert report["op"]["count"] == 2
    assert abs(report["op"]["total_seconds"] - 0.4) < 1e-9
    assert abs(report["op"]["mean_seconds"] - 0.2) < 1e-9
    assert report["op"]["max_seconds"] == 0.3
    assert report["op"]["min_seconds"] == 0.1


def test_track_context_manager_records_elapsed():
    p = Profiler()
    with p.track("sleep_op"):
        time.sleep(0.01)
    report = p.report()
    assert report["sleep_op"]["count"] == 1
    assert report["sleep_op"]["total_seconds"] >= 0.01


def test_disabled_profiler_records_nothing():
    p = Profiler(enabled=False)
    with p.track("noop"):
        pass
    p.record("also_noop", 1.0)
    assert p.report() == {}


def test_reset_clears_records():
    p = Profiler()
    p.record("a", 1.0)
    p.reset()
    assert p.report() == {}


def test_multiple_names_tracked_independently():
    p = Profiler()
    p.record("a", 1.0)
    p.record("b", 2.0)
    report = p.report()
    assert set(report.keys()) == {"a", "b"}
