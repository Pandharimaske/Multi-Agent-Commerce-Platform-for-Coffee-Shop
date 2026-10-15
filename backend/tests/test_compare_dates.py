"""A midnight timestamp is the same value as its date; a real time of day is not."""
import datetime as dt

from evaluation.compare import normalize_value, results_match


def test_midnight_timestamp_equals_date():
    assert normalize_value("2026-08-01T00:00:00") == normalize_value("2026-08-01")
    assert normalize_value("2026-08-01 00:00:00+00:00") == "2026-08-01"
    assert normalize_value(dt.datetime(2026, 8, 1)) == normalize_value(dt.date(2026, 8, 1))


def test_real_time_of_day_still_differs():
    assert normalize_value("2026-08-01T10:30:00") != normalize_value("2026-08-01")
    assert normalize_value(dt.datetime(2026, 8, 1, 10, 30)) != normalize_value(dt.date(2026, 8, 1))


def test_m06_shape_now_matches():
    ref = [{"month": "2026-08-01", "revenue": 85780}, {"month": "2026-09-01", "revenue": 109375}]
    got = [{"month_start": "2026-08-01T00:00:00", "revenue": 85780.0}, {"month_start": "2026-09-01T00:00:00", "revenue": 109375.0}]
    assert results_match(ref, got)
    assert not results_match(ref, [{"month_start": "2026-08-02T00:00:00", "revenue": 85780}, got[1]])
