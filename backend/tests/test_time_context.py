"""Shop clock helpers: the time of day must follow IST, not the server's timezone."""
from src.utils.util import IST, now_ist_text, time_of_day


def test_time_of_day_boundaries():
    assert time_of_day(5) == "morning"
    assert time_of_day(11) == "morning"
    assert time_of_day(12) == "afternoon"
    assert time_of_day(16) == "afternoon"
    assert time_of_day(17) == "evening"
    assert time_of_day(20) == "evening"
    assert time_of_day(21) == "night"
    assert time_of_day(2) == "night"


def test_ist_offset_is_five_thirty():
    assert IST.utcoffset(None).total_seconds() == 5.5 * 3600


def test_now_text_mentions_ist_and_a_period():
    text = now_ist_text()
    assert "IST" in text
    assert any(p in text for p in ("morning", "afternoon", "evening", "night"))
