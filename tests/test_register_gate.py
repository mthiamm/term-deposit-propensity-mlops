import pytest

from src.register import MIN_DELTA, should_promote


@pytest.mark.parametrize(
    "candidate, champion, expected",
    [
        (0.40, None, True),  # no champion yet: always promote
        (0.50, 0.48, True),  # clearly better
        (0.481, 0.48, False),  # better, but inside the margin: not worth the churn
        (0.48, 0.48, False),  # tie
        (0.47, 0.48, False),  # worse
    ],
)
def test_should_promote(candidate, champion, expected):
    assert should_promote(candidate, champion) is expected


def test_margin_is_respected():
    # just above and just below champion + MIN_DELTA
    assert should_promote(0.48 + MIN_DELTA + 1e-4, 0.48)
    assert not should_promote(0.48 + MIN_DELTA - 1e-4, 0.48)


def test_custom_min_delta():
    assert should_promote(0.4801, 0.48, min_delta=0.0)
