import numpy as np
import pytest

from robust_scn import budgeted_worst_case_extra


def test_budgeted_worst_case_extra_integer_gamma() -> None:
    values = np.array([4.0, 10.0, 7.0, 1.0])
    assert budgeted_worst_case_extra(values, 0.0) == pytest.approx(0.0)
    assert budgeted_worst_case_extra(values, 1.0) == pytest.approx(10.0)
    assert budgeted_worst_case_extra(values, 2.0) == pytest.approx(17.0)
    assert budgeted_worst_case_extra(values, 99.0) == pytest.approx(22.0)


def test_budgeted_worst_case_extra_fractional_gamma() -> None:
    values = np.array([10.0, 7.0, 4.0])
    assert budgeted_worst_case_extra(values, 1.5) == pytest.approx(13.5)
