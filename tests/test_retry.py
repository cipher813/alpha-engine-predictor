"""Tests for retry.py decorator."""

from unittest.mock import patch

import pytest

from retry import retry


class TestRetry:
    @patch("retry.time.sleep")
    def test_succeeds_first_try(self, mock_sleep):
        @retry(max_attempts=3)
        def good():
            return "ok"
        assert good() == "ok"
        mock_sleep.assert_not_called()

    @patch("retry.time.sleep")
    def test_retries_then_succeeds(self, mock_sleep):
        count = {"n": 0}

        @retry(max_attempts=3, backoff_base=1)
        def flaky():
            count["n"] += 1
            if count["n"] < 3:
                raise ValueError("transient")
            return "ok"

        assert flaky() == "ok"
        assert count["n"] == 3
        assert mock_sleep.call_count == 2

    @patch("retry.time.sleep")
    def test_exhausts_retries(self, mock_sleep):
        @retry(max_attempts=2, backoff_base=1)
        def always_fail():
            raise RuntimeError("permanent")

        with pytest.raises(RuntimeError, match="permanent"):
            always_fail()

    @patch("retry.time.sleep")
    def test_custom_retryable(self, mock_sleep):
        @retry(max_attempts=2, retryable=(ValueError,))
        def raises_wrong_type():
            raise TypeError("not retryable")

        with pytest.raises(TypeError):
            raises_wrong_type()
        mock_sleep.assert_not_called()

    @patch("retry.time.sleep")
    def test_custom_label(self, mock_sleep):
        @retry(max_attempts=2, label="my_op")
        def fail():
            raise ValueError("oops")

        with pytest.raises(ValueError):
            fail()
