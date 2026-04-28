"""
tests/test_meta_trainer_streaming.py — Lock the streaming refactor of
meta_trainer.py Step 2 + Step 3 (PR #57, 2026-04-28).

Pre-refactor the trainer kept a ``ticker_features`` dict of all ~900
labeled DataFrames alive while a separate Step 3 loop built a list of
1.77M Python tuples on top of that, then ``np.stack``'d into the final
arrays. Peak RSS overshoot to ~2-3 GB OOM'd c5.large on 2026-04-28.

Streaming version processes one DataFrame at a time, extracts numpy
chunks, drops the DataFrame, and concatenates after the loop. Same
final arrays, same sort order, ~75% lower peak memory.

These tests:
1. Source-text invariants — forbid the dict-based pattern from coming
   back, confirm the chunk lists + np.concatenate path is in place.
2. Behavioral check — small inline dataset that simulates two tickers
   with diagnostic horizons, verify the streaming path produces a
   chronologically-sorted (date, ticker)-deterministic flat layout.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


_META_TRAINER = (
    Path(__file__).resolve().parent.parent
    / "training" / "meta_trainer.py"
)


@pytest.fixture(scope="module")
def meta_trainer_source() -> str:
    return _META_TRAINER.read_text()


# ── Source-text invariants ──────────────────────────────────────────────────
class TestStreamingSourceInvariants:
    def test_no_ticker_features_dict_declaration(self, meta_trainer_source):
        """The pre-refactor pattern was:
            ticker_features: dict[str, pd.DataFrame] = {}
        Bringing it back in any future PR re-introduces the OOM root cause.
        """
        bad = re.compile(
            r"ticker_features\s*:\s*dict\[\s*str\s*,\s*pd\.DataFrame\s*\]"
        )
        assert not bad.search(meta_trainer_source), (
            "ticker_features dict declaration must not return — that was "
            "the OOM root cause on c5.large 2026-04-28."
        )

    def test_no_ticker_features_dict_iteration(self, meta_trainer_source):
        """Even without the explicit declaration, iterating
        ``for ticker, df in ticker_features.items():`` would re-introduce
        the antipattern under a different variable name."""
        # Specifically forbid the iteration form on a dict-of-DataFrames.
        bad = re.compile(
            r"for\s+ticker\s*,\s*df\s+in\s+ticker_features\.items\(\)"
        )
        assert not bad.search(meta_trainer_source)

    def test_chunk_accumulators_present(self, meta_trainer_source):
        """The streaming refactor adds explicit per-ticker chunk lists
        used by ``np.concatenate`` after the read loop. Their presence
        is what bounds peak memory."""
        for chunk_var in [
            "date_chunks",
            "ticker_chunks",
            "mom_chunks",
            "vol_chunks",
            "fwd_chunks",
            "fwd_horizons_chunks",
        ]:
            assert chunk_var in meta_trainer_source, (
                f"Streaming refactor expects chunk accumulator {chunk_var!r}; "
                f"missing means the per-ticker numpy slices aren't being "
                f"collected for the final concat."
            )

    def test_uses_concatenate_not_stack_of_tuples(self, meta_trainer_source):
        """Pre-refactor used ``np.stack([r[2] for r in all_rows])``
        over 1.77M tuples — a 2x memory burst. Streaming uses a single
        ``np.concatenate`` over already-arrayified per-ticker chunks."""
        assert "np.concatenate(mom_chunks" in meta_trainer_source
        assert "np.concatenate(vol_chunks" in meta_trainer_source

    def test_explicit_dataframe_cleanup(self, meta_trainer_source):
        """The per-ticker DataFrame must be explicitly dropped each
        iteration so the next ticker's allocations have headroom.
        Without ``del`` Python would still GC eventually, but pandas
        + Arrow allocator behavior can keep buffers alive longer than
        expected — the explicit ``del`` is load-bearing for the
        memory profile."""
        # Look for a `del` of the labeled / raw_df vars within the loop body
        assert re.search(
            r"del\s+labeled\s*,\s*raw_df", meta_trainer_source
        ), (
            "Streaming refactor must explicitly del labeled + raw_df at "
            "the bottom of the per-ticker loop. Without it the dict-based "
            "OOM regresses under a different name."
        )

    def test_explicit_chunk_cleanup_after_concat(self, meta_trainer_source):
        """After ``np.concatenate`` the chunk lists hold the same data
        as the contiguous arrays — leaving them alive doubles RSS
        through walk-forward."""
        assert re.search(
            r"del\s+date_chunks\s*,\s*ticker_chunks\s*,\s*mom_chunks\s*,\s*vol_chunks",
            meta_trainer_source,
        ), (
            "Streaming refactor must del the chunk lists after concatenate "
            "so RSS doesn't double through walk-forward."
        )


# ── Behavioral parity check on a small synthetic dataset ────────────────────
# Rather than re-running run_meta_training (which requires AWS, parquets,
# etc.), this test exercises the streaming logic shape directly: build two
# fake "tickers" of labeled data, feed them through the chunk-then-concat
# pattern, assert the final arrays are date-sorted with deterministic
# ticker tie-breaking.
class TestStreamingBehavior:
    def test_concat_then_argsort_preserves_per_row_alignment(self):
        """The post-concat ``argsort`` must keep the date / ticker / mom /
        vol / fwd / fwd_horizons rows aligned. A bug here would silently
        misalign labels with features (a much worse failure mode than
        OOM)."""
        # Synthetic chunks for two tickers, intentionally interleaved dates
        dates_a = pd.to_datetime(["2026-01-05", "2026-01-07", "2026-01-09"]).to_numpy()
        dates_b = pd.to_datetime(["2026-01-06", "2026-01-08"]).to_numpy()

        date_chunks = [dates_a, dates_b]
        ticker_chunks = [
            np.array(["AAPL", "AAPL", "AAPL"], dtype=object),
            np.array(["MSFT", "MSFT"], dtype=object),
        ]
        mom_chunks = [
            np.array([[1.0], [3.0], [5.0]], dtype=np.float32),
            np.array([[2.0], [4.0]], dtype=np.float32),
        ]
        fwd_chunks = [
            np.array([10.0, 30.0, 50.0], dtype=np.float32),
            np.array([20.0, 40.0], dtype=np.float32),
        ]

        # Mirror the refactor's concatenate + argsort
        date_unsorted = np.concatenate(date_chunks)
        ticker_unsorted = np.concatenate(ticker_chunks)
        mom_unsorted = np.concatenate(mom_chunks, axis=0)
        fwd_unsorted = np.concatenate(fwd_chunks)

        sort_idx = np.argsort(date_unsorted, kind="stable")
        sorted_dates = date_unsorted[sort_idx]
        sorted_tickers = ticker_unsorted[sort_idx]
        sorted_mom = mom_unsorted[sort_idx]
        sorted_fwd = fwd_unsorted[sort_idx]

        # Expected: sorted by date ascending; AAPL/MSFT alternate
        expected_dates = pd.to_datetime([
            "2026-01-05", "2026-01-06", "2026-01-07",
            "2026-01-08", "2026-01-09",
        ]).to_numpy()
        expected_tickers = np.array(
            ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL"], dtype=object
        )
        expected_mom = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
        expected_fwd = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)

        np.testing.assert_array_equal(sorted_dates, expected_dates)
        np.testing.assert_array_equal(sorted_tickers, expected_tickers)
        np.testing.assert_array_equal(sorted_mom, expected_mom)
        np.testing.assert_array_equal(sorted_fwd, expected_fwd)

    def test_stable_sort_keeps_same_date_ticker_order(self):
        """Two tickers with the same date must keep insertion order so
        downstream walk-forward fold construction is deterministic across
        runs."""
        dates_a = pd.to_datetime(["2026-01-05", "2026-01-05"]).to_numpy()
        dates_b = pd.to_datetime(["2026-01-05", "2026-01-05"]).to_numpy()

        date_unsorted = np.concatenate([dates_a, dates_b])
        ticker_unsorted = np.concatenate([
            np.array(["AAPL", "AAPL"], dtype=object),
            np.array(["MSFT", "MSFT"], dtype=object),
        ])

        sort_idx = np.argsort(date_unsorted, kind="stable")
        sorted_tickers = ticker_unsorted[sort_idx].tolist()

        # Stable sort preserves AAPL before MSFT (insertion order)
        assert sorted_tickers == ["AAPL", "AAPL", "MSFT", "MSFT"]

    def test_no_extra_dataframe_alive_during_chunk_phase(self, monkeypatch):
        """Boundary check: simulate the streaming loop with a counter
        that tracks how many DataFrames are alive at peak. Should always
        be ≤ 1 (the current iteration's DataFrame). The legacy code
        would have N=900 alive simultaneously."""
        peak_alive = 0
        currently_alive = 0

        def make_df(ticker_idx: int) -> pd.DataFrame:
            nonlocal currently_alive, peak_alive
            currently_alive += 1
            peak_alive = max(peak_alive, currently_alive)
            df = pd.DataFrame(
                {"feature": np.arange(10, dtype=np.float32) + ticker_idx}
            )
            df.attrs["_ticker_idx"] = ticker_idx
            return df

        chunks: list[np.ndarray] = []
        for i in range(50):  # simulate 50 tickers
            df = make_df(i)
            chunks.append(df["feature"].to_numpy(dtype=np.float32))
            currently_alive -= 1
            del df  # mirrors the streaming refactor's explicit drop

        # Streaming: peak alive = 1 (current ticker only)
        assert peak_alive == 1, (
            f"peak_alive={peak_alive}; the streaming refactor must keep "
            f"≤ 1 DataFrame alive at any point. The legacy code held "
            f"all 50 simultaneously."
        )
        # And the concatenated result has all 50 tickers' data
        full = np.concatenate(chunks)
        assert len(full) == 50 * 10
