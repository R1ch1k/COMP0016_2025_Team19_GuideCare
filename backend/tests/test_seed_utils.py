"""
Tests for app.seed and app.orchestration.utils — no live DB or LLM required.
"""

import asyncio
import datetime
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# ── seed.py ───────────────────────────────────────────────────────────────────

class TestCalculateAge:
    def test_birthday_already_passed_this_year(self):
        from app.seed import calculate_age
        # Person born Jan 1, 1990 → age is current year - 1990 (bday already passed)
        dob = datetime.date(1990, 1, 1)
        age = calculate_age(dob)
        today = datetime.date.today()
        expected = today.year - 1990 - ((today.month, today.day) < (1, 1))
        assert age == expected

    def test_birthday_not_yet_this_year(self):
        from app.seed import calculate_age
        # Born Dec 31 — birthday has not passed yet for most of the year
        dob = datetime.date(1985, 12, 31)
        age = calculate_age(dob)
        today = datetime.date.today()
        expected = today.year - 1985 - ((today.month, today.day) < (12, 31))
        assert age == expected

    def test_young_patient(self):
        from app.seed import calculate_age
        dob = datetime.date(2000, 6, 15)
        age = calculate_age(dob)
        assert 20 <= age <= 30  # Sanity range for 2024/2025

    def test_returns_integer(self):
        from app.seed import calculate_age
        assert isinstance(calculate_age(datetime.date(1970, 3, 1)), int)


class TestSeedIfEmpty:
    @pytest.mark.asyncio
    async def test_skips_when_patients_already_exist(self):
        """seed_if_empty should do nothing if a patient already exists."""
        mock_patient = MagicMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = mock_patient

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()

        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=mock_db)
        cm.__aexit__ = AsyncMock(return_value=False)

        with patch("app.seed.AsyncSessionLocal", return_value=cm):
            from app.seed import seed_if_empty
            await seed_if_empty()

        # DB commit should NOT be called (no seeding needed)
        mock_db.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_inserts_sample_patients_when_empty(self):
        """seed_if_empty should insert SAMPLE_PATIENTS when DB is empty."""
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = None  # empty DB

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.add = MagicMock()
        mock_db.commit = AsyncMock()

        cm = AsyncMock()
        cm.__aenter__ = AsyncMock(return_value=mock_db)
        cm.__aexit__ = AsyncMock(return_value=False)

        with patch("app.seed.AsyncSessionLocal", return_value=cm):
            from app.seed import seed_if_empty, SAMPLE_PATIENTS
            await seed_if_empty()

        # Should add one Patient per SAMPLE_PATIENTS entry
        assert mock_db.add.call_count == len(SAMPLE_PATIENTS)
        mock_db.commit.assert_awaited_once()


# ── orchestration/utils.py ───────────────────────────────────────────────────

class TestWithRetryTimeout:
    @pytest.mark.asyncio
    async def test_returns_result_on_success(self):
        from app.orchestration.utils import with_retry_timeout

        async def fast_fn():
            return "ok"

        result = await with_retry_timeout(fast_fn, timeout=5.0)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_passes_args_and_kwargs(self):
        from app.orchestration.utils import with_retry_timeout

        async def add(a, b):
            return a + b

        result = await with_retry_timeout(add, 3, 4, timeout=5.0)
        assert result == 7

    @pytest.mark.asyncio
    async def test_raises_on_timeout(self):
        from app.orchestration.utils import with_retry_timeout

        async def slow_fn():
            await asyncio.sleep(10)

        with pytest.raises(asyncio.TimeoutError):
            await with_retry_timeout(slow_fn, timeout=0.01, retries=0)

    @pytest.mark.asyncio
    async def test_retries_on_failure_then_succeeds(self):
        from app.orchestration.utils import with_retry_timeout

        call_count = {"n": 0}

        async def flaky():
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise ValueError("first failure")
            return "success"

        result = await with_retry_timeout(flaky, timeout=5.0, retries=1)
        assert result == "success"
        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_raises_after_all_retries_exhausted(self):
        from app.orchestration.utils import with_retry_timeout

        async def always_fails():
            raise RuntimeError("always fails")

        with pytest.raises(RuntimeError, match="always fails"):
            await with_retry_timeout(always_fails, timeout=5.0, retries=2)

    def test_log_step_does_not_raise(self):
        from app.orchestration.utils import log_step
        # Should not raise regardless of input
        log_step("conv-123", "triage", urgency="moderate", guideline="NG84")
        log_step("conv-456", "extract")
