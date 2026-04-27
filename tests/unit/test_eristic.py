"""Unit tests for the eristic-stratagem catalog.

The Devil's Advocate persona pulls from this catalog at construction time.
The tests below pin the data shape rather than the qualitative content of
each stratagem (which is curated text).
"""

from __future__ import annotations

import pytest

from agents.specialized.eristic import (
    STRATAGEMS,
    Stratagem,
    get_stratagem,
    random_stratagem,
)


class TestCatalog:
    def test_non_empty(self):
        assert len(STRATAGEMS) >= 5  # curated subset, ≥5 for variety

    def test_all_dataclass_instances(self):
        for s in STRATAGEMS:
            assert isinstance(s, Stratagem)
            assert s.id > 0
            assert s.name
            assert s.brief
            assert s.prompt_directive

    def test_unique_ids(self):
        ids = [s.id for s in STRATAGEMS]
        assert len(set(ids)) == len(ids)

    def test_directives_are_imperative(self):
        """Each directive should read like an instruction (avoid description)."""
        for s in STRATAGEMS:
            # heuristic: at least one verb-ish word
            words = s.prompt_directive.split()
            assert len(words) > 5  # non-trivially long


class TestGetStratagem:
    def test_lookup_known_id(self):
        for s in STRATAGEMS:
            got = get_stratagem(s.id)
            assert got is s

    def test_lookup_unknown_raises(self):
        with pytest.raises(KeyError, match="not in curated subset"):
            get_stratagem(99)


class TestRandomStratagem:
    def test_returns_from_catalog(self):
        s = random_stratagem()
        assert s in STRATAGEMS

    def test_exclude_respected(self):
        """Excluding all but one id pins the random pick."""
        all_ids = {s.id for s in STRATAGEMS}
        keep = next(iter(all_ids))
        exclude = all_ids - {keep}
        for _ in range(20):
            s = random_stratagem(exclude=exclude)
            assert s.id == keep

    def test_full_exclusion_falls_back_to_full_pool(self):
        """If every id is excluded, the picker still returns *something*
        (defensive against starvation in the DRTAG retry loop)."""
        all_ids = {s.id for s in STRATAGEMS}
        s = random_stratagem(exclude=all_ids)
        assert s in STRATAGEMS
