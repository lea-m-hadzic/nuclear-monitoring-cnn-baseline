"""Lock in the reproduction of the latest manuscript (Manuscript 5.docx) facility-level numbers."""
import pytest

from reproduce_manuscript import reproduce

# Narrative claims that the workbook does NOT support; documented, expected to stay unmatched until
# the authors respond. If one starts matching, update this list deliberately.
KNOWN_DISCREPANCIES = {("Results text Saratov z17", "missed silos with a 'cut off' label")}


@pytest.fixture(scope="module")
def rep(cfg):
    return reproduce(cfg)


def test_all_table_values_reproduced(rep):
    tables = rep[rep.source.str.startswith(("Table 2", "Table 3", "Table 4"))]
    assert len(tables) >= 60
    bad = tables[~tables.match]
    assert bad.empty, bad.to_string()


def test_results_text_counts_reproduced(rep):
    text = rep[rep.source.str.startswith("Results text")]
    bad = {(r.source, r.quantity) for r in text[~text.match].itertuples()}
    assert bad == KNOWN_DISCREPANCIES


@pytest.mark.parametrize("key,surfaced", [
    (("Saratov", 16, "single_pass"), 9), (("Saratov", 16, "adaptive"), 13),
    (("Saratov", 17, "single_pass"), 11), (("Saratov", 17, "adaptive"), 18),
    (("Natanz", 16, "adaptive"), 1), (("Fordow", 16, "adaptive"), 1)])
def test_facility_counts(rep, key, surfaced):
    region, zoom, wf = key
    row = rep[(rep.source.str.endswith(f"{region} z{zoom} {wf}")) & (rep.quantity == "facilities surfaced")]
    assert len(row) == 1 and row.reproduced.iloc[0] == surfaced


def test_table3_threshold_sensitivity(rep):
    t3 = rep[rep.source.str.startswith("Table 3")]
    assert len(t3) == 7 * 4 and t3.match.all()
