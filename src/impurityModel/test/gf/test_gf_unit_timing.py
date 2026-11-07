"""The unit loop's tail summary (S5.1): where the wall time of a GF unit loop went."""

from impurityModel.ed.gf_units import format_unit_timing


def test_the_summary_names_the_makespan_the_busy_share_and_the_longest_unit():
    # colour 0 ran units 3 and 1 back to back (0-10, 10-14); colour 1 ran unit 0 for 2 s and then idled
    records = [(3, 0.0, 10.0), (1, 10.0, 14.0), (0, 0.0, 2.0)]
    lines = format_unit_timing(records, [2, 1])
    assert len(lines) == 1
    assert "makespan 14.0 s" in lines[0] and "['100%', '14%']" in lines[0]
    assert "#3 10.0 s (71% of the makespan) on colour 0" in lines[0]


def test_verbose_adds_one_line_per_unit_with_its_colour():
    records = [(3, 0.0, 10.0), (0, 0.0, 2.0)]
    lines = format_unit_timing(records, [1, 1], verbose=True)
    assert len(lines) == 3 and "unit #0: colour 1" in lines[2]


def test_no_units_no_lines():
    assert format_unit_timing([], [0, 0]) == []
