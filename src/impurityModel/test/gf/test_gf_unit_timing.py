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


def test_rank_seconds_and_utilisation_weight_each_colour_by_its_width():
    # colour 0: 3 ranks busy 10 s of a 10 s makespan; colour 1: 1 rank busy 2 s -> 32 of 40 rank-s
    records = [(0, 0.0, 10.0), (1, 0.0, 2.0)]
    (line,) = format_unit_timing(records, [1, 1], colour_ranks=[3, 1])
    assert "32 rank-s of work, 80% utilisation" in line
    # unweighted fallback: every colour counts once
    (line,) = format_unit_timing(records, [1, 1])
    assert "12 rank-s of work, 60% utilisation" in line


def test_the_per_unit_lines_carry_each_units_passes_and_exit_status_even_when_not_verbose():
    passes = dict(
        growth_blocks=32769, growth_seconds=19500.0, growth_stop="frozen", restart_blocks=65529,
        restart_seconds=11800.0, status="plateau",
    )  # fmt: skip
    records = [(0, 0.0, 10.0), (5, 0.0, 39747.0, passes)]
    lines = format_unit_timing(records, [2])
    assert len(lines) == 3 and "longest unit #5" in lines[0]
    assert lines[2].endswith("growth 32,769 blk/19500.0 s (frozen), restart 65,529 blk/11800.0 s, status plateau")
    assert lines[1].endswith("(10.0 s)")  # a unit that reported nothing prints no passes
