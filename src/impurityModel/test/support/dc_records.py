"""Read a double-counting record back out of captured stdout.

Deliberately a dumb line parser rather than a helper shared with
:mod:`impurityModel.ed.dc_record`'s writer: the point of the format is that someone can ``awk`` it
out of an RSPt log, and a test that reused the writer's own structure would keep passing however
unparseable the text became.
"""

from impurityModel.ed import dc_record


def parse_record(out):
    """The last record in ``out`` as ``{key: value-with-annotation}``; raises if there is none.

    Values are the raw text after the ``=``, annotation included -- ``sector`` comes back as
    ``"8   (nominal 8)  OK"``. Callers that want a number convert it themselves, which keeps the
    annotations under test instead of parsed away.
    """
    lines = out.splitlines()
    if dc_record.RECORD_HEADER not in lines:
        # Assert rather than return None. Callers subscript the result, so a missing record used to
        # surface as `TypeError: 'NoneType' object is not subscriptable` three frames away from the
        # thing that actually went wrong, which is that no record was emitted at all.
        raise AssertionError(f"no double-counting record was emitted:\n{out}")
    start = len(lines) - 1 - lines[::-1].index(dc_record.RECORD_HEADER)
    record = {}
    for line in lines[start + 1 :]:
        if line == dc_record.RECORD_FOOTER:
            return record
        key, _, value = line.partition("=")
        record[key.strip()] = value.strip()
    raise AssertionError(f"record header with no footer:\n{out}")


def records_in(out):
    """How many records ``out`` contains -- one per double-counting call, and never two."""
    return out.count(dc_record.RECORD_HEADER)
