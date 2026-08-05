_WHITESPACE = " \t\n\r"
_DIGITS = "0123456789-"
_NUMBER_CHARS = "0123456789+-.eE"


def skip_whitespaces(text):
    """
    Remove leading whitespaces, tabs, and newlines from a string.

    Parameters
    ----------
    text : str
        The input string.

    Returns
    -------
    str
        The string with leading whitespaces removed.

    Examples
    --------
    >>> skip_whitespaces("   hello  ")
    'hello  '
    >>> skip_whitespaces("   ")
    ''
    """
    return text.lstrip(_WHITESPACE)


def read_state_tuple(text):
    """
    Parse a tuple of integers representing a state from the string.

    Reads characters up to the closing parenthesis ')', converting
    comma-separated values to a tuple of integers.

    Parameters
    ----------
    text : str
        The input string starting with elements of the tuple, e.g., "1, 2) remaining".

    Returns
    -------
    tuple[str, tuple[int, ...]]
        A tuple containing the remaining string after the closing parenthesis,
        and the parsed state tuple.

    Raises
    ------
    ValueError
        If no closing ``)`` is present, or a segment is not an integer. The message
        names what was expected; ``parse_file`` adds the file and line number.

    Examples
    --------
    >>> read_state_tuple("1, 2) residue")
    (' residue', (1, 2))
    >>> read_state_tuple("2,) residue")
    (' residue', (2,))
    """

    def to_int(segment, allow_empty):
        # A trailing comma before ')' -- the singleton form "(2,)" -- yields an empty
        # final segment, which is not an error.
        if not segment and allow_empty:
            return None
        try:
            return int(segment)
        except ValueError:
            raise ValueError(f"expected an integer in a state tuple, got {segment!r} in {text!r}") from None

    res = []
    num = ""
    for i, c in enumerate(text):
        if c == ")":
            value = to_int(num, allow_empty=True)
            if value is not None:
                res.append(value)
            return (text[i + 1 :], tuple(res))
        elif c == ",":
            res.append(to_int(num, allow_empty=False))
            num = ""
        elif c in _DIGITS:
            num += c
    raise ValueError(f"unterminated state tuple: expected a closing ')' in {text!r}")


def read_real(text):
    """
    Parse a floating-point number from the beginning of the string.

    Skips leading whitespaces and reads characters corresponding to a real number.

    Parameters
    ----------
    text : str
        The input string.

    Returns
    -------
    tuple[str, float]
        A tuple of the remaining string and the parsed float value.

    Raises
    ------
    ValueError
        If no floating-point number is present at the start of the string.

    Examples
    --------
    >>> read_real("   -1.23e-4 remainder")
    (' remainder', -0.000123)
    >>> read_real("1.5")
    ('', 1.5)
    """
    text = skip_whitespaces(text)
    end = 0
    while end < len(text) and text[end] in _NUMBER_CHARS:
        end += 1
    num = text[:end]
    try:
        val = float(num)
    except ValueError:
        raise ValueError(f"expected a real number, got {text!r}") from None
    return (text[end:], val)


def read_real_imag(text):
    """
    Parse real and imaginary parts from a string to form a complex number.

    Parameters
    ----------
    text : str
        The input string containing real and imaginary values separated by whitespace.

    Returns
    -------
    complex
        The parsed complex number.

    Examples
    --------
    >>> read_real_imag(" 1.5 -2.0")
    (1.5-2j)
    """
    text, real = read_real(skip_whitespaces(text))
    _, imag = read_real(skip_whitespaces(text))
    return complex(real, imag)


def extract_operator(line):
    """
    Parse a single line representing a creation-annihilation operator pair and its amplitude.

    The line must format the two state tuples first, followed by the complex amplitude
    (real and imaginary parts).

    Parameters
    ----------
    line : str
        The line to parse, e.g. "(1, 2) (3, 4) 1.0 -0.5".

    Returns
    -------
    tuple[tuple[tuple[tuple[int, ...], str], tuple[tuple[int, ...], str]], complex]
        A tuple containing the parsed operator descriptor key and the complex amplitude value.

    Raises
    ------
    ValueError
        If the line is not in this format -- notably for the bare-integer flat-index form
        ``"  0   0 -4.12 0.0"`` written by ``rspt2spectra``, which needs
        :func:`impurityModel.ed.h0_format.read_legacy_flat_h0` instead.

    Examples
    --------
    >>> extract_operator("(2,) (1,) 1.5 0.0")
    ((((2,), 'c'), ((1,), 'a')), (1.5+0j))
    """
    line, state1 = read_state_tuple(line)
    line = skip_whitespaces(line)
    line, state2 = read_state_tuple(line)
    amp = read_real_imag(line)
    return (((state1, "c"), (state2, "a")), amp)


def read_name(line):
    """
    Return the operator name from a line if it is not a state specification.

    If the line starts with '(', it indicates the beginning of an operator definition line,
    so this function returns an empty string. Otherwise, returns the line itself (the name).

    Parameters
    ----------
    line : str
        The input line.

    Returns
    -------
    str
        The operator name, or an empty string.
    """
    if not line.startswith("("):
        return line
    return ""


def parse_file(filename):
    """
    Parse a file containing multiple operator definitions.

    Reads sections separated by blank lines. Each section starts with an optional name,
    followed by lines defining creation/annihilation operators and amplitudes.

    Parameters
    ----------
    filename : str
        Path to the file to parse.

    Returns
    -------
    dict[str, dict[tuple, complex]]
        A dictionary mapping operator names to dictionaries of their constituent operators and amplitudes.
    """
    operators = {}
    with open(filename, "r") as f:
        name = ""
        op = {}
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if line and line[0] != "#":
                if not name:
                    name = read_name(line)
                    if name:
                        continue
                    else:
                        name = "Op_" + repr(len(operators) + 1)

                # The leaf parsers know nothing about where the text came from, so the file
                # and line number are attached here. Without this a format mismatch reports a
                # bare parse error against a line the caller cannot locate -- which is exactly
                # how the rspt2spectra flat-index format went undiagnosed.
                try:
                    key, val = extract_operator(line)
                except ValueError as exc:
                    raise ValueError(
                        f"{filename}:{lineno}: expected '(i, ...) (j, ...) re im', got {line!r} ({exc})"
                    ) from None
                if key in op:
                    op[key] += val
                else:
                    op[key] = val
            elif not line and op:
                operators[name] = op
                op = {}
                name = ""
    if op:
        operators[name] = op
    return operators
