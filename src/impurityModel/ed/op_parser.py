def skip_whitespaces(str):
    """
    Remove leading whitespaces, tabs, and newlines from a string.

    Parameters
    ----------
    str : str
        The input string.

    Returns
    -------
    str
        The string with leading whitespaces removed.

    Examples
    --------
    >>> skip_whitespaces("   hello  ")
    'hello  '
    """
    for i, c in enumerate(str):
        if c not in [" ", "\t", "\n"]:
            break
    return str[i:]


def read_state_tuple(str):
    """
    Parse a tuple of integers representing a state from the string.

    Reads characters up to the closing parenthesis ')', converting
    comma-separated values to a tuple of integers.

    Parameters
    ----------
    str : str
        The input string starting with elements of the tuple, e.g., "1, 2) remaining".

    Returns
    -------
    tuple[str, tuple[int, ...]]
        A tuple containing the remaining string after the closing parenthesis,
        and the parsed state tuple.

    Examples
    --------
    >>> read_state_tuple("1, 2) residue")
    (' residue', (1, 2))
    """
    res = []
    num = ""
    for i, c in enumerate(str):
        if c == ")":
            try:
                res.append(int(num))
            except:
                print("Error casting string " + num + ", to integer.")
                raise
            break
        elif c == ",":
            try:
                res.append(int(num))
            except:
                print("Error casting string " + num + ", to integer.")
                raise
            num = ""
        elif c in "0123456789-":
            num += c
    return (str[i + 1 :], tuple(res))


def read_real(str):
    """
    Parse a floating-point number from the beginning of the string.

    Skips leading whitespaces and reads characters corresponding to a real number.

    Parameters
    ----------
    str : str
        The input string.

    Returns
    -------
    tuple[str, float]
        A tuple of the remaining string and the parsed float value.

    Examples
    --------
    >>> read_real("   -1.23e-4 remainder")
    (' remainder', -0.000123)
    """
    str = skip_whitespaces(str)
    num = ""
    val = 0
    for i, c in enumerate(str):
        if c in "0123456789+-.eE":
            num += c
        else:
            break
    try:
        val = float(num)
    except:
        print("Error casting string " + num + ", to floating point number.")
        raise
    return (str[i:], val)


def read_real_imag(str):
    """
    Parse real and imaginary parts from a string to form a complex number.

    Parameters
    ----------
    str : str
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
    real = 0
    imag = 0
    str = skip_whitespaces(str)
    str, real = read_real(str)
    str = skip_whitespaces(str)
    str, imag = read_real(str)

    try:
        val = complex(real, imag)
    except:
        print("Error casting numbers (" + repr(real) + ", " + repr(imag) + ") to complex number.")
        raise
    return val


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
    if line[0] != "(":
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
        for linue_number, line in enumerate(f):
            line = line.strip()
            if line and line[0] != "#":
                if not name:
                    name = read_name(line)
                    if name:
                        continue
                    else:
                        name = "Op_" + repr(len(operators) + 1)

                key, val = extract_operator(line)
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


def operator_to_string(operator):
    """
    Convert an individual creation/annihilation operator tuple to its string representation.

    Parameters
    ----------
    operator : tuple[tuple[int, ...], str]
        A tuple of (state_tuple, operator_type) where operator_type is 'c' or 'a'.

    Returns
    -------
    str
        String representation, e.g. "c(1, 2)".

    Examples
    --------
    >>> operator_to_string(((1, 2), "c"))
    'c(1, 2)'
    """
    state, op = operator
    return op + repr(state)


def key_to_string(key):
    """
    Convert a sequence of operators (a key) to a comma-separated string.

    Parameters
    ----------
    key : tuple[tuple[tuple[int, ...], str], ...]
        A sequence of operator descriptions.

    Returns
    -------
    str
        Comma-separated operator string.

    Examples
    --------
    >>> key_to_string((((1,), "c"), ((2,), "a")))
    'c(1,), a(2,)'
    """
    res = []
    for operator in key:
        res.append(operator_to_string(operator))
    return ", ".join(res)


def value_to_string(value):
    """
    Convert a complex amplitude value to its string representation.

    Parameters
    ----------
    value : complex or float
        The amplitude value.

    Returns
    -------
    str
        The string representation of the amplitude.
    """
    return repr(value)


def key_value_to_string(key, value):
    """
    Format a key-value operator pair as a line in the output format.

    Parameters
    ----------
    key : tuple
        The operator key.
    value : complex or float
        The operator amplitude.

    Returns
    -------
    str
        Formatted string line ending with a newline.
    """
    return key_to_string(key) + " : " + value_to_string(value) + "\n"


def write_operators_to_file(operators, filename):
    """
    Write a collection of operators to a file in a formatted text layout.

    Parameters
    ----------
    operators : list[dict[tuple, complex]]
        A list of dictionaries representing the operators.
    filename : str
        The output file path.
    """
    with open(filename, "w+"):
        pass

    strings = []
    for operator in operators:
        s = ""
        for key, value in operator.items():
            s += key_value_to_string(key, value)
        strings.append(s)
    with open(filename, "a") as f:
        f.write("\n".join(strings))
