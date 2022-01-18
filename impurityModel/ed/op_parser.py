def skip_whitespaces(str):
    for i, c in enumerate(str):
        if c not in [" ", "\t", "\n"]:
            break
    return str[i:]


def read_state_tuple(str):
    res = []
    num = ""
    for i,c in enumerate(str):
        if c == ')':
            try:
                res.append(int(num))
            except:
                print ("Error casting string " + num + ", to integer.")
                raise
            break
        elif c == ',':
            try:
                res.append(int(num))
            except:
                print ("Error casting string " + num + ", to integer.")
                raise
            num = ""
        elif c in "0123456789-":
            num += c
    return (str[i + 1 :], tuple(res))


# def read_complex(str):
#     num = ""
#     for i, c in enumerate(str):
#         if c == ',':
#             break
#         elif c in "0123456789+-ij.eE":
#             num += c
#     try:
#         val = complex(num.replace('i', 'j'))
#     except:
#         print ("Error casting string " + num + ", to complex number.")
#         raise
#     return (str[i+1:], val)

def read_real(str):
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
        print ("Error casting string " + num + ", to floating point number.")
        raise
    return (str[i:], val)


def read_real_imag(str):
    num = ""
    real = 0
    imag = 0
    str = skip_whitespaces(str)
    str, real = read_real(str)
    str = skip_whitespaces(str)
    str, imag = read_real(str)

    try:
        val = complex(real, imag)
    except:
        print ("Error casting numbers (" + repr(real) + ", " + repr(imag) + ") to floating point number.")
        raise
    return val


def extract_operator(line):
    res = {}
    states = []
    line, state1 = read_state_tuple(line)
    line = skip_whitespaces(line)
    line, state2 = read_state_tuple(line)
    amp = read_real_imag(line)
    return (((state1, 'c'), (state2, 'a')), amp)


# def extract_operators(string):
    # res = []
    # remainder = string
    # while remainder:
    #     remainder = skip_whitespaces(remainder)
    #     key = []
    #     while remainder and remainder[0] != ':':
    #         if remainder[0] == ',':
    #             remainder = skip_whitespaces(remainder[1:])
    #         op = remainder[0]
    #         if op not in "ac":
    #             raise RuntimeError("Operator, " + op +" must be either a or c.")
    #         remainder = skip_whitespaces(remainder[1:])
    #         assert(remainder[0] == '(')
    #         if remainder[0] not in "(":
    #             raise RuntimeError("Tuple designating state must follow operator."
    #             " Tuples start with (, not " + remainder[0])
    #         remainder, state = read_state_tuple(remainder)
    #         remainder = skip_whitespaces(remainder)
    #         if remainder[0] not in ':,':
    #             raise RuntimeError("Operators are separated by , or ended by :."
    #             " Not " + remainder[0])
    #         key.append((state, op))
    #     remainder, val = read_complex(remainder)
    #     res.append((tuple(key), val))
    # return res

def read_name(line):
    if line[0] != '(':
        return line
    return ""


def parse_file(filename):
    operators = {}
    with open(filename, 'r') as f:
        name = ""
        op = {}
        for linue_number, line in enumerate(f):
            line = line.strip()
            if line and line[0] != '#':

                if not name:
                    name = read_name(line)
                    if name:
                        continue
                    else:
                        name = "Op_" + repr(len(operators) + 1)

                # extracted_ops = extract_operators(line)
                # for key, val in extracted_ops:
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
    state, op = operator
    return op+repr(state)

def key_to_string(key):
    res = []
    for operator in key:
        res.append(operator_to_string(operator))
    return ", ".join(res)

def value_to_string(value):
    return repr(value)

def key_value_to_string(key, value):
    return key_to_string(key) + " : " + value_to_string(value) + "\n"

def write_operators_to_file(operators, filename):
    with open(filename, 'w+'):
        pass

    strings = []
    for operator in operators:
        s = ""
        for key, value in operator.items():
            s += key_value_to_string(key, value)
        strings.append(s)
    with open(filename, 'a') as f:
        f.write("\n".join(strings))
