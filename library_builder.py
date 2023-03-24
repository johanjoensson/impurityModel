import cffi
ffibuilder = cffi.FFI()

with open('impmod_interface.h') as f:
    data = ''.join([line for line in f if not line.startswith('#')])
    ffibuilder.embedding_api(data)

ffibuilder.set_source("impmod_ed", r'''
    #include "impmod_interface.h"
''')

ffibuilder.embedding_init_code(r"""
    import impmod_interface
""")

ffibuilder.emit_c_code("impmod_interface.c")
