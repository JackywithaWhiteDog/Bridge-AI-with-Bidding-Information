import ctypes
from ctypes import c_int, c_char

from bridge.dds.struct import BoardsPBN, SolvedBoards

dds = ctypes.cdll.LoadLibrary("./lib/libdds.so")

RETURN_NO_FAULT = 1

# Functions
set_max_threads = dds.SetMaxThreads
"""
Arguments:
    int: user threads
Returns:
    None
"""
set_max_threads.argtypes = [c_int]
set_max_threads.restype = None

solve_all_boards = dds.SolveAllBoards
"""
Arguments:
    pointer to struct BoardsPBN: list of boards
    pointer to struct SolvedBoards: list of scores to cards for input boards
Returns
    int: return code
"""
solve_all_boards.argtypes = [ctypes.POINTER(BoardsPBN), ctypes.POINTER(SolvedBoards)]
solve_all_boards.restype = c_int

error_message = dds.ErrorMessage
"""
Arguments:
    int: code
    char *: 80
Returns
    None
"""
error_message.argtypes = [c_int, ctypes.POINTER(c_char)]
error_message.restype = None
