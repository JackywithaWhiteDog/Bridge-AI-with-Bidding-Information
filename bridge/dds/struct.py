import ctypes
from ctypes import c_int, c_char

MAXNOOFBOARDS = 200

class DealPBN(ctypes.Structure):
    _fields_ = [
        ("trump", c_int),
        ("first", c_int),
        ("currentTrickSuit", c_int * 3),
        ("currentTrickRank", c_int * 3),
        ("remainCards", c_char * 80)
    ]

class BoardsPBN(ctypes.Structure):
    _fields_ = [
        ("noOfBoards", c_int),
        ("deals", DealPBN * MAXNOOFBOARDS),
        ("target", c_int * MAXNOOFBOARDS),
        ("solutions", c_int * MAXNOOFBOARDS),
        ("mode", c_int * MAXNOOFBOARDS)
    ]

class FutureTricks(ctypes.Structure):
    _fields_ = [
        ("nodes", c_int),
        ("cards", c_int),
        ("suit", c_int * 13),
        ("rank", c_int * 13),
        ("equals", c_int * 13),
        ("score", c_int * 13)
    ]

class SolvedBoards(ctypes.Structure):
    _fields_ = [
        ("noOfBoards", c_int),
        ("solvedBoards", FutureTricks * MAXNOOFBOARDS)
    ]