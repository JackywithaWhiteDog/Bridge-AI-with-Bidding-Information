from dataclasses import dataclass
from typing import ClassVar, List

@dataclass
class Suit:
    SPADES: ClassVar[int] = 0
    HEARTS: ClassVar[int] = 1
    DIAMONDS: ClassVar[int] = 2
    CLUBS: ClassVar[int] = 3
    NT: ClassVar[int] = 4

    names: ClassVar[List[str]] = ["Spades", "Hearts", "Diamonds", "Clubs", "No Trump"]
    simple_names: ClassVar[List[str]] = ['♠', '♥', '♦', '♣', "NT"]

    @staticmethod
    def to_list(nt=False):
        result = [Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]
        if nt:
            result.append(Suit.NT)
        return result

    @staticmethod
    def idx2str(suit, simple=True):
        if simple:
            return Suit.simple_names[suit]
        return Suit.names[suit]

@dataclass
class Side:
    NORTH: int = 0
    EAST: int = 1
    SOUTH: int = 2
    WEST: int = 3

    names: ClassVar[List[str]] = ["North", "East", "South", "West"]
    simple_names: ClassVar[List[str]] = ["N", "E", "S", "W"]

    @staticmethod
    def get_left(side):
        return (side + 1) % 4

    @staticmethod
    def get_right(side):
        return (side + 3) % 4

    @staticmethod
    def get_partner(side):
        return (side + 2) % 4

    @staticmethod
    def to_list():
        return [Side.NORTH, Side.EAST, Side.SOUTH, Side.WEST]

    @staticmethod
    def idx2str(suit, simple=False):
        if simple:
            return Side.simple_names[suit]
        return Side.names[suit]

@dataclass
class Rank:
    names: ClassVar[List[str]] = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

    @staticmethod
    def idx2str(rank):
        '''rank = 2~14'''
        return Rank.names[int(rank - 2)]
