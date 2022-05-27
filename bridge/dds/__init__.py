from typing import Union, Literal, List, Dict, Tuple
import math
import ctypes
from ctypes import c_int, c_char

from bridge.hand import Card
from bridge.dds.struct import DealPBN, BoardsPBN, SolvedBoards, FutureTricks, MAXNOOFBOARDS
from bridge.dds.utils import set_max_threads, solve_all_boards, error_message, RETURN_NO_FAULT

set_max_threads(0)

def validate_future_trick(future_tricks: FutureTricks, target: int, solution: int, num_legal_actions: int) -> bool:
    """Return true if the future tricks are legal"""
    if solution == 1:
        # Return only one card
        if future_tricks.contents.cards != 1:
            return False
    elif solution == 2 and (target > 0 or target == -1):
        # Return cards meeting the target or only one card if target cannot be achieved
        if future_tricks.contents.cards == 0:
            return False
    elif solution == 3 or (solution == 2 and target == 0):
        # Return all card
        if future_tricks.contents.cards != num_legal_actions:
            return False
    else:
        return False
    return True

def dds_score(
    trump: Literal[0, 1, 2, 3, 4],
    first: Literal[0, 1, 2, 3],
    current_cards: List[Card],
    pbn_hands: Union[List[str], str],
    num_legal_actions: int,
    reduction: Literal['mean', 'max', 'min']='mean',
    solution: int=2
) -> Dict[Tuple[Literal[0, 1, 2, 3], Literal[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]], float]:
    current_suits = [0] * 3
    current_ranks = [0] * 3
    for i, card in enumerate(current_cards):
        current_suits[i] = card.suit
        current_ranks[i] = card.rank

    if not isinstance(pbn_hands, list):
        pbn_hands = [pbn_hands]
    total_boards = len(pbn_hands)

    if reduction == 'mean':
        result = dict()
    elif reduction == 'max':
        result = dict()
    elif reduction == 'min':
        result = dict()
    else:
        raise ValueError(f"Reduction {reduction} not supported, should be one of mean, max, and min")

    for start_idx in range(0, total_boards, MAXNOOFBOARDS):
        end_idx = min(start_idx + MAXNOOFBOARDS, total_boards)
        no_of_boards = int(end_idx - start_idx)
        deals = []
        for pbn in pbn_hands[start_idx:end_idx]:
            deal = DealPBN()
            deal.trump = trump
            deal.first = first
            deal.currentTrickSuit = (c_int * 3)(*current_suits)
            deal.currentTrickRank = (c_int * 3)(*current_ranks)
            deal.remainCards = pbn.encode("utf-8")
            deals.append(deal)
        targets = [-1] * no_of_boards
        solutions = [solution] * no_of_boards
        modes = [0] * no_of_boards
        
        boards = BoardsPBN()
        boards.noOfBoards = no_of_boards
        boards.deals = (DealPBN * MAXNOOFBOARDS)(*deals)
        boards.target = (c_int * MAXNOOFBOARDS)(*targets)
        boards.solutions = (c_int * MAXNOOFBOARDS)(*solutions)
        boards.mode = (c_int * MAXNOOFBOARDS)(*modes)
        solved_boards = SolvedBoards()
        res = solve_all_boards(ctypes.pointer(boards), ctypes.pointer(solved_boards))
        
        if res != RETURN_NO_FAULT:
            error = ctypes.create_string_buffer(80)
            error_message(res, error)
            raise RuntimeError("DDS Error: {}".format(error.value.decode("utf-8")))

        for i in range(no_of_boards):
            future_tricks = ctypes.pointer(solved_boards.solvedBoards[i])
            if not validate_future_trick(future_tricks , targets[i], solutions[i], num_legal_actions):
                # suits = [future_tricks.contents.suit[i] for i in range(future_tricks.contents.cards)]
                # ranks = [future_tricks.contents.rank[i] for i in range(future_tricks.contents.cards)]
                # scores = [future_tricks.contents.score[i] for i in range(future_tricks.contents.cards)]
                # from bridge.hand import Hand
                # tmp = Hand(remain_cards=[Card(suit=int(suit), rank=int(rank)) for suit, rank, score in zip(suits, ranks, scores)])
                # print(f"PBN: {pbn_hands[start_idx+i]}")
                # print(f"Return cards: {tmp.to_pbn()}")
                raise RuntimeError("Illegal DDS returns")

        for i in range(no_of_boards):
            future_tricks = ctypes.pointer(solved_boards.solvedBoards[i])
            suits = [future_tricks.contents.suit[i] for i in range(future_tricks.contents.cards)]
            ranks = [future_tricks.contents.rank[i] for i in range(future_tricks.contents.cards)]
            scores = [future_tricks.contents.score[i] for i in range(future_tricks.contents.cards)]
            for suit, rank, score in zip(suits, ranks, scores):
                suit = int(suit)
                rank = int(rank)
                if (suit, rank) not in result:
                    if reduction == 'mean':
                        result[(suit, rank)] = score / total_boards
                    elif reduction in ['min', 'max']:
                        result[(suit, rank)] = score
                else:
                    if reduction == 'mean':
                        result[(suit, rank)] += score / total_boards
                    elif reduction == 'max':
                        result[(suit, rank)] = max(result[(suit, rank)], score)
                    elif reduction == 'min':
                        result[(suit, rank)] = min(result[(suit, rank)], score)

    return result
