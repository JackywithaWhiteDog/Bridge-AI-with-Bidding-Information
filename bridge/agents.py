import abc
from dataclasses import dataclass
import random
from typing import List, Callable, Literal
import copy

import numpy as np

from bridge.state import State, Suit
from bridge.hand import Card, Hand, hands2pbn
from bridge.dds import dds_score
from bridge.dds.utils import set_max_threads
from bridge.bid_infer import Goren_bidding_infer

from ipdb import set_trace as st

@dataclass
class BaseAgent(abc.ABC):
    @abc.abstractmethod
    def get_action(self, state: State) -> Card:
        return

@dataclass
class RandomAgent(BaseAgent):
    def get_action(self, state: State) -> Card:
        actions = state.get_legal_actions()
        return random.choice(actions)

@dataclass
class DDSAgent(BaseAgent):
    max_threads: int=0

    def get_action(self, state: State) -> Card:
        set_max_threads(self.max_threads)
        actions = state.get_legal_actions()
        scores = dds_score(
            trump=state.trump,
            first=state.first,
            current_cards=state.current_cards,
            pbn_hands=hands2pbn(state.hands, state.declarer),
            num_legal_actions=len(actions),
            solution=1
        )
        suit, rank = next(iter(scores.keys()))
        result = Card(suit=suit, rank=rank)
        if result not in actions:
            print(result, actions)
            raise ValueError("DDS action not legal")
        return result

def get_unknown_sides(state: State) -> List[Literal[0, 1, 2, 3]]:
    if sum(state.tricks) == 0 and len(state.current_cards) == 0:
        # The first player of the first round doesn't know all the others' hands
        return [i for i in range(4) if i != state.current_player]
    if state.current_player in state.declarer_side:
        # Does not know the cards in defenders' hands
        # Since the declarer will play dummy's hand, the agent for dummy also know te declarer's hand
        return [
            (state.current_player + 1) % 4,
            (state.current_player + 3) % 4
        ]
    else:
        # Does not know the cards in partner's and declarer's hands
        return [
            (state.current_player + 2) % 4,
            state.declarer
        ]

def uniform_hands(state: State, n: int) -> List[List[Hand]]:
    unknown_sides = get_unknown_sides(state)
    unknown_cards = []
    for side in unknown_sides:
        unknown_cards += state.hands[side].remain_cards
    num_cards = [
        len(state.hands[side].remain_cards)
        for side in unknown_sides
    ]
    result = []
    for i in range(n):
        shuffled_cards = np.random.permutation(unknown_cards)
        hands = copy.deepcopy(state.hands)
        offset = 0
        for side, num in zip(unknown_sides, num_cards):
            hands[side] = Hand(remain_cards=shuffled_cards[offset:offset+num].tolist())
            offset += num
        result.append(hands)
    return result

def suit_hands(state: State, n: int) -> List[List[Hand]]:
    unknown_sides = get_unknown_sides(state)
    unknown_cards_list = [[] for i in range(4)]
    num_cards_list = [[0] * 4 for i in range(4)]
    for side in unknown_sides:
        for card in state.hands[side].remain_cards:
            unknown_cards_list[card.suit].append(card)
            num_cards_list[side][card.suit] += 1
    result = []
    for i in range(n):
        assigned_cards = [[] for i in range(len(unknown_sides))]  
        for num_cards, cards in zip(num_cards_list, unknown_cards_list):
            shuffled_cards = np.random.permutation(cards)
            offset = 0
            for i, num in enumerate(num_cards):
                assigned_cards[i] += shuffled_cards[offset:offset+num].tolist()
                offset += num
        hands = copy.deepcopy(state.hands)
        for side, cards in zip(unknown_sides, assigned_cards):
            hands[side] = Hand(remain_cards=cards)
        result.append(hands)
    return result


def suit_hands_backup(state: State, n: int) -> List[List[Hand]]:
    '''
    suit: ['♠', '♥', '♦', '♣']
    '''
    unknown_sides = get_unknown_sides(state)
    unknown_cards_list = [[] for i in range(4)]
    num_cards_list = [[0] * 4 for i in range(4)]
    for side in unknown_sides:
        for card in state.hands[side].remain_cards:
            unknown_cards_list[card.suit].append(card) # remaining hands card: list of ['♠', '♥', '♦', '♣']
            num_cards_list[side][card.suit] += 1 # suit number for each player, ex: [[0, 0, 0, 0], [4, 2, 4, 3], [5, 6, 1, 1], [2, 3, 4, 4]]
    result = []
    for i in range(n):
        assigned_cards = [[] for i in range(len(unknown_sides))] # add range
        for ii, cards in enumerate(unknown_cards_list): # per suit
            shuffled_cards = np.random.permutation(cards)
            offset = 0
            for iii, side_index in enumerate(unknown_sides):
                num = num_cards_list[side_index][ii]
                assigned_cards[iii] += shuffled_cards[offset:offset+num].tolist()
                offset += num
        hands = copy.deepcopy(state.hands)
        for side, cards in zip(unknown_sides, assigned_cards):
            hands[side] = Hand(remain_cards=cards)
        result.append(hands)
    return result


def bidding_inference_hands(state: State, n: int) -> List[List[Hand]]:
    '''
    suit: ['♠', '♥', '♦', '♣']
    '''
    unknown_sides = get_unknown_sides(state)
    result = Goren_bidding_infer(state, unknown_sides,nn=n)
    # st()
    return result


@dataclass
class MCTSAgent(BaseAgent):
    n: int=10
    generate_hands: Callable[[State, int], List[List[Hand]]]=bidding_inference_hands # uniform_hands, bidding_inference_hands, suit_hands, suit_hands_backup
    reduction: Literal['mean', 'max', 'min']='mean'
    max_threads: int=0

    def get_action(self, state: State) -> Card:
        set_max_threads(self.max_threads)
        actions = state.get_legal_actions()
        possible_hands = self.generate_hands(state, self.n)
        scores = dds_score(
            trump=state.trump,
            first=state.first,
            current_cards=state.current_cards,
            pbn_hands=[hands2pbn(hands, state.declarer) for hands in possible_hands],
            num_legal_actions=len(actions),
            reduction=self.reduction,
            solution=2
        )
        suit, rank = max(scores, key=scores.get)
        result = Card(suit=suit, rank=rank)
        if result not in actions:
            print(result, actions)
            raise ValueError("DDS action not legal")
        return result

AGENT_DICT = {
    "Random": RandomAgent,
    "DDS": DDSAgent,
    "MCTS": MCTSAgent
}
