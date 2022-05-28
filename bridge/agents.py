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
    if state.current_player in state.declarer_side:
        # Does not know the cards in defenders' hands
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
    unknown_cards = state.hands[unknown_sides[0]].remain_cards + state.hands[unknown_sides[1]].remain_cards
    first_num_cards = len(state.hands[unknown_sides[0]].remain_cards)
    result = []
    for i in range(n):
        shuffled_cards = np.random.permutation(unknown_cards)
        hands = copy.deepcopy(state.hands)
        hands[unknown_sides[0]] = Hand(remain_cards=shuffled_cards[:first_num_cards].tolist())
        hands[unknown_sides[1]] = Hand(remain_cards=shuffled_cards[first_num_cards:].tolist())
        result.append(hands)
    return result

def suit_hands(state: State, n: int) -> List[List[Hand]]:
    unknown_sides = get_unknown_sides(state)
    unknown_cards_list = [[] for i in range(4)]
    first_num_cards = [0] * 4
    for i, side in enumerate(unknown_sides):
        for card in state.hands[side].remain_cards:
            unknown_cards_list[card.suit].append(card)
            if i == 0:
                first_num_cards[card.suit] += 1
    result = []
    for i in range(n):
        assigned_cards = [[], []]
        for num, cards in zip(first_num_cards, unknown_cards_list):
            shuffled_cards = np.random.permutation(cards)
            assigned_cards[0] += shuffled_cards[:num].tolist()
            assigned_cards[1] += shuffled_cards[num:].tolist()
        hands = copy.deepcopy(state.hands)
        for side, cards in zip(unknown_sides, assigned_cards):
            hands[side] = Hand(remain_cards=cards)
        result.append(hands)
    return result

@dataclass
class MCTSAgent(BaseAgent):
    n: int=10
    generate_hands: Callable[[State, int], List[List[Hand]]]=uniform_hands
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
