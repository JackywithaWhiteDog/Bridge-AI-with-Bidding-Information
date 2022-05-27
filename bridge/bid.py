from dataclasses import dataclass, field, InitVar
from typing import Literal, List

from bridge.constants import Suit, Side
from bridge.hand import Hand, Card
from bridge.state import State, Result
from bridge.agents import BaseAgent
import random
import numpy as np
from ipdb import set_trace as st
import copy

@dataclass
class Goren_bidding:
    """bidding system (Goren), used before playing phase"""
    hands: InitVar[List[Hand]]
    declarer_starter: InitVar[Literal[0, 1, 2, 3]] = Side.NORTH
    num_cards_in_hand: InitVar[int] = 13

    state: State = field(init=False)

    def __post_init__(
        self,
        hands,
        trump,
        declarer,
        declarer_goal,
        end_when_goal_achieved,
        num_cards_in_hand
    ):

        current_player = Side.get_left(declarer)
        self.state = State(
            hands=hands,
            trump=trump,
            first=current_player,
            current_player=current_player,
            declarer=declarer,
            declarer_goal=declarer_goal,
            end_when_goal_achieved=end_when_goal_achieved,
            num_cards_in_hand=num_cards_in_hand
        )

    def run(self) -> Result:
        while not self.state.reach_end():
            if self.state.current_player in self.state.declarer_side:
                card = self.declarer_agent.get_action(self.state)
            else:
                card = self.defender_agent.get_action(self.state)
            self.state.apply_action(card)
        return self.state.get_result()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

full_cards: List[Card] = [
    Card(suit=suit, rank=rank)
    for suit in Suit.to_list(nt=False)
    for rank in range(2, 14+1, 1)
]


if __name__ == '__main__':
    set_seed(777)
    # assert len(full_cards) == 52, 'There are not 52 cards'
    hands = [
        Hand(remain_cards=cards.tolist())
        for cards in np.split(np.random.permutation(full_cards)[:4*13], 4)
    ]
    # trump = random.choice(Suit.to_list(nt=True))
    declarer_starter = random.choice(Side.to_list())
    game = Goren_bidding(
        hands=copy.deepcopy(hands),
        declarer_starter=declarer_starter,
        num_cards_in_hand=13
    )
    result_1 = game.run()