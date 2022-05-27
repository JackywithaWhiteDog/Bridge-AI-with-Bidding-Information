from dataclasses import dataclass, field, InitVar
from typing import List, Literal
import random

import numpy as np

from bridge.constants import Suit, Side
from bridge.hand import Card, Hand
from bridge.game import Game
from bridge.agents import BaseAgent, AGENT_DICT

DECK: List[Card] = [
    Card(suit=suit, rank=rank)
    for suit in Suit.to_list(nt=False)
    for rank in range(2, 14+1, 1)
]

@dataclass
class Match:
    """Class for playing match"""
    declarer_agent_name: InitVar[Literal["Random"]]
    defender_agent_name: InitVar[Literal["Random"]]
    num_games: int = 100
    num_cards_in_hand: int = 13
    max_threads: InitVar[int] = 0
    declarer_agent: BaseAgent = field(init=False)
    defender_agent: BaseAgent = field(init=False)

    def __post_init__(self, declarer_agent_name, defender_agent_name, max_threads):
        if self.num_cards_in_hand > 13 or self.num_cards_in_hand <= 0:
            raise ValueError(f"The number of cards in hand ({self.num_cards_in_hand}) is illegal; it should be 1 ~ 13")

        if declarer_agent_name in ["DDS",]:
            self.declarer_agent = AGENT_DICT[declarer_agent_name](max_threads=max_threads)
        else:
            self.declarer_agent = AGENT_DICT[declarer_agent_name]()

        if defender_agent_name in ["DDS",]:
            self.defender_agent = AGENT_DICT[defender_agent_name](max_threads=max_threads)
        else:
            self.defender_agent = AGENT_DICT[defender_agent_name]()

    def run(self):
        win_counts = {'declarer': 0, 'defender': 0}
        for i in range(self.num_games):
            hands = [
                Hand(remain_cards=cards.tolist())
                for cards in np.split(np.random.permutation(DECK)[:4*self.num_cards_in_hand], 4)
            ]
            trump = random.choice(Suit.to_list(nt=True))
            declarer = random.choice(Side.to_list())
            print(f"[Game {i+1}] Deal: 1{Suit.idx2str(trump, simple=True)} | Declarer: {Side.idx2str(declarer, simple=False)}")
            game = Game(
                declarer_agent=self.declarer_agent,
                defender_agent=self.defender_agent,
                hands=hands,
                trump=trump,
                declarer=declarer,
                declarer_goal=7,
                end_when_goal_achieved=False,
                num_cards_in_hand=self.num_cards_in_hand
            )
            result = game.run()
            win_counts[result.winner] += 1
            print(f"[Game {i+1}] Winner: {result.winner} | Declarer tricks: {result.tricks[0]} | Defender tricks: {result.tricks[1]}")
        print(f"=> Declarer vs. Defender: {win_counts['declarer']}:{win_counts['defender']}")
