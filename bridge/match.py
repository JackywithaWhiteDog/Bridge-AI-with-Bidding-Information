from dataclasses import dataclass, field, InitVar
from typing import List, Literal
import random
import copy

import numpy as np
from tqdm import tqdm

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
    agent_a_name: InitVar[Literal["Random"]]
    agent_b_name: InitVar[Literal["Random"]]
    num_games: int = 100
    num_cards_in_hand: int = 13
    max_threads: InitVar[int] = 0
    agent_a: BaseAgent = field(init=False)
    agent_b: BaseAgent = field(init=False)

    def __post_init__(self, agent_a_name, agent_b_name, max_threads):
        if self.num_cards_in_hand > 13 or self.num_cards_in_hand <= 0:
            raise ValueError(f"The number of cards in hand ({self.num_cards_in_hand}) is illegal; it should be 1 ~ 13")

        if agent_a_name in ["DDS",]:
            self.agent_a = AGENT_DICT[agent_a_name](max_threads=max_threads)
        else:
            self.agent_a = AGENT_DICT[agent_a_name]()

        if agent_b_name in ["DDS",]:
            self.agent_b = AGENT_DICT[agent_b_name](max_threads=max_threads)
        else:
            self.agent_b = AGENT_DICT[agent_b_name]()

    def run(self):
        print(f"Agent A: {self.agent_a} | Agent B: {self.agent_b}")
        win_counts = {'win': 0, 'draw': 0, 'loss': 0}
        progress_bar = tqdm(range(self.num_games))
        for i in progress_bar:
            hands = [
                Hand(remain_cards=cards.tolist())
                for cards in np.split(np.random.permutation(DECK)[:4*self.num_cards_in_hand], 4)
            ]
            trump = random.choice(Suit.to_list(nt=True))
            declarer = random.choice(Side.to_list())
            # progress_bar.set_description(f"[Game {i+1}/{self.num_games}] Deal: 1{Suit.idx2str(trump, simple=True)} | Declarer: {Side.idx2str(declarer, simple=False)}")
            game = Game(
                declarer_agent=self.agent_a,
                defender_agent=self.agent_b,
                hands=copy.deepcopy(hands),
                trump=trump,
                declarer=declarer,
                declarer_goal=7,
                end_when_goal_achieved=False,
                num_cards_in_hand=self.num_cards_in_hand
            )
            result_1 = game.run()
            game = Game(
                declarer_agent=self.agent_b,
                defender_agent=self.agent_a,
                hands=copy.deepcopy(hands),
                trump=trump,
                declarer=declarer,
                declarer_goal=7,
                end_when_goal_achieved=False,
                num_cards_in_hand=self.num_cards_in_hand
            )
            result_2 = game.run()
            if result_1.winner == result_2.winner:
                win_counts['draw'] += 1
                result_msg = 'Draw'
            elif result_1.winner == 'declarer':
                win_counts['win'] += 1
                result_msg = 'Agent A'
            else:
                win_counts['loss'] += 1
                result_msg = 'Agent B'
            progress_bar.set_description(f"[Game {i+1}/{self.num_games}] Total Result (Agent A): {win_counts['win']}-{win_counts['draw']}-{win_counts['loss']} | Result: {result_msg.ljust(6)}")
        print(f"Agent A => Win-Draw-Loss: {win_counts['win']}-{win_counts['draw']}-{win_counts['loss']}")
