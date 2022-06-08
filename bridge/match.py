from dataclasses import dataclass, field, InitVar
from typing import List, Dict, Any
import random
import copy
from pathlib import Path
import json

import numpy as np
from tqdm import tqdm

from bridge.constants import Suit, Side
from bridge.hand import Card, Hand, pbn2hands
from bridge.game import Game
from bridge.agents import BaseAgent, get_agent
from bridge.bid import Goren_bidding, Bid, str2biddings

from ipdb import set_trace as st

DECK: List[Card] = [
    Card(suit=suit, rank=rank)
    for suit in Suit.to_list(nt=False)
    for rank in range(2, 14+1, 1)
]

@dataclass
class Match:
    """Class for playing match"""
    agent_a_kwargs: InitVar[Dict[str, Any]]
    agent_b_kwargs: InitVar[Dict[str, Any]]
    num_games: int = 100
    num_cards_in_hand: int = 13
    match_file: str=None
    output_file: str=None
    agent_a: BaseAgent = field(init=False)
    agent_b: BaseAgent = field(init=False)
    # matches: List[Dict[str, Any]] = field(init=False)
    matches: List[Dict[str, Any]] = field(init=False, default=None)

    def __post_init__(self, agent_a_kwargs, agent_b_kwargs):
        if self.num_cards_in_hand > 13 or self.num_cards_in_hand <= 0:
            raise ValueError(f"The number of cards in hand ({self.num_cards_in_hand}) is illegal; it should be 1 ~ 13")

        self.agent_a = get_agent(**agent_a_kwargs)
        self.agent_b = get_agent(**agent_b_kwargs)

        if self.match_file is not None:
            with open(self.match_file, "r") as f:
                self.matches = json.load(f)
                print(f"Load match data from {self.match_file}.")
                if self.num_games > len(self.matches):
                    self.num_games = len(self.matches)
                    print(f"The number of games (={self.num_games}) is larger than data in match file ({len(self.matches)}). Set the number of game to {len(self.matches)}")
        
    def trump_and_declarer(self, bidding_info, declarer_starter):
        '''
        bidding suit: ['♣', '♦', '♥', '♠', 'NT', 'pass']
        playing suit: ['♠', '♥', '♦', '♣', "NT"]
        return:
            the suit of trump
            the goal that declarer needs to achieve
            declarer's side
        '''
        mapping_dic = {0:3, 1:2, 2:1, 3:0, 4:4}
        result_info = [i for i in bidding_info if i != 0]
        last_bid = result_info[-4]
        declarer = (declarer_starter + len(result_info)) % 4
        return mapping_dic[last_bid.suit], last_bid.rank+6, declarer


    def run(self):
        print(f"Agent A: {self.agent_a} | Agent B: {self.agent_b}")
        win_counts = {'win': 0, 'draw': 0, 'loss': 0}
        a_scores = []
        progress_bar = tqdm(range(self.num_games))
        cnt = 1
        logs = []
        while cnt <= self.num_games:
            if self.matches is None:
                hands = [
                    Hand(remain_cards=cards.tolist())
                    for cards in np.split(np.random.permutation(DECK)[:4*self.num_cards_in_hand], 4)
                ]
                # trump = random.choice(Suit.to_list(nt=True))
                # declarer = random.choice(Side.to_list())
                # st()
                declarer_starter = random.choice(Side.to_list())
                bidding_game = Goren_bidding(            
                    hands=copy.deepcopy(hands),
                    declarer_starter=declarer_starter,
                    num_cards_in_hand=13)
                bidding_result = bidding_game.run()
                if [i.to_str() for i in bidding_result[:4] if i != 0].count('pass') == 4:
                    continue
                trump, declarer_goal, declarer = self.trump_and_declarer(bidding_result, declarer_starter)
                # progress_bar.set_description(f"[Game {i+1}/{self.num_games}] Deal: 1{Suit.idx2str(trump, simple=True)} | Declarer: {Side.idx2str(declarer, simple=False)}")
            else:
                data = self.matches[cnt-1]
                hands = pbn2hands(data['deal'])
                declarer_starter = Side.str2idx(data['declarer_starter'], simple=True)
                bidding_result = str2biddings(data['bidding'])
                declarer = Side.str2idx(data['declarer'], simple=True)
                declarer_goal = int(data["contract"][0]) + 6
                trump = Suit.char2idx(data["contract"][1:].replace("X", ""))
            game = Game(
                declarer_agent=self.agent_a,
                defender_agent=self.agent_b,
                hands=copy.deepcopy(hands),
                bidding_info=bidding_result,
                trump=trump,
                declarer=declarer,
                declarer_goal=declarer_goal,
                end_when_goal_achieved=False,
                num_cards_in_hand=self.num_cards_in_hand,
                declarer_starter=declarer_starter
            )
            result_1 = game.run()
            game = Game(
                declarer_agent=self.agent_b,
                defender_agent=self.agent_a,
                hands=copy.deepcopy(hands),
                bidding_info=bidding_result,
                trump=trump,
                declarer=declarer,
                declarer_goal=declarer_goal,
                end_when_goal_achieved=False,
                num_cards_in_hand=self.num_cards_in_hand,
                declarer_starter=declarer_starter
            )
            result_2 = game.run()
            bidding_length = sum(not isinstance(b, int) for b in bidding_result)
            logs.extend([
                {
                    'trump': trump,
                    'is_declarer': True,
                    'declarer_goal': declarer_goal,
                    'tricks': result_1.tricks[0],
                    'win': result_1.winner == 'declarer',
                    'bidding_length': bidding_length
                },
                {
                    'trump': trump,
                    'is_declarer': False,
                    'declarer_goal': declarer_goal,
                    'tricks': result_2.tricks[1],
                    'win': result_2.winner == 'defender',
                    'bidding_length': bidding_length
                }
            ])
            a_scores.append(result_1.tricks[0] - result_2.tricks[0])
            if result_1.winner == result_2.winner:
                win_counts['draw'] += 1
            elif result_1.winner == 'declarer':
                win_counts['win'] += 1
            else:
                win_counts['loss'] += 1
            progress_bar.update(1)
            progress_bar.set_description(f"[Game {cnt}/{self.num_games}] Result: {result_1.tricks[0]}:{result_1.tricks[1]} / {result_2.tricks[0]}:{result_2.tricks[1]} | Total Result (Agent A): {win_counts['win']}-{win_counts['draw']}-{win_counts['loss']} | Score A: {sum(a_scores) / (cnt):.2f}")
            cnt += 1
        a_scores = np.array(a_scores)
        print(f"Agent A => Win-Draw-Loss: {win_counts['win']}-{win_counts['draw']}-{win_counts['loss']} | Score: {a_scores.mean():.2f} (std: {a_scores.std():.2f})")
        if self.output_file is not None:
            output_file = Path(self.output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(logs, f, indent=4)
