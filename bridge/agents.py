import abc
from dataclasses import dataclass, field
import random
from typing import List, Callable, Literal, Tuple
import copy
from functools import partial

import numpy as np
import torch

from bridge.state import State, Suit, Side
from bridge.hand import Card, Hand, hands2pbn
from bridge.dds import dds_score
from bridge.dds.utils import set_max_threads
from bridge.rnn.models import HandsClassifier

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
        assigned_cards = [[] for i in len(unknown_sides)]
        for num_cards, cards in zip(num_cards_list, unknown_cards_list):
            shuffled_cards = np.random.permutation(cards)
            offset = 0
            for i, num in enumerate(num_cards):
                assigned_cards[i] += shuffled_cards[offset:offset+num].tolist
                offset += num
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

@dataclass
class RNN_MCTSAgent(MCTSAgent):
    # Set default value to avoid error => TypeError: non-default argument 'model_path' follows default argument
    model_path: str=None
    generation_mode: Literal['greedy', 'random']='greedy'
    use_cuda: bool = True
    t: int = 1.0 # Softmax temperature
    model: HandsClassifier = field(init=False)

    def __post_init__(self):
        assert self.model_path is not None
        self.model = HandsClassifier.load_from_checkpoint(self.model_path)
        if self.use_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
        else:
            self.use_cuda = False
        self.generate_hands = self.rnn_generate

    def prepare_hands(self, state: State) -> torch.Tensor:
        unknown_sides = get_unknown_sides(state)
        sides = Side.to_list()
        sides = sides[state.declarer_starter:] + sides[:state.declarer_starter]
        hands = state.hands[state.declarer_starter:] + state.hands[:state.declarer_starter]
        one_hot_hands = torch.zeros(52 * 4)
        card_indices = []
        offset = 0
        known_cards = torch.zeros(52).bool()
        for side, hand in zip(sides, hands):
            if side not in unknown_sides:
                known_hand = hand.remain_cards + hand.cards_played
                one_hot_hands[offset:offset+52] = -1
            else:
                known_hand = hand.cards_played
            for card in known_hand:
                card_indices.append(int(offset + (card.suit * 13) + (card.rank-2)))
                known_cards[int((card.suit * 13) + (card.rank-2))] = True
            offset += 52
        for side in sides:
            if side in unknown_sides:
                one_hot_hands[side*52:(side+1)*52][known_cards] = -1
        one_hot_hands[card_indices] = 1
        return one_hot_hands.unsqueeze(0)

    def prepare_biddings(self, state: State) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        bidding suit: ['♣', '♦', '♥', '♠', 'NT', 'pass']
        encoded suit: ['♠', '♥', '♦', '♣', 'NT', 'pass']
        bidding rank: range(1,7)
        '''
        biddings = state.bidding_info
        one_hot_biddings = torch.zeros(38, 36)
        cnt = 0
        for i, bid in enumerate(state.bidding_info):
            # 0 for action stop
            if isinstance(bid, int):
                break

            # Reverse suit
            if bid.suit <= 3:
                suit = 3 - bid.suit
            else:
                suit = bid.suit

            if suit == 5: # pass
                one_hot_biddings[i, int(suit * 7)] = 1
            else:
                one_hot_biddings[i, int(suit * 7 + (bid.rank - 1))] = 1

            cnt += 1
        return one_hot_biddings.unsqueeze(0), torch.tensor([cnt])

    def validate_guess(self, hands: List[Hand], state: State) -> bool:
        unknown_sides = get_unknown_sides(state)
        sides = Side.to_list()
        for side, hand, gt_hand in zip(sides, hands, state.hands):
            if set(hand.cards_played) != set(gt_hand.cards_played):
                return False
            if side not in unknown_sides:
                if set(hand.remain_cards) != set(gt_hand.remain_cards):
                    return False
        return True

    def decode_hands(self, guesses: torch.Tensor, state: State) -> List[List[Hand]]:
        north_idx = (4 - state.declarer_starter) % 4
        guesses = guesses.split(52, dim=-1)
        guesses = guesses[north_idx:] + guesses[:north_idx]
        guesses = torch.stack(guesses, dim=1)
        result = []
        for guess in guesses:
            hands = []
            for side, encoded_hand in enumerate(guess):
                indices = (encoded_hand == 1).nonzero().T.tolist()[0]
                remain_cards = []
                cards_played = []
                for idx in indices:
                    suit = idx // 13
                    rank = idx % 13 + 2
                    card = Card(suit=suit, rank=rank)
                    if card in state.hands[side].cards_played:
                        cards_played.append(card)
                    else:
                        remain_cards.append(card)
                hands.append(Hand(remain_cards=remain_cards, cards_played=cards_played))
            # assert self.validate_guess(hands, state)
            result.append(hands)
        return result

    def rnn_generate(self, state: State, n: int) -> List[List[Hand]]:
        hand = self.prepare_hands(state)
        bidding, length = self.prepare_biddings(state)
        if self.use_cuda:
            hand = hand.cuda()
            bidding = bidding.cuda()
        with torch.no_grad():
            output = self.model(hand, bidding, length, return_raw_outputs=True)
        if self.generation_mode == 'greedy':
            guesses = self.model.greedy_generate(output.cpu(), hints=hand.cpu())[0].unsqueeze(0)
        elif self.generation_mode == 'random':
            guesses = self.model.random_generate(output.cpu(), hints=hand.cpu(), n=self.n, t=self.t)[0]
        else:
            raise NotImplementedError()
        possible_hands = self.decode_hands(guesses, state)
        return possible_hands

GENERATE_HANDS = {
    'uniform_hands': uniform_hands,
    'suit_hands': suit_hands
}

def get_agent(
    name: str,
    n: int,
    reduction: str,
    generate_hands: str,
    max_threads: int,
    model_path: str,
    generation_mode: str,
    t: float
) -> BaseAgent:
    if name == "Random":
        return RandomAgent()
    elif name == "DDS":
        return DDSAgent(max_threads=max_threads)
    elif name == "MCTS":
        return MCTSAgent(
            n=n,
            generate_hands=GENERATE_HANDS[generate_hands],
            reduction=reduction,
            max_threads=max_threads
        )
    elif name == "RNN":
        return RNN_MCTSAgent(
            n=n,
            generate_hands=GENERATE_HANDS[generate_hands],
            reduction=reduction,
            max_threads=max_threads,
            model_path=model_path,
            generation_mode=generation_mode,
            t=t
        )
    else:
        raise ValueError(f"No such agent {name}: Required to be one of Random, DDS, MCTS or RNN")
