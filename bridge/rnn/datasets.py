import copy
from typing import List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

SIDE2IDX = {
    "N": 0,
    "E": 1,
    "S": 2,
    "W": 3
}

HIGH_RANK2IDX = {
    "T": 10,
    "J": 11,
    "Q": 12,
    "K": 13,
    "A": 14
}

SUIT2IDX = {
    '♠': 0,
    '♥': 1,
    '♦': 2,
    '♣': 3,
    'NT': 4,
    'pass': 5
}

def get_one_hot_hands(hands: List[str]) -> Tuple[Tensor, int]:
    result = torch.zeros(52 * 4)
    indices = []
    offset = 0
    for side, hand in enumerate(hands):
        for suit, suit_hand in enumerate(hand.split(".")):
            for rank in suit_hand:
                if rank in HIGH_RANK2IDX:
                    rank = HIGH_RANK2IDX[rank]
                else:
                    rank = int(rank)
                indices.append(offset + (rank - 2))
            offset += 13
    result[indices] = 1
    return result

def get_one_hot_bidding(bidding: str) -> Tensor:
    result = torch.zeros((38, 36))
    bidding_list = bidding.split(".")
    for i, action in enumerate(bidding_list):
        action_pair = action.strip().split("_")
        if len(action_pair) == 1: # pass
            result[i, SUIT2IDX[action_pair[0]] * 7] = 1
        else:
            result[i, SUIT2IDX[action_pair[0]] * 7 + (int(action_pair[1]) - 1)] = 1
    return result, len(bidding_list)

def random_mask(hand: Tensor) -> Tensor:
    result = hand.clone()
    result[result == 0] = -1
    num_mask = torch.randint(low=2, high=3+1, size=(1,))[0]
    shuffle_sides = torch.randperm(n=4)
    known_cards = torch.zeros(52).bool()
    for known_side in shuffle_sides[num_mask:]:
        known_cards[result[known_side*52:(known_side+1)*52] > 0] = True
    for unknown_side in shuffle_sides[:num_mask]:
        result[unknown_side*52:(unknown_side+1)*52] = 0
        result[unknown_side*52:(unknown_side+1)*52][known_cards] = -1
    return result

class BidDataset(Dataset):
    def __init__(self, filename):
        self.first_declarers = []
        self.hands = []
        self.biddings = []
        self.lengths = []
        with open(filename, "r") as f:
            for line in f:
                hands, bid = line.split(" || ")
                first_declarer, bidding = bid.split(":")
                hands = hands.split("$")
                # Rotate the hands to make the first declarer at index 0
                hands = hands[SIDE2IDX[first_declarer]:] + hands[:SIDE2IDX[first_declarer]]
                hands = get_one_hot_hands(hands)

                bidding, length = get_one_hot_bidding(bidding)

                self.first_declarers.append(SIDE2IDX[first_declarer])
                self.hands.append(hands)
                self.biddings.append(bidding)
                self.lengths.append(length)
        self.hands = torch.stack(self.hands)
        self.biddings = torch.stack(self.biddings)
        self.lengths = torch.tensor(self.lengths)

    def __len__(self):
        return len(self.first_declarers)

    def __getitem__(self, idx):
        masked_hand = random_mask(self.hands[idx])
        return masked_hand, self.biddings[idx], self.lengths[idx], self.hands[idx]
