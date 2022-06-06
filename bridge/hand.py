from dataclasses import dataclass, field
from typing import Literal, List

from bridge.constants import Suit, Side, Rank

@dataclass
class Card:
    '''
    suit: ['♠', '♥', '♦', '♣',]
    rank: ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    '''
    suit: Literal[0, 1, 2, 3]
    rank: Literal[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

    def __hash__(self):
        return hash((self.suit, self.rank))

@dataclass
class Hand:
    remain_cards: List[Card]
    cards_played: List[Card] = field(default_factory=list)

    def play_card(self, card: Card) -> None:
        self.remain_cards.remove(card)
        self.cards_played.append(card)

    def to_pbn(self) -> str:
        ranks_list = [[] for i in range(4)]
        for card in self.remain_cards:
            ranks_list[card.suit].append(card.rank)
        return '.'.join(
            ''.join(Rank.idx2str(rank) for rank in sorted(ranks, reverse=True))
            for ranks in ranks_list
        )

def hands2pbn(hands: List[Hand], declarer: Literal[0, 1, 2, 3]=0) -> str:
    result = f'{Side.idx2str(declarer, simple=True)}:'
    result += ' '.join(
        hands[(declarer + i) % 4].to_pbn()
        for i in range(4)
    )
    return result

def pbn2hands(pbn: str) -> List[Hand]:
    first = pbn[0]
    first_idx = Side.str2idx(first, simple=True)
    hands = pbn[2:].split(" ")
    hands = hands[((4-first_idx) % 4):] + hands[:((4-first_idx) % 4)]
    result = []
    for hand in hands:
        remain_cards = []
        for suit, ranks in enumerate(hand.split(".")):
            for rank in ranks:
                remain_cards.append(Card(suit=suit, rank=Rank.str2rank(rank)))
        result.append(Hand(remain_cards=remain_cards))
    return result
