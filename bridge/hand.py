from dataclasses import dataclass, field
from typing import Literal, List

@dataclass
class Card:
    suit: Literal[0, 1, 2, 3]
    rank: Literal[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

@dataclass
class Hand:
    remain_cards: List[Card]
    cards_played: List[Card] = field(default_factory=list)

    def play_card(self, card: Card) -> None:
        self.remain_cards.remove(card)
        self.cards_played.append(card)
