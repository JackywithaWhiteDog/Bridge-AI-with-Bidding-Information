from dataclasses import dataclass, field
from typing import List, Literal

from bridge.constants import Suit, Side
from bridge.hand import Hand, Card

"""
boardsPBN
- noOfBoards = N
- deals
- target = -1
- solutions = 1
- mode = 0

dealPBN
- trump = 0~4
- first = 0~3
- currentTrickSuit = [0~4] * 3
- currentTrickRank = [0 or 2~14] * 3
- ramainCards = PBN encoding

W:T5.K4.652.A98542 K6.QJT976.QT7.Q6 432.A.AKJ93.JT73 AQJ987.8532.84.K
"""

@dataclass
class Result:
    tricks: List[int]
    winner: str = None

@dataclass
class State:
    hands: List[Hand]
    trump: int
    first: int
    current_player: int
    declarer: Literal[0, 1, 2, 3] = Side.NORTH
    declarer_goal: Literal[7, 8, 9, 10, 11, 12, 13] = 7
    end_when_goal_achieved: bool = False
    num_cards_in_hand: int = 13
    declarer_side: List[int] = field(init=False)
    defender_goal: int = field(init=False)
    tricks: List[int] = field(init=False, default_factory=lambda:[0, 0])
    current_cards: List[Card] = field(init=False, default_factory=list)

    def __post_init__(self):
        if self.num_cards_in_hand > 13 or self.num_cards_in_hand <= 0:
            raise ValueError(f"The number of cards in hand ({self.num_cards_in_hand}) is illegal; it should be 1 ~ 13")
        if self.declarer_goal > self.num_cards_in_hand:
            raise ValueError(f"The declarer's goal ({self.declarer_goal}) is higher than the number of cards in hand ({self.num_cards_in_hand})")

        self.declarer_side = [self.declarer, Side.get_partner(self.declarer)]
        self.defender_goal = self.num_cards_in_hand - self.declarer_goal + 1

    def reach_end(self) -> bool:
        if self.end_when_goal_achieved:
            return (self.tricks[0] >= self.declarer_goal) or (self.tricks[1] >= self.defender_goal)
        else:
            return sum(self.tricks) >= self.num_cards_in_hand

    def apply_action(self, card: Card):
        if not self.is_legal_action(card):
            raise ValueError("Action illegal")

        # Update hand
        self.hands[self.current_player].play_card(card)

        # Update current cards
        self.current_cards.append(card)

        if len(self.current_cards) == 4: # Round end
            # Estimate the trick taker
            taker = self.first
            best_card = self.current_cards[0]
            for i, card in enumerate(self.current_cards[1:], start=1):
                if self.larger(card, best_card):
                    best_card = card
                    taker = (self.first + i) % 4
            
            # Update tricks
            if taker in self.declarer_side:
                self.tricks[0] += 1
            else:
                self.tricks[1] += 1

            # Define next round
            self.first = taker
            self.current_player = taker
            self.current_cards = []
        else:
            self.current_player = Side.get_left(self.current_player)

    def is_legal_action(self, card: Card) -> bool:
        """Return True if the action is legal"""

        # Check the card is in player's hand
        if card not in self.hands[self.current_player].remain_cards:
            return False

        # Check the card is the lead or the suit follows the lead
        if (len(self.current_cards) == 0) or (card.suit == self.current_cards[0].suit):
            return True

        # Check if player does not have the card with the suit following the lead
        lead_suit = self.current_cards[0].suit
        for card_hold in self.hands[self.current_player].remain_cards:
            if card_hold.suit == lead_suit:
                return False
        return True

    def larger(self, card_1: Card, card_2: Card) -> bool:
        """Return True if card 1 is larger than card 2"""
        if len(self.current_cards) == 0:
            raise ValueError("NT requires the suit of the first card as the trump; there is no card played in this round.")
        if self.trump != Suit.NT:
            is_trump_1 = card_1.suit == self.trump
            is_trump_2 = card_2.suit == self.trump
            if is_trump_1 and (not is_trump_2):
                return True
            if (not is_trump_1) and is_trump_2:
                return False
            if is_trump_1 and is_trump_2:
                return card_1.rank > card_2.rank

        # NT or both cards are not trump
        first_suit = self.current_cards[0].suit
        follow_first_1 = card_1.suit == first_suit
        follow_first_2 = card_2.suit == first_suit
        if follow_first_1 and (not follow_first_2):
            return True
        if (not follow_first_1) and follow_first_2:
            return False
        if follow_first_1 and follow_first_2:
            return card_1.rank > card_2.rank
        # Both cards are not trump and does not follow the first suit, no need to compare
        return False

    def get_legal_actions(self) -> List[Card]:
        """Return a list of legal actions"""
        # If the current player is the first one of the round, all cards in hand are legal
        if len(self.current_cards) == 0:
            return self.hands[self.current_player].remain_cards.copy()

        lead_suit = self.current_cards[0].suit
        cards_in_same_suit = [
            card_hold
            for card_hold in self.hands[self.current_player].remain_cards
            if card_hold.suit == lead_suit
        ]

        # If the current player does not have card in same suit, all cards in hand are legal
        if len(cards_in_same_suit) == 0:
            return self.hands[self.current_player].remain_cards.copy()

        # Otherwise, only the cards in same suit are legal
        return cards_in_same_suit

    def get_result(self) -> Result:
        if self.tricks[0] >= self.declarer_goal:
            winner = 'declarer'
        elif self.tricks[1] >= self.defender_goal:
            winner = 'defender'
        else:
            winner = None
        return Result(
            tricks=self.tricks.copy(),
            winner=winner
        )
