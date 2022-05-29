from dataclasses import dataclass, field, InitVar
from locale import currency
from typing import Literal, List

import sys
sys.path.insert(0, "/home/kung/code/Bridge-AI-with-Bidding-Information/")
# print(sys.path)

from bridge.constants import Suit, Side
from bridge.hand import Hand, Card
from bridge.state import State, Result

import random
import numpy as np
from ipdb import set_trace as st
import copy


legel_bid = [f'{i}_{j}' for i in ['spade', 'heart', 'dimond', 'club', 'NT'] for j in range(1,8)]
legel_bid.append('pass')


@dataclass
class Bid:
    '''
    suit: ['♣', '♦', '♥', '♠', 'NT', 'pass']
    rank: range(1,7)
    '''
    suit: Literal[0, 1, 2, 3, 4, 5]
    rank: Literal[1, 2, 3, 4, 5, 6, 7]
    def to_str(self):
        suit_dic = ['♣', '♦', '♥', '♠', 'NT', 'pass']
        if self.suit == 5:
            return 'pass'
        else:
            return f'{suit_dic[self.suit]}_{self.rank}'


@dataclass
class Goren_bidding:
    """
    bidding system (Goren), used before playing phase
    Side: [0,1,2,3] <- [N,E,S,W] 
    bidding order: 0->3->2->1->0...
    """
    hands: List[Hand]
    declarer_starter: Literal[0, 1, 2, 3] = Side.NORTH
    num_cards_in_hand: int = 13
    # hcp: List[int] = field(init=False, default_factory=lambda:[0, 0, 0, 0])
    # hcp: List[int] = field(init=False, default_factory=list)
    def __post_init__(self):
        self.hcp = [[] for i in range(4)]
        self.count_suit = [[] for i in range(4)] # 0,1,2,3 <- N E S W
        for i, hand in enumerate(self.hands):
            self.hcp[i], self.count_suit[i] = self.calculate_HCP(hand)

    def run(self):
        '''
        return: a bidding table: (1,28) np.array, 
        the first element is the the first player to bid
        the second element is his right playrer's bidding
        '''
        bidding_table = np.zeros(28, dtype=Bid)
        pass_bid = 0
        turns = 0
        cureent_bid_player = self.declarer_starter
        last_non_pass_bid = None
        # st()
        while pass_bid < 3:
            # hand = self.hands[cureent_bid_player]
            # cureent_bid_player_partner = Side.get_partner(cureent_bid_player)
            if bidding_table[turns-2] == 0: # first bid
                bid_decision = self.make_first_bid(hcp_list=self.hcp[cureent_bid_player], 
                                                   suit_count=self.count_suit[cureent_bid_player],
                                                   last_non_pass_bid=last_non_pass_bid)
            else: # response bid
                partner_bid = bidding_table[turns-2]
                bid_decision = self.make_response_bid(hcp_list=self.hcp[cureent_bid_player], 
                                                      suit_count=self.count_suit[cureent_bid_player],
                                                      partner_bid=partner_bid,
                                                      last_non_pass_bid=last_non_pass_bid)
            bidding_table[turns] = bid_decision
            if bid_decision.suit == 5: # pass
                pass_bid += 1
            else:
                pass_bid = 0
                last_non_pass_bid = bid_decision
            turns += 1
            cureent_bid_player = Side.get_right(cureent_bid_player)
        return bidding_table
    
    def make_first_bid(self, hcp_list, suit_count, last_non_pass_bid):
        '''first bid'''
        hcp = np.sum(hcp_list)
        if hcp < 5:
            return Bid(5, 0) # pass
        if 5<=hcp<11 and suit_count[2]==3 and suit_count[3]==3 and self.is_legal_bid(last_non_pass_bid,  Bid(0, 1)):
            return Bid(0, 1) #'club_1'
        if 5<=hcp<11 and suit_count[2]==4 and suit_count[3]==4 and self.is_legal_bid(last_non_pass_bid,  Bid(1, 1)):
            return Bid(1, 1) # 'dimond_1'
        if hcp>=13 and (suit_count[1]==5 or suit_count[1]==6) and suit_count[1]>suit_count[0] and self.is_legal_bid(last_non_pass_bid,  Bid(2, 1)):
            return Bid(2, 1) # 'heart_1'
        if hcp>=13 and (suit_count[0]==5 or suit_count[0]==6) and self.is_legal_bid(last_non_pass_bid,  Bid(3, 1)):
            return Bid(3, 1) #'spade_1'
        if 17>=hcp>=15 and suit_count[0]>1 and suit_count[1]>1 and suit_count[2]>1 and suit_count[3]>1 and self.is_legal_bid(last_non_pass_bid,  Bid(4, 1)):
            return Bid(4, 1) # 'NT_1'
        if hcp >= 22 and self.is_legal_bid(last_non_pass_bid,  Bid(0, 2)):
            return Bid(0, 2) #'club_2'
        if 5<=hcp<=11:
            if suit_count[0]>6 and self.is_legal_bid(last_non_pass_bid,  Bid(3, 2)):
                return Bid(3, 2) # 'spade_2'
            if suit_count[1]>6 and self.is_legal_bid(last_non_pass_bid,  Bid(2, 2)):
                return Bid(2, 2) # 'heart_2'
            if suit_count[2]>6 and self.is_legal_bid(last_non_pass_bid,  Bid(1, 2)):
                return Bid(1, 2) #'dimond_2'
        if hcp>=20 and hcp<=21 and suit_count[0]>1 and suit_count[1]>1 and suit_count[2]>1 and suit_count[3]>1 and self.is_legal_bid(last_non_pass_bid,  Bid(4, 2)):
            return Bid(4, 2)  #'NT_2'
        return Bid(5, 0) # pass

    def make_response_bid(self, hcp_list, suit_count, partner_bid, last_non_pass_bid):
        '''response to the partner'''
        hcp = np.sum(hcp_list)
        if partner_bid.suit == 0 and partner_bid.rank == 1: # 1 C
            if 13<=hcp<=15 and self.is_legal_bid(last_non_pass_bid,  Bid(4, 2)):
                return Bid(4, 2) # 2 NT
            if 16<=hcp<=17 and self.is_legal_bid(last_non_pass_bid,  Bid(4, 3)):
                return Bid(4, 3) #  3NT
        if partner_bid.suit == 1 and partner_bid.rank == 1: # 1 D
            if 13<=hcp<=15 and self.is_legal_bid(last_non_pass_bid,  Bid(4, 2)):
                return Bid(4, 2) # 2 NT
            if 16<=hcp<=17 and self.is_legal_bid(last_non_pass_bid,  Bid(4, 3)):
                return Bid(4, 3) #  3NT
        if (partner_bid.suit == 2 or partner_bid.suit==3) and partner_bid.rank == 1: # 1 H or 1 S
            if hcp>=6 and suit_count[0]>=4 and suit_count[1]==0 and self.is_legal_bid(last_non_pass_bid,  Bid(3, 1)):
                return Bid(3, 1) # 1 S
            if 9>=hcp>=6 and suit_count[0]!=4 and suit_count[1]!=3 and self.is_legal_bid(last_non_pass_bid,  Bid(4, 1)):
                return Bid(4, 1) # 1 NT
            if hcp>=10 and suit_count[2]>=4 and self.is_legal_bid(last_non_pass_bid,  Bid(1, 1)):
                return Bid(1, 1)
            if hcp>=10 and suit_count[3]>=4 and self.is_legal_bid(last_non_pass_bid,  Bid(0, 1)):
                return Bid(0, 1)
            if hcp>=13 and self.is_legal_bid(last_non_pass_bid,  Bid(4, 2)):
                return Bid(4, 2)  #'NT_2'
            if 10<=hcp<=12 and suit_count[1]>=3 and self.is_legal_bid(last_non_pass_bid,  Bid(2, 3)):
                return Bid(2, 3) # 'heart_3'
            if 15<=hcp<=17 and suit_count[1]>=3 and suit_count[0]>1 and suit_count[1]>1 and suit_count[2]>1 and suit_count[3]>1 and self.is_legal_bid(last_non_pass_bid,  Bid(4, 3)):
                return Bid(4, 3) # 'NT_3'
            if hcp<10 and suit_count[1]>=5 and self.is_legal_bid(last_non_pass_bid,  Bid(2, 4)):
                return Bid(2, 4) # 'heart_4'
        if partner_bid.suit == 4 and partner_bid.rank == 1: # 1NT
            if hcp>=8 and self.is_legal_bid(last_non_pass_bid,  Bid(0, 2)):
                return Bid(0, 2) #'club_2'
        if partner_bid.suit == 0 and partner_bid.rank == 2: # 2 C
            if hcp>=8 and suit_count[0]>=5 and self.is_legal_bid(last_non_pass_bid,  Bid(3, 2)) : # spade>=5
                return Bid(3, 2) # 'spade_2' 
            if hcp>=8 and suit_count[1]>=5 and self.is_legal_bid(last_non_pass_bid,  Bid(2, 2)): # heart>=5
                return Bid(2, 2) # 'heart_2' 
            if hcp>=8 and suit_count[2]>=5 and self.is_legal_bid(last_non_pass_bid,  Bid(1, 2)): # dimond>=5
                return Bid(1, 2) # 'dimond_2' 
            if hcp>=8 and suit_count[3]>=5 and self.is_legal_bid(last_non_pass_bid,  Bid(0, 2)): # club>=5
                return Bid(0, 2) # 'club_2' 
            if hcp==8 and suit_count[1]>=3 and suit_count[0]>1 and suit_count[1]>1 and suit_count[2]>1 and suit_count[3]>1 and self.is_legal_bid(last_non_pass_bid,  Bid(4, 2)):
                return Bid(4, 2) # 'NT_2' 
        if (partner_bid.suit == 1 or partner_bid.suit == 2 or partner_bid.suit == 3) and partner_bid.rank == 2 and self.is_legal_bid(last_non_pass_bid,  Bid(4, 2)): # Response to 2D,2H,2S contract
            return Bid(4, 2) # 'NT_2'
        if partner_bid.suit == 4 and partner_bid.rank == 2 and self.is_legal_bid(last_non_pass_bid,  Bid(0, 3)): # Response to 2NT contract
            return Bid(0, 3) # 'club_3'
        if partner_bid.suit == 4 and partner_bid.rank == 3 and self.is_legal_bid(last_non_pass_bid,  Bid(0, 4)): # Response to 3NT contract
            return Bid(0, 4) # 'club_4'
        return Bid(5, 0) # pass

    def calculate_HCP(self, hand):
        '''
        input: 13 card
        return [SPADES: hcp,  HEARTS: hcp, DIAMONDS: hcp, CLUBS: hcp] , [len(x) for x in suits]
        Rule: 
            High-card points (HCP): A=4, K=3, Q=2, J=1
            Long-suit points: Add 1 point for a good 5-card suit, 2 for a 6-card suit, 3 for a 7-card suit.
            Short-suit points: If you have a trump fit with partner, add 1 point for a doubleton in a side suit, 2 for a singleton, 3 for a void.
        '''
        hand_cards = hand.remain_cards
        hcp_suit = [0, 0, 0, 0]
        count_suit = [0, 0, 0, 0]
        for hand_card in hand_cards:
            hcp_suit[hand_card.suit] += max(0, hand_card.rank-10)
            count_suit[hand_card.suit] += 1
        for ii, counts  in enumerate(count_suit):
            if counts >= 5:
                hcp_suit[ii] += counts-4
        return hcp_suit, count_suit
    
    def is_legal_bid(self, last_bid, next_bid):
        if last_bid is None: # no last bid
            return True
        if next_bid.suit == 5: # next bid is pass
            return True
        if next_bid.rank == last_bid.rank:
            if next_bid.suit > last_bid.suit:
                return True
            else:
                return False
        elif next_bid.rank > last_bid.rank:
            return True
        else:
            return False



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
    result = game.run()
    st()