import argparse
import random
import copy
import numpy as np
from typing import Literal, List

from bridge.bid import Bid, Goren_bidding
from bridge.constants import Suit, Side
from bridge.hand import Hand, Card

from ipdb import set_trace as st


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Generate biding results with corresponding hands')
    parser.add_argument('--seed', type=int, default=777,)
    parser.add_argument('--data_size', type=int, default=10000,)
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


full_cards: List[Card] = [
    Card(suit=suit, rank=rank)
    for suit in Suit.to_list(nt=False)
    for rank in range(2, 14+1, 1)
]


if __name__ == '__main__':
    '''
    # Bidding oeder: 0 3 2 1  (i.e.,NESW) 
    # playing order: 0 1 2 3  (i.e.,NWSE)
    output and save: str based txt file. The format is:
        N's hand $ E's hand $ S's hand $ W's hand || declarer_starter: bidding sequence
    For example:
    'AKJ.J865.972.J65$T86542.T4.T4.973$Q73.9.KQJ863.KQ4$9.AKQ732.A5.AT82 || E:pass.♣_1.pass.NT_2.pass.♣_3.pass.pass.pass'
    it means:
        N: AKJ.J865.972.J65    (PS:['♠', '♥', '♦', '♣',])
        E: T86542.T4.T4.973
        S: Q73.9.KQJ863.KQ4
        W: 9.AKQ732.A5.AT82
        and E is the declarer_starter. He passed. N bidded ♣_1. S passed. S bidded NT_2
    '''
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    side_dic = list('NESW')
    ff = open ( "data.txt" , "w" )

    for j in range(args.data_size):
        print(f'Generating data: {j+1}/{args.data_size}', end = ' \r')
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
        hand_cards = [i.to_pbn() for i in hands]
        hand_cards_str = '$'.join(hand_cards)
        result_str = side_dic[declarer_starter] + ':' + '.'.join([i.to_str() for i in result if i != 0])
        output_str = hand_cards_str + ' || ' + result_str
        ff.write(output_str+'\n')
        # st()
    ff.close()