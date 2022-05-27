import argparse
import random

import numpy as np

from bridge.match import Match
from bridge.agents import AGENT_DICT

def parse_args() -> argparse.Namespace:
    """ Parses command line arguments. Returns namespace of arguments."""
    parser = argparse.ArgumentParser(description='Run the matches for bridge AIs')
    parser.add_argument(
        '--declarer_agent',
        type=str,
        choices=AGENT_DICT.keys(),
        default="Random",
        help="The name of declarer agent; currently support Random, DDS. (Default: Random)"
    )
    parser.add_argument(
        '--defender_agent',
        type=str,
        choices=AGENT_DICT.keys(),
        default="DDS",
        help="The name of defender agent; currently support Random, DDS. (Default: DDS)"
    )
    parser.add_argument(
        '--num_games',
        type=int,
        default=100,
        help="Number of games to play. (Default: 100)"
    )
    parser.add_argument(
        '--num_cards_in_hand',
        type=int,
        default=13,
        help="Number of cards in players' hands; generally be 13 or 5. (Default: 13)"
    )
    parser.add_argument(
        '--max_threads',
        type=int,
        default=1,
        help="Max threads used by DDS; set to 0 for automatically deciding. (Default: 1)"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help="Random seed for reproduction. (Default: None)"
    )
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)
    match = Match(
        declarer_agent_name=args.declarer_agent,
        defender_agent_name=args.defender_agent,
        num_games=args.num_games,
        num_cards_in_hand=args.num_cards_in_hand,
        max_threads=args.max_threads
    )
    match.run()
    print("Completed")
