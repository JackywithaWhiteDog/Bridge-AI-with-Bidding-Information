import argparse
import random

import numpy as np

from bridge.match import Match

def parse_args() -> argparse.Namespace:
    """ Parses command line arguments. Returns namespace of arguments."""
    parser = argparse.ArgumentParser(description='Run the matches for bridge AIs')
    parser.add_argument(
        '--agent_a_name',
        type=str,
        choices=["Random", "DDS", "MCTS", "RNN"],
        default="Random",
        help="The name of agent A; currently support Random, DDS, MCTS. (Default: Random)"
    )
    parser.add_argument(
        '--agent_a_n',
        type=int,
        default=10,
        help="Number of possible hands to generate. Only use in MCTS, RNN random agent. (Default: 1)"
    )
    parser.add_argument(
        '--agent_a_reduction',
        type=str,
        choices=['mean', 'max', 'min'],
        default='mean',
        help="Reduction method. Only use in MCTS, RNN random agent. (Default: mean)"
    )
    parser.add_argument(
        '--agent_a_generate_hands',
        type=str,
        choices=['unifrom_hands', 'suit_hands', 'bid_infer_hands', 'bid_forward_infer'],
        default='uniform_hands',
        help="Function name for hands generation. Only use in MCTS agent. (Default: uniform_hands)"
    )
    parser.add_argument(
        '--agent_a_max_threads',
        type=int,
        default=1,
        help="Max threads used by DDS; set to 0 for automatically deciding. Only use in DDS, MCTS, RNN random agent. (Default: 1)"
    )
    parser.add_argument(
        '--agent_a_model_path',
        type=str,
        default="",
        help="Path to model for hands prediction. Only use in RNN agent. (Default: None)"
    )
    parser.add_argument(
        '--agent_a_generation_mode',
        type=str,
        choices=["greedy", "random"],
        default="random",
        help="Generation mode in RNN model. Only use in RNN agent. (Default: random)"
    )
    parser.add_argument(
        '--agent_a_t',
        type=float,
        default=1.0,
        help="Temperature to predict hands. Only use in RNN random agent. (Default: 1.0)"
    )
    parser.add_argument(
        '--agent_b_name',
        type=str,
        choices=["Random", "DDS", "MCTS", "RNN"],
        default="DDS",
        help="The name of agent B; currently support Random, DDS, MCTS. (Default: DDS)"
    )
    parser.add_argument(
        '--agent_b_n',
        type=int,
        default=10,
        help="Number of possible hands to generate. Only use in MCTS, RNN random agent. (Default: 1)"
    )
    parser.add_argument(
        '--agent_b_reduction',
        type=str,
        choices=['mean', 'max', 'min'],
        default='mean',
        help="Reduction method. Only use in MCTS, RNN random agent. (Default: mean)"
    )
    parser.add_argument(
        '--agent_b_generate_hands',
        type=str,
        choices=['unifrom_hands', 'suit_hands', 'bid_infer_hands', 'bid_forward_infer'],
        default='uniform_hands',
        help="Function name for hands generation. Only use in MCTS agent. (Default: uniform_hands)"
    )
    parser.add_argument(
        '--agent_b_max_threads',
        type=int,
        default=1,
        help="Max threads used by DDS; set to 0 for automatically deciding. Only use in DDS, MCTS, RNN random agent. (Default: 1)"
    )
    parser.add_argument(
        '--agent_b_model_path',
        type=str,
        default="",
        help="Path to model for hands prediction. Only use in RNN agent. (Default: None)"
    )
    parser.add_argument(
        '--agent_b_generation_mode',
        type=str,
        choices=["greedy", "random"],
        default="random",
        help="Generation mode in RNN model. Only use in RNN agent. (Default: random)"
    )
    parser.add_argument(
        '--agent_b_t',
        type=float,
        default=1.0,
        help="Temperature to predict hands. Only use in RNN random agent. (Default: 1.0)"
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
        '--output_file',
        type=str,
        default=None,
        help="Path to output matching results"
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
    agent_a_kwargs = {k.replace('agent_a_', ''): v for k, v in args.__dict__.items() if k.startswith('agent_a_')}
    agent_b_kwargs = {k.replace('agent_b_', ''): v for k, v in args.__dict__.items() if k.startswith('agent_b_')}
    match = Match(
        agent_a_kwargs=agent_a_kwargs,
        agent_b_kwargs=agent_b_kwargs,
        num_games=args.num_games,
        num_cards_in_hand=args.num_cards_in_hand,
        output_file=args.output_file
    )
    match.run()
    print("Completed")
