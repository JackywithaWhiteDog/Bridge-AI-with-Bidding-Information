from dataclasses import dataclass, field, InitVar
from typing import Literal, List

from bridge.constants import Suit, Side
from bridge.hand import Hand
from bridge.state import State, Result
from bridge.agents import BaseAgent

@dataclass
class Game:
    """Class for bridge game"""
    declarer_agent: BaseAgent
    defender_agent: BaseAgent

    hands: InitVar[List[Hand]]
    trump: InitVar[Literal[0, 1, 2, 3, 4]] = Suit.NT
    declarer: InitVar[Literal[0, 1, 2, 3]] = Side.NORTH
    declarer_goal: InitVar[Literal[7, 8, 9, 10, 11, 12, 13]] = 7
    end_when_goal_achieved: InitVar[bool] = False
    num_cards_in_hand: InitVar[int] = 13

    state: State = field(init=False)

    def __post_init__(
        self,
        hands,
        trump,
        declarer,
        declarer_goal,
        end_when_goal_achieved,
        num_cards_in_hand
    ):

        current_player = Side.get_left(declarer)
        self.state = State(
            hands=hands,
            trump=trump,
            first=current_player,
            current_player=current_player,
            declarer=declarer,
            declarer_goal=declarer_goal,
            end_when_goal_achieved=end_when_goal_achieved,
            num_cards_in_hand=num_cards_in_hand
        )

    def run(self) -> Result:
        while not self.state.reach_end():
            if self.state.current_player in self.state.declarer_side:
                card = self.declarer_agent.get_action(self.state)
            else:
                card = self.defender_agent.get_action(self.state)
            self.state.apply_action(card)
        return self.state.get_result()
