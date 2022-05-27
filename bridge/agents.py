import abc
from dataclasses import dataclass
import random

from bridge.state import State
from bridge.hand import Card, hands2pbn
from bridge.dds import dds_score
from bridge.dds.utils import set_max_threads

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

AGENT_DICT = {
    "Random": RandomAgent,
    "DDS": DDSAgent
}
