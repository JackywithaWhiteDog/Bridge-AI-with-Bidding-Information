import abc
from dataclasses import dataclass
import random

from bridge.state import State
from bridge.hand import Card

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

AGENT_DICT = {
    "Random": RandomAgent
}
