import random
from abc import ABC, abstractmethod, abstractstaticmethod
from typing import Dict, Union
from collections import deque


class AdaptiveUpdatePolicy(ABC):

    @abstractmethod
    def next_to_update(self, metrics: Dict[str, float]) -> Union[str, None]:
        """Decides which component to update next: actor or critic."""
        pass

    @abstractstaticmethod
    def get_type() -> str:
        """Returns the type of the update policy."""
        pass


class DefaultUpdatePolicy(AdaptiveUpdatePolicy):
    """Alternates between actor and critic."""

    is_actor_updated: bool = False

    def next_to_update(self, metrics: Dict[str, float]) -> Union[str, None]:
        next_to_update = "actor" if not self.is_actor_updated else "critic"
        self.is_actor_updated = not self.is_actor_updated
        return next_to_update

    @staticmethod
    def get_type() -> str:
        return "default"


class RandomUpdatePolicy(AdaptiveUpdatePolicy):
    """Randomly chooses between actor and critic."""

    def __init__(self, actor_prob: float):
        self.actor_prob = actor_prob

    def next_to_update(self, metrics: Dict[str, float]) -> Union[str, None]:
        return "actor" if random.random() < self.actor_prob else "critic"

    @staticmethod
    def get_type() -> str:
        return "random"


class HeuristicUpdatePolicy:
    """Chooses between actor and critic based on the loss values."""
    
    def __init__(self, window_size: int = 100):
        self.min_actor_loss = float('inf')
        self.max_actor_loss = float('-inf')
        self.min_critic_loss = float('inf')
        self.max_critic_loss = float('-inf')
        # Using deque to automatically evict old values
        self.actor_loss_window = deque(maxlen=window_size)
        self.critic_loss_window = deque(maxlen=window_size)
        self.window_size = window_size
        self.last_updated = None

    def update_loss_statistics(self, metrics: Dict[str, float]):
        actor_loss = metrics.get('actor_loss', None)
        critic_loss = metrics.get('critic_loss', None)

        if actor_loss is not None:
            self.min_actor_loss = min(self.min_actor_loss, actor_loss)
            self.max_actor_loss = max(self.max_actor_loss, actor_loss)
            self.actor_loss_window.append(actor_loss)

        if critic_loss is not None:
            self.min_critic_loss = min(self.min_critic_loss, critic_loss)
            self.max_critic_loss = max(self.max_critic_loss, critic_loss)
            self.critic_loss_window.append(critic_loss)

    def next_to_update(self, metrics: Dict[str, float]) -> Union[str, None]:
        self.update_loss_statistics(metrics)

        if len(self.actor_loss_window) < self.window_size or len(self.critic_loss_window) < self.window_size:
            next_to_update = "actor" if self.last_updated is None or self.last_updated == "critic" else "critic"
            self.last_updated = next_to_update
            return next_to_update

        # Normalize the losses
        last_actor_loss = self.actor_loss_window[-1]
        last_critic_loss = self.critic_loss_window[-1]
        
        norm_actor_loss = (last_actor_loss - self.min_actor_loss) / \
            (self.max_actor_loss - self.min_actor_loss + 1e-10)
        norm_critic_loss = (last_critic_loss - self.min_critic_loss) / \
            (self.max_critic_loss - self.min_critic_loss + 1e-10)

        # Introducing stochasticity
        decision_score_actor = norm_actor_loss + random.uniform(0, 1) * 0.1
        decision_score_critic = norm_critic_loss + random.uniform(0, 1) * 0.1

        if decision_score_actor > decision_score_critic:
            self.last_updated = "actor"
            return "actor"
        else:
            self.last_updated = "critic"
            return "critic"

    @staticmethod
    def get_type() -> str:
        return "heuristic"
