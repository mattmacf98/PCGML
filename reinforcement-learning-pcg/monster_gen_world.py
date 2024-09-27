import random
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class MonsterWorldGenEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self):
        self.opponent_stats = {"health": 50, "speed": 25, "damage": 25, "armor": 25}
        self.observation_space = spaces.Dict(
            {
                "stats": spaces.Box(1, 50, shape=(4,), dtype=int),
            }
        )

        self.action_space = spaces.Discrete(8)

        self._action_to_direction = {
            0: np.array([1,0,0,0]),
            1: np.array([-1,0,0,0]),
            2: np.array([0,1,0,0]),
            3: np.array([0,-1,0,0]),
            4: np.array([0, 0, 1, 0]),
            5: np.array([0, 0, -1, 0]),
            6: np.array([0, 0, 0, 1]),
            7: np.array([0, 0, 0, -1]),
        }

    def _get_obs(self):
        return {"stats": self._stats}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._stats = self.np_random.integers(1, 50, size=4, dtype=int)

        observation = self._get_obs()

        return observation, {}

    def step(self, action):
        direction = self._action_to_direction[action]
        self._stats = np.clip(self._stats + direction, 1, 50)

        observation = self._get_obs()

        win_rate = self.play_opponent()

        reward = (0.5 - abs(win_rate - 0.5)) / 0.5
        terminated = 1.0 - reward <= 0.01
        if terminated:
            reward = 10
        else:
            reward -= 1.0

        return observation, reward, terminated, False, {}

    def play_opponent(self):
        wins = 0
        for i in range(100):
            monster1 = Monster(self._stats[0], self._stats[1], self._stats[2], self._stats[3])
            monster2 = Monster(self.opponent_stats['health'], self.opponent_stats['damage'], self.opponent_stats['speed'], self.opponent_stats['armor'])
            while monster1.health > 0 and monster2.health > 0:
                monster1.take_turn(monster2)
                if monster2.health <= 0:
                    wins += 1
                    break
                monster2.take_turn(monster1)
                if monster1.health <= 0:
                    break
        return wins / 100

    def render(self):
        pass


class Monster:
    def __init__(self, health, damage, speed, armor):
        self.health = health
        self.damage = damage
        self.speed = speed
        self.armor = armor

    def take_turn(self, opponent):
        # Calculate if opponent dodges
        dodge_chance = opponent.speed / 100
        if random.random() < dodge_chance:
            return

        # Calculate damage dealt
        damage_dealt = max(1, self.damage - opponent.armor)
        opponent.health -= damage_dealt
