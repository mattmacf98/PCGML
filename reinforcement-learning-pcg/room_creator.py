import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces


class RoomEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, size=5):
        self.size = size
        self.window_size = 512
        self.furniture_count = 5

        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(0, size, shape=(size, size), dtype=int)
            }
        )

        self.action_space = spaces.Discrete(size * size * self.furniture_count + 1)

        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return { "grid": self._grid}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._grid = np.zeros((self.size, self.size), dtype=int)

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation, {}

    def step(self, action):
        reward = 0
        terminated = action == self.size * self.size * self.furniture_count

        if terminated:
            reward = 100
            for row in range(1, self.size - 1):
                for col in range(1, self.size - 1):
                    cur = self._grid[row][col]
                    if cur == self._grid[row + 1][col] or cur == self._grid[row - 1][col] or cur == self._grid[row][col + 1] or cur == self._grid[row][col - 1]:
                        reward -= 5
        else:
            action_cell = int(action / self.furniture_count)
            action_row = int(action_cell / self.size)
            action_col = int(action_cell % self.size)
            value = int(action % self.furniture_count)
            if self._grid[action_row][action_col] == value:
                reward -= 1

            self._grid[action_row][action_col] = value

        observation = self._get_obs()
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255,255,255))
        pix_square_size = (self.window_size / self.size)

        for r in range(self.size):
            for c in range(self.size):
                if self._grid[r][c] == 1:
                    color = (255,0,0)
                elif self._grid[r][c] == 2:
                    color = (0, 255, 0)
                elif self._grid[r][c] == 3:
                    color = (255, 255, 0)
                elif self._grid[r][c] == 4:
                    color = (0, 0, 255)
                else:
                    color = (255, 255, 255)
                pygame.draw.rect(canvas, color, pygame.Rect(pix_square_size * np.array([r,c]), (pix_square_size, pix_square_size)))

        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

