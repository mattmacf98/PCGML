import numpy as np
import pygame
import heapq

import gymnasium as gym
from gymnasium import spaces


def dijkstra(maze, start, end):
    """ Dijkstra's algorithm to compute the shortest path in the maze. """
    size = len(maze)
    distances = np.full((size, size), np.inf)
    distances[tuple(start)] = 0
    queue = [(0, tuple(start))]  # (distance, node)

    # if start or end are in walls early return
    if maze[start[0], start[1]] == 1 or maze[end[0], end[1]] == 1:
        return np.inf

    while len(queue) > 0:
        current_dist, current_pos = heapq.heappop(queue)

        if current_pos == tuple(end):
            return current_dist  # Return the distance to the end

        x, y = current_pos

        # Explore neighbors (up, down, left, right)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and maze[nx, ny] == 0:  # Check within bounds and no wall
                new_dist = current_dist + 1
                if new_dist < distances[nx, ny]:
                    distances[nx, ny] = new_dist
                    heapq.heappush(queue, (new_dist, (nx, ny)))

    return np.inf  # No path to the end


class MazeWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 64}

    def __init__(self, render_mode=None, size=5):
        self.size = size
        self.window_size = 512

        self.observation_space = spaces.Dict(
            {
                "start": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "end": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "maze": spaces.Box(0, 1, shape=(size, size), dtype=int)
            }
        )

        self.action_space = spaces.Tuple((
            spaces.Discrete(self.size),  # X-coordinate
            spaces.Discrete(self.size),  # Y-coordinate
            spaces.Discrete(2)  # Toggle (0 for path, 1 for wall)
        ))

        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"start": self._start_location, "end": self._end_location, "maze": self._maze}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(self._start_location - self._end_location, ord=1)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._start_location = np.array([0, 0])
        self._end_location = np.array([self.size - 1, self.size - 1])
        self._distance = np.linalg.norm(self._start_location - self._end_location, ord=1)

        self._maze = np.zeros((self.size, self.size))

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        x, y = action
        terminated = False
        if (x == self._start_location[0] and y == self._start_location[1]) or (x == self._end_location[0] and y == self._end_location[1]):
            # invalid move
            reward = -1 * self.size * self.size
        else:
            self._maze[x, y] = int((self._maze[x, y] + 1) % 2)
            path_length = dijkstra(self._maze, self._start_location, self._end_location)

            if path_length == np.inf:
                # Penalize if the maze is unsolvable
                reward = -5 * self.size * self.size
            else:
                # Reward based on the difficulty of the maze (longer paths are better)
                reward = (path_length - self._distance) * self.size
                # reward lots of wall
                reward += np.sum(self._maze) * 2.5
                if path_length > self.size * 2.2:
                    # a big bonus for finding a good maze
                    terminated = True
                    reward += self.size * self.size * 25

        if self.render_mode == "human":
            self._render_frame()

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info

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

        for row in range(self.size):
            for col in range(self.size):
                if self._maze[row][col] == 1:
                    pygame.draw.rect(canvas, (0, 0, 0), pygame.Rect(pix_square_size * np.array([row, col]), (pix_square_size, pix_square_size)))  # wall

        pygame.draw.circle(canvas, (0, 0, 255), (self._start_location + 0.5) * pix_square_size, pix_square_size / 3) # start
        pygame.draw.circle(canvas, (0, 255, 0), (self._end_location + 0.5) * pix_square_size, pix_square_size / 3)  # end

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

