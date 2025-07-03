import numpy as np

class GridWorld:
    def __init__(self, size=3):
        self.size = size
        self.grid = np.zeros((size, size))
        self.start_pos = (0, 0)
        self.goal_pos = (size-1, size-1)
        self.obstacle_pos = (1, 1)
        self.current_pos = self.start_pos
        self.actions = ['up', 'right', 'down', 'left']
        self.action_effects = {
            0: (-1, 0),  # up
            1: (0, 1),   # right
            2: (1, 0),   # down
            3: (0, -1)   # left
        }
        
    def reset(self):
        self.current_pos = self.start_pos
        return self.get_state()
    
    def get_state(self):
        return self.current_pos[0] * self.size + self.current_pos[1]
    
    def step(self, action):
        if self.current_pos == self.goal_pos:
            return self.get_state(), 0, True, {}
        
        # Calculate new position
        effect = self.action_effects[action]
        new_pos = (self.current_pos[0] + effect[0], self.current_pos[1] + effect[1])
        
        # Check if new position is valid
        if self.is_valid_position(new_pos):
            self.current_pos = new_pos
        
        # Calculate reward
        reward = self.get_reward()
        done = self.current_pos == self.goal_pos
        
        return self.get_state(), reward, done, {}
    
    def is_valid_position(self, pos):
        row, col = pos
        if row < 0 or row >= self.size or col < 0 or col >= self.size:
            return False
        if pos == self.obstacle_pos:
            return False
        return True
    
    def get_reward(self):
        if self.current_pos == self.goal_pos:
            return 100
        elif self.current_pos == self.obstacle_pos:
            return -100
        else:
            return -1
    
    def get_state_from_pos(self, pos):
        return pos[0] * self.size + pos[1]
    
    def get_pos_from_state(self, state):
        return (state // self.size, state % self.size)
