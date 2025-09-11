import pygame
import random
import numpy as np
from numpy.ma.core import right_shift, count
from traits.trait_types import false

pygame.init()

width=300
height=300
grid_size=5
cell_size=width//grid_size
fps=5

white = (255,255,255)
green = (0,255,0)
red = (255,0,0)
black = (0,0,0)

up = (0,-1)
down = (0,1)
left = (-1,0)
right = (1,0)
actions = [up,down,left,right]

win = pygame.display.set_mode((width,height))
pygame.display.set_caption("Snake Q-Learning - Small Grid")
font = pygame.font.SysFont(None,24)

class Food:
    def __init__(self):
        self.position = (random.randint(0,grid_size-1),random.randint(0,grid_size-1))

    def randomize(self):
        self.position = (random.randint(0,grid_size-1),random.randint(0,grid_size-1))


class Snake:
    def __init__(self):
        self.positions=[(2,2)]
        self.direction=random.choice(actions)
    def move(self,action):
        self.direction=action
        head_x,head_y=self.positions[0]
        dir_x,dir_y=self.direction
        new_head = (head_x+dir_x,head_y+dir_y)
        self.positions=[new_head]+self.positions[:-1]
    def grow(self):
        tail = self.positions[-1]
        self.positions.append(tail)
    def collision(self):
        head = self.positions[0]
        return (
            head in self.positions[1:] or
            head[0] < 0 or head[0] >=grid_size or
            head[1] < 0 or head[1] >= grid_size
         )

q_table = np.zeros((grid_size,grid_size,4))

alpha = 0.1
gamma = 0.9
epsilon = 0.2
num_episodes = 200
count_of_positive_reward=0

def get_state(snake):
    return snake.positions[0]

def get_reward(snake,food):
    if snake.positions[0]==food.position:
        return 10
    elif snake.collision():
        return -100
    else:
        return -1

clock = pygame.time.Clock()

for episode in range(1,num_episodes+1):
    snake = Snake()
    food = Food()
    done = False
    total_reward = 0

    while not done:
        clock.tick(fps)

        for event in pygame.event.get():
            if event.type == pygame.quit:
                pygame.quit()
                exit()
        current_state = get_state(snake)

        if random.uniform(0,1)<epsilon:
            action_idx = random.randint(0,3)
        else:
            action_idx = np.argmax(q_table[current_state[0],current_state[1]])

        action = actions[action_idx]
        snake.move(action)

        reward = get_reward(snake,food)
        total_reward+=reward

        next_state = get_state(snake)

        if 0 <= next_state[0] < grid_size and 0 <= next_state[1] < grid_size:
            q_table[current_state[0],current_state[1],action_idx] = q_table[current_state[0],current_state[1],action_idx] + alpha *(reward + gamma * np.max(q_table[next_state[0],next_state[1]])- q_table[current_state[0],current_state[1],action_idx])

        if snake.positions[0] == food.position:
            snake.grow()
            food.randomize()

        win.fill(white)
        for i in range(grid_size):
            for j in range(grid_size):
                pygame.draw.rect(win,black,(i*cell_size,j*cell_size,cell_size,cell_size),1)

        for pos in snake.positions:
            pygame.draw.rect(win, green, (pos[0] * cell_size, pos[1] * cell_size, cell_size, cell_size))
            label = font.render("Snake", True, black)
            win.blit(label, (pos[0] * cell_size + 5, pos[1] * cell_size + 5))

        pygame.draw.rect(win, red, (food.position[0] * cell_size, food.position[1] * cell_size, cell_size, cell_size))
        label = font.render("Food", True, black)
        win.blit(label, (food.position[0] * cell_size + 5, food.position[1] * cell_size + 5))

        pygame.display.update()

        if snake.collision():
            done = True
    if total_reward > 0:
        count_of_positive_reward+=1

    print(f"Episode {episode} finished with Total Reward: {total_reward}")

print(f"The Episodes ended with a total of {count_of_positive_reward} positive rewards during {episode} episodes")