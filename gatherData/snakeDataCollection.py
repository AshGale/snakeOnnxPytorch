import pygame
import random
import heapq
import csv
from collections import deque

# Initialize Pygame
pygame.init()

# Game Constants
WIDTH = 400
HEIGHT = 400
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Initialize game window
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game - Data Collection")


# A* Algorithm
class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f


def astar(maze, start, end):
    start_node = Node(None, start)
    end_node = Node(None, end)

    open_list = []
    closed_set = set()

    heapq.heappush(open_list, (0, id(start_node), start_node))

    max_iterations = GRID_WIDTH * GRID_HEIGHT * 4
    iterations = 0

    while open_list and iterations < max_iterations:
        iterations += 1
        current_node = heapq.heappop(open_list)[2]

        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]

        closed_set.add(current_node.position)

        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            node_position = ((current_node.position[0] + new_position[0]) % GRID_WIDTH,
                             (current_node.position[1] + new_position[1]) % GRID_HEIGHT)

            if maze[node_position[1]][node_position[0]] == 1:
                continue

            new_node = Node(current_node, node_position)

            if new_node.position in closed_set:
                continue

            new_node.g = current_node.g + 1
            new_node.h = min(abs(new_node.position[0] - end_node.position[0]), GRID_WIDTH - abs(new_node.position[0] - end_node.position[0])) + \
                         min(abs(new_node.position[1] - end_node.position[1]), GRID_HEIGHT - abs(new_node.position[1] - end_node.position[1]))
            new_node.f = new_node.g + new_node.h

            if any(new_node == open_node and new_node.g > open_node.g for _, _, open_node in open_list):
                continue

            heapq.heappush(open_list, (new_node.f, id(new_node), new_node))

    return None


# Snake Game
class Snake:
    def __init__(self):
        self.length = 1
        self.positions = [((GRID_WIDTH // 2), (GRID_HEIGHT // 2))]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.color = GREEN

    def get_head_position(self):
        return self.positions[0]

    def turn(self, point):
        if self.length > 1 and (point[0] * -1, point[1] * -1) == self.direction:
            return
        else:
            self.direction = point

    def move(self):
        cur = self.get_head_position()
        x, y = self.direction
        new = (((cur[0] + x) % GRID_WIDTH), (cur[1] + y) % GRID_HEIGHT)
        if len(self.positions) > 2 and new in self.positions[2:]:
            return True  # Collision occurred
        else:
            self.positions.insert(0, new)
            if len(self.positions) > self.length:
                self.positions.pop()
        return False  # No collision

    def reset(self):
        self.length = 1
        self.positions = [((GRID_WIDTH // 2), (GRID_HEIGHT // 2))]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])

    def draw(self, surface):
        for p in self.positions:
            r = pygame.Rect((p[0] * GRID_SIZE, p[1] * GRID_SIZE), (GRID_SIZE, GRID_SIZE))
            pygame.draw.rect(surface, self.color, r)
            pygame.draw.rect(surface, BLACK, r, 1)


class Food:
    def __init__(self, snake):
        self.snake = snake
        self.position = (0, 0)
        self.color = RED
        self.randomize_position()

    def randomize_position(self):
        while True:
            position = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if position not in self.snake.positions:
                self.position = position
                break

    def draw(self, surface):
        r = pygame.Rect((self.position[0] * GRID_SIZE, self.position[1] * GRID_SIZE), (GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(surface, self.color, r)
        pygame.draw.rect(surface, BLACK, r, 1)


def create_maze(snake, food):
    maze = [[0 for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    for body_part in snake.positions[1:]:
        maze[body_part[1]][body_part[0]] = 1
    return maze


def get_move(snake, food):
    head = snake.get_head_position()
    maze = create_maze(snake, food)
    path = astar(maze, head, food.position)

    if path and len(path) > 1:
        next_move = path[1]
        for direction in [RIGHT, LEFT, DOWN, UP]:
            new_head = ((head[0] + direction[0]) % GRID_WIDTH, (head[1] + direction[1]) % GRID_HEIGHT)
            if new_head == next_move and new_head not in snake.positions[1:]:
                return direction
    return snake.direction


def main():
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)

    surface = pygame.Surface(screen.get_size())
    surface = surface.convert()
    surface.fill(WHITE)

    snake = Snake()
    food = Food(snake)

    score = 0
    game_count = 0
    max_games = 1000  # Set a limit to the number of games

    # Open a CSV file for writing the training data
    with open('snake_training_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['game_id', 'snake_head_x', 'snake_head_y', 'food_x', 'food_y', 'snake_length', 'direction'])

        while game_count < max_games:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            move = get_move(snake, food)
            snake.turn(move)
            collision = snake.move()

            if snake.get_head_position() == food.position:
                snake.length += 1
                score += 1
                food.randomize_position()

            # Log the game state and the chosen move
            head = snake.get_head_position()
            direction_index = {UP: 0, RIGHT: 1, DOWN: 2, LEFT: 3}
            writer.writerow([game_count, head[0], head[1], food.position[0], food.position[1], snake.length,
                             direction_index[move]])

            if collision or snake.length == GRID_WIDTH * GRID_HEIGHT:
                game_count += 1
                snake.reset()
                food.randomize_position()
                score = 0

            # surface.fill(WHITE)
            # snake.draw(surface)
            # food.draw(surface)
            # screen.blit(surface, (0, 0))
            # pygame.display.update()
            # clock.tick(100)  # Increased speed for faster data collection

        print(f"Data collection complete. {game_count} games played.")


# Directional constants
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

if __name__ == "__main__":
    main()