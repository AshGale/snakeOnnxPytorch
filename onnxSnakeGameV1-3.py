import pygame
import random
import numpy as np
import onnxruntime as ort
import logging
import scipy.special

# Set up logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("snake_game.log"),
        logging.StreamHandler()
    ]
)

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
pygame.display.set_caption("Snake Game with ONNX")

# Directional constants
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Initialize ONNX Runtime session
ort_session = ort.InferenceSession("nextMoveSnake.onnx")

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

def softmax(x):
    return scipy.special.softmax(x)

def get_move_from_onnx(snake, food):
    head = snake.get_head_position()
    # Prepare input for ONNX model (snake_head_x, snake_head_y, food_x, food_y, snake_length)
    input_data = np.array([head[0], head[1], food.position[0], food.position[1], snake.length], dtype=np.float32).reshape(1, -1)
    
    # Run inference
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # Get the raw output (logits for each direction)
    logits = ort_outs[0][0]
    
    # Apply softmax to convert logits to probabilities
    probabilities = softmax(logits)
    
    # Interpret the output (assuming it's a direction: 0=up, 1=right, 2=down, 3=left)
    direction_index = np.argmax(probabilities)
    confidence = probabilities[direction_index]
    
    # Map the direction index to the actual direction
    direction_map = {0: UP, 1: RIGHT, 2: DOWN, 3: LEFT}
    return direction_map[direction_index], confidence

def game_loop():
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
    surface = pygame.Surface(screen.get_size())
    surface = surface.convert()
    surface.fill(WHITE)

    snake = Snake()
    food = Food(snake)

    score = 0
    game_over = False

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "QUIT"
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return "QUIT"
                elif event.key == pygame.K_r:
                    return "RESTART"

        move, confidence = get_move_from_onnx(snake, food)
        snake.turn(move)
        
        head = snake.get_head_position()
        direction_index = {UP: 0, RIGHT: 1, DOWN: 2, LEFT: 3}
        logging.info(f"Head: ({head[0]}, {head[1]}), Food: ({food.position[0]}, {food.position[1]}), Move: {direction_index[move]}, Confidence: {confidence:.2%}")
        collision = snake.move()

        if snake.get_head_position() == food.position:
            snake.length += 1
            score += 1
            food.randomize_position()
            logging.info(f"Snake ate food! New length: {snake.length}")

        if collision or snake.length == GRID_WIDTH * GRID_HEIGHT:
            game_over = True

        surface.fill(WHITE)
        snake.draw(surface)
        food.draw(surface)
        screen.blit(surface, (0, 0))
        pygame.display.update()
        clock.tick(10)  # Adjust game speed here

    return "RESTART"

def main():
    while True:
        result = game_loop()
        if result == "QUIT":
            break
        # If result is "RESTART", the loop will continue and start a new game

    pygame.quit()

if __name__ == "__main__":
    main()
