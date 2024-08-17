import pygame
import random
import numpy as np
import onnxruntime as ort

# Initialize Pygame
pygame.init()

# Set up the game window
width = 400
height = 400
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Snake Game with ONNX")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Snake and food
snake_block = 20
snake_speed = 15

# Initialize ONNX Runtime session
ort_session = ort.InferenceSession("nextMoveSnake.onnx")

def our_snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(window, GREEN, [x[0], x[1], snake_block, snake_block])

def game_loop():
    game_over = False
    game_close = False

    x1 = width / 2
    y1 = height / 2

    x1_change = 0
    y1_change = 0

    snake_list = []
    length_of_snake = 1

    foodx = round(random.randrange(0, width - snake_block) / 20.0) * 20.0
    foody = round(random.randrange(0, height - snake_block) / 20.0) * 20.0

    clock = pygame.time.Clock()

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True

        # Prepare input for ONNX model
        input_data = np.array([x1, y1, foodx, foody, length_of_snake], dtype=np.float32).reshape(1, -1)
        
        # Run inference
        ort_inputs = {ort_session.get_inputs()[0].name: input_data}
        ort_outs = ort_session.run(None, ort_inputs)
        
        # Interpret the output (assuming it's a direction: 0=left, 1=right, 2=up, 3=down)
        direction = np.argmax(ort_outs[0])
        
        if direction == 0:
            x1_change = -snake_block
            y1_change = 0
        elif direction == 1:
            x1_change = snake_block
            y1_change = 0
        elif direction == 2:
            y1_change = -snake_block
            x1_change = 0
        elif direction == 3:
            y1_change = snake_block
            x1_change = 0

        if x1 >= width or x1 < 0 or y1 >= height or y1 < 0:
            game_close = True
        x1 += x1_change
        y1 += y1_change
        window.fill(BLACK)
        pygame.draw.rect(window, RED, [foodx, foody, snake_block, snake_block])
        snake_head = []
        snake_head.append(x1)
        snake_head.append(y1)
        snake_list.append(snake_head)
        if len(snake_list) > length_of_snake:
            del snake_list[0]

        for x in snake_list[:-1]:
            if x == snake_head:
                game_close = True

        our_snake(snake_block, snake_list)
        pygame.display.update()

        if x1 == foodx and y1 == foody:
            foodx = round(random.randrange(0, width - snake_block) / 20.0) * 20.0
            foody = round(random.randrange(0, height - snake_block) / 20.0) * 20.0
            length_of_snake += 1

        clock.tick(snake_speed)

    pygame.quit()
    quit()

game_loop()