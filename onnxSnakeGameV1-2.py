import pygame
import random
import numpy as np
import onnxruntime as ort
import logging

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
snake_speed = 5

# Initialize ONNX Runtime session
ort_session = ort.InferenceSession("nextMoveSnake.onnx")

def our_snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(window, GREEN, [x[0], x[1], snake_block, snake_block])

def message(msg, color):
    font_style = pygame.font.SysFont(None, 50)
    mesg = font_style.render(msg, True, color)
    window.blit(mesg, [width / 6, height / 3])

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
        while game_close:
            window.fill(BLACK)
            message("You Lost! Press Space-Bar to Play Again or Q to Quit", RED)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return "QUIT"
                    if event.key == pygame.K_SPACE:
                        return "RESTART"

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "QUIT"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return "RESTART"
                if event.key == pygame.K_q:
                    return "QUIT"

        # Prepare input for ONNX model
        input_data = np.array([x1, y1, foodx, foody, length_of_snake], dtype=np.float32).reshape(1, -1)
        
        # Run inference
        ort_inputs = {ort_session.get_inputs()[0].name: input_data}
        ort_outs = ort_session.run(None, ort_inputs)
        
        # Interpret the output (assuming it's a direction: 0=left, 1=right, 2=up, 3=down)
        direction = np.argmax(ort_outs[0])
        
        # Log the model's decision
        direction_map = {0: "left", 1: "right", 2: "up", 3: "down"}
        logging.info(f"Snake position: ({x1}, {y1}), Food position: ({foodx}, {foody}), Length: {length_of_snake}, Model decided to move: {direction_map[direction]}")
        
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

        # Implement screen wrapping
        x1 += x1_change
        y1 += y1_change
        
        if x1 >= width:
            x1 = 0
        elif x1 < 0:
            x1 = width - snake_block
        if y1 >= height:
            y1 = 0
        elif y1 < 0:
            y1 = height - snake_block

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
            logging.info(f"Snake ate food! New length: {length_of_snake}")

        clock.tick(snake_speed)

    return "QUIT"

def main():
    while True:
        result = game_loop()
        if result == "QUIT":
            break
        # If result is "RESTART", the loop will continue and start a new game

    pygame.quit()
    quit()

if __name__ == "__main__":
    main()

