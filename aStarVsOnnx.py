import pygame
import random
import numpy as np
import onnxruntime as ort
from openvino.runtime import Core
import logging
import heapq
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("snake_game.log"),
        logging.StreamHandler()
    ]
)

# Game Constants
WIDTH, HEIGHT = 400, 400
GRID_SIZE = 20
GRID_WIDTH, GRID_HEIGHT = WIDTH // GRID_SIZE, HEIGHT // GRID_SIZE

# Colors
BLACK, WHITE, RED, GREEN = (0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0)

# Directional constants
UP, DOWN, LEFT, RIGHT = (0, -1), (0, 1), (-1, 0), (1, 0)
DIRECTIONS = [UP, RIGHT, DOWN, LEFT]

# Initialize Pygame and ONNX
pygame.init()
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game with A* and ONNX")

openvino_model_path = "./openVino/nextMoveSnake.xml"
onnx_model_path = "./onnx/nextMoveSnake.onnx"

def compile_model_with_fallback(openvino_model_path, onnx_model_path, devices=['NPU', 'GPU', 'CPU']):
    core = Core()
    openvino_model = core.read_model(openvino_model_path)
    
    # Try OpenVINO first
    for device in devices:
        try:
            if device in core.available_devices:
                compiled_model = core.compile_model(openvino_model, device)
                print(f"OpenVINO model compiled successfully on {device}")
                return ('openvino', compiled_model, device)
        except Exception as e:
            print(f"Failed to compile OpenVINO model on {device}: {str(e)}")
    
    # If OpenVINO fails, try ONNX
    print("Falling back to ONNX model")
    try:
        ort_session = ort.InferenceSession(onnx_model_path)
        print("ONNX model loaded successfully")
        return ('onnx', ort_session, 'ONNX Runtime')
    except Exception as e:
        print(f"Failed to load ONNX model: {str(e)}")
    
    raise RuntimeError("Failed to compile/load model with OpenVINO and ONNX")

# Usage
try:
    # mo --input_model ./onnx/nextMoveSnake.onnx --output_dir optimized_model
    model_type, model, used_device = compile_model_with_fallback(openvino_model_path, onnx_model_path)
    
    if model_type == 'openvino':
        # OpenVINO model
        input_layer = model.input(0)
        output_layer = model.output(0)
        print(f"OpenVINO model compiled and running on {used_device}")
        # Your OpenVINO inference code here...
    else:
        # ONNX model
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        print(f"ONNX model loaded and running with {used_device}")
        # Your ONNX inference code here...

except RuntimeError as e:
    print(f"Error: {str(e)}")


class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = self.h = self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f

class Snake:
    def __init__(self):
        self.reset()

    def get_head_position(self):
        return self.positions[0]

    def turn(self, point):
        if self.length > 1 and (point[0] * -1, point[1] * -1) == self.direction:
            return
        self.direction = point

    def move(self):
        cur = self.get_head_position()
        x, y = self.direction
        new = ((cur[0] + x) % GRID_WIDTH, (cur[1] + y) % GRID_HEIGHT)
        if len(self.positions) > 2 and new in self.positions[2:]:
            return True
        self.positions.insert(0, new)
        if len(self.positions) > self.length:
            self.positions.pop()
        return False

    def reset(self):
        self.length = 1
        self.positions = [((GRID_WIDTH // 2), (GRID_HEIGHT // 2))]
        self.direction = random.choice(DIRECTIONS)
        self.color = GREEN
        logging.info("Game Reset")

    def draw(self, surface):
        for p in self.positions:
            r = pygame.Rect((p[0] * GRID_SIZE, p[1] * GRID_SIZE), (GRID_SIZE, GRID_SIZE))
            pygame.draw.rect(surface, self.color, r)
            pygame.draw.rect(surface, BLACK, r, 1)

class Food:
    def __init__(self):
        self.position = (0, 0)
        self.color = RED
        self.randomize_position()

    def randomize_position(self):
        self.position = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))

    def draw(self, surface):
        r = pygame.Rect((self.position[0] * GRID_SIZE, self.position[1] * GRID_SIZE), (GRID_SIZE, GRID_SIZE))
        pygame.draw.rect(surface, self.color, r)
        pygame.draw.rect(surface, BLACK, r, 1)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def get_move_from_model(model_info, snake, food):
    start_time = time.perf_counter_ns()
    
    head = snake.get_head_position()
    input_data = np.array([[head[0], head[1], food.position[0], food.position[1], snake.length]], dtype=np.float32)
    
    model_type, model, _ = model_info
    
    if model_type == 'openvino':
        # OpenVINO inference
        results = model([input_data])[model.output(0)]
    elif model_type == 'onnx':
        # ONNX Runtime inference
        ort_inputs = {model.get_inputs()[0].name: input_data}
        ort_outs = model.run(None, ort_inputs)
        results = ort_outs[0]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Process results
    probabilities = softmax(results[0])
    direction_index = np.argmax(probabilities)
    confidence = probabilities[direction_index]
    
    duration = time.perf_counter_ns() - start_time
    return DIRECTIONS[direction_index], confidence, duration

def astar(maze, start, end):
    start_node = Node(None, start)
    end_node = Node(None, end)
    open_list, closed_set = [], set()
    heapq.heappush(open_list, (0, id(start_node), start_node))

    while open_list:
        current_node = heapq.heappop(open_list)[2]
        if current_node == end_node:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        closed_set.add(current_node.position)
        for new_position in DIRECTIONS:
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

def get_move_from_astar(snake, food):
    start_time = time.perf_counter_ns()
    head = snake.get_head_position()
    maze = [[1 if (x, y) in snake.positions[1:] else 0 for x in range(GRID_WIDTH)] for y in range(GRID_HEIGHT)]
    path = astar(maze, head, food.position)

    move = snake.direction
    if path and len(path) > 1:
        next_pos = path[1]
        for direction in DIRECTIONS:
            new_head = ((head[0] + direction[0]) % GRID_WIDTH, (head[1] + direction[1]) % GRID_HEIGHT)
            if new_head == next_pos:
                move = direction
                break

    duration = time.perf_counter_ns() - start_time
    return move, duration

def game_loop(initial_use_onnx=True):
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((WIDTH, HEIGHT), 0, 32)
    surface = pygame.Surface(screen.get_size()).convert()
    surface.fill(WHITE)

    snake = Snake()
    food = Food()
    use_onnx = initial_use_onnx

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "QUIT"
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return "QUIT"
                elif event.key == pygame.K_r:
                    return "RESTART"
                elif event.key == pygame.K_t:
                    use_onnx = not use_onnx
                    logging.info(f"Switched to {'ONNX' if use_onnx else 'A*'} pathfinding")

        model_info = compile_model_with_fallback(openvino_model_path, onnx_model_path)
    

        onnx_move, onnx_confidence, onnx_duration = get_move_from_model(model_info, snake, food)
        astar_move, astar_duration = get_move_from_astar(snake, food)

        move = onnx_move if use_onnx else astar_move

        previous_head = snake.get_head_position()
        snake.turn(move)
        collision = snake.move()
        new_head = snake.get_head_position()

        actual_move = (new_head[0] - previous_head[0], new_head[1] - previous_head[1])
        if actual_move[0] > 1: actual_move = (-1, actual_move[1])
        elif actual_move[0] < -1: actual_move = (1, actual_move[1])
        if actual_move[1] > 1: actual_move = (actual_move[0], -1)
        elif actual_move[1] < -1: actual_move = (actual_move[0], 1)

        # Combined log entry
        log_entry = (f"Head: {new_head}, Food: {food.position}, "
                     f"Snake Length: {snake.length}, "
                     f"ONNX Move: {onnx_move}, ONNX Time: {onnx_duration / 1_000_000:.3f} ms, "
                     f"ONNX Confidence: {onnx_confidence:.3f}, "
                     f"A* Move: {astar_move}, A* Time: {astar_duration / 1_000_000:.3f} ms, "
                     f"Chosen Move: {move} ({('ONNX' if use_onnx else 'A*')})")
        logging.info(log_entry)

        if new_head == food.position:
            snake.length += 1
            food.randomize_position()

        if collision or snake.length == GRID_WIDTH * GRID_HEIGHT:
            return "RESTART"

        surface.fill(WHITE)
        snake.draw(surface)
        food.draw(surface)
        screen.blit(surface, (0, 0))
        pygame.display.update()
        clock.tick(10)

def main():
    use_onnx = True
    logging.info("Press 'T' to toggle between ONNX and A* pathfinding.")
    while True:
        result = game_loop(use_onnx)
        if result == "QUIT":
            break
        elif result == "RESTART":
            logging.info("Restarting game")
    pygame.quit()

if __name__ == "__main__":
    main()