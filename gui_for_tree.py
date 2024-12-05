import copy
import time
import pygame
import sys
import tkinter as tk
import math

pygame.init()   
pygame.font.init()  # Explicitly initialize the font module

# Constants
ROWS = 6
COLS = 7
SQUARESIZE = 100  # Size of each square
RADIUS = 40  # Radius of the discs
WIDTH = COLS * SQUARESIZE
HEIGHT = (ROWS + 1) * SQUARESIZE
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)

# Fonts
FONT = pygame.font.SysFont("monospace", 75)
SMALL_FONT = pygame.font.SysFont("monospace", 40)


class TreeVisualizer:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=1200, height=800, bg='white')
        self.canvas.pack(expand=True, fill='both')
        
        # Visual parameters
        self.node_radius = 25
        self.vertical_gap = 100
        self.horizontal_spacing = 80
        
    def draw_node(self, x, y, value, is_max, column=None):
        # Draw circle
        self.canvas.create_oval(x-self.node_radius, y-self.node_radius,
                              x+self.node_radius, y+self.node_radius,
                              fill='lightblue')
        
        # Draw value
        self.canvas.create_text(x, y, text=str(value))
        
        # Draw Max/Min and column info
        label = "Max" if is_max else "Min"
        if column is not None:
            label += f" (Col:{column})"
        self.canvas.create_text(x, y-self.node_radius-10, text=label)
        
    def draw_line(self, x1, y1, x2, y2):
        self.canvas.create_line(x1, y1, x2, y2)
        
    def visualize_tree(self, node, x, y, is_max=True):
        self.draw_node(x, y, node['value'], is_max, node.get('column'))
        
        if node['children']:
            num_children = len(node['children'])
            total_width = (num_children - 1) * self.horizontal_spacing
            start_x = x - total_width/2
            
            for i, child in enumerate(node['children']):
                child_x = start_x + i * self.horizontal_spacing
                child_y = y + self.vertical_gap
                self.draw_line(x, y+self.node_radius, child_x, child_y-self.node_radius)
                self.visualize_tree(child, child_x, child_y, not is_max)

def show_tree(tree):
    root = tk.Tk()
    root.title("Minimax Tree Visualization")
    viz = TreeVisualizer(root)
    viz.visualize_tree(tree, 600, 50)  # Start at center-top
    root.mainloop()



def draw_board(board, screen):
    # Draw the blue board with empty black circles
    for c in range(COLS):
        for r in range(ROWS):
            # Draw blue rectangle
            pygame.draw.rect(screen, BLUE, 
                           (c * SQUARESIZE, 
                            (r + 1) * SQUARESIZE, 
                            SQUARESIZE, 
                            SQUARESIZE))
            # Draw empty black circle
            pygame.draw.circle(screen, BLACK,
                             (int(c * SQUARESIZE + SQUARESIZE / 2),
                              int((r + 1) * SQUARESIZE + SQUARESIZE / 2)),
                             RADIUS)

    # Draw the game pieces (x and o)
    for c in range(COLS):
        for r in range(ROWS):
            if board[r][c] == "x":
                pygame.draw.circle(screen, RED,
                                 (int(c * SQUARESIZE + SQUARESIZE / 2),
                                  int((r + 1) * SQUARESIZE + SQUARESIZE / 2)),
                                 RADIUS)
            elif board[r][c] == "o":
                pygame.draw.circle(screen, YELLOW,
                                 (int(c * SQUARESIZE + SQUARESIZE / 2),
                                  int((r + 1) * SQUARESIZE + SQUARESIZE / 2)),
                                 RADIUS)

    pygame.display.update()

def print_tree(node, depth=0, maximizing_player=None, column=None):
    if node is None:
        return
    indent = '  ' * depth  # Indent to represent tree depth
    move_type = "Max" if maximizing_player else "Min"
    col_info = f" (Chip placed at column = {column})" if column is not None else ""
    print(f'{indent}{move_type}{col_info}: Node at depth = {depth}: Heuristic Value: {node["value"]}')
    for i, child in enumerate(node['children']):
        print_tree(child, depth + 1, not maximizing_player, child.get('column')) 

def create_tree(board, depth, maximizing_player, player, column=None):
    if depth == 0 or game_over(board):
        return {'value': evaluate_heuristic(board, player), 'children': [], 'column': column}

    valid_moves = get_empty_columns(board)
    children = []
    for col in valid_moves:
        child_board = copy.deepcopy(board)
        row = drop_chip(child_board, col, player)  
        child = create_tree(child_board, depth - 1, not maximizing_player, player, column=col)  
        children.append(child)

    return {'value': evaluate_heuristic(board, player), 'children': children, 'column': column}

def get_probabilities(board, col):
    # Initialize probabilities (for all columns)
    probs = [0] * len(board[0])
    probs[col] = 0.6
    
    if col > 0:
        probs[col - 1] = 0.2
    
    if col < len(board[0]) - 1:
        probs[col + 1] = 0.2

    if col == 0:
        probs[col + 1] = 0.4
    
    if col == len(board[0]) - 1:
        probs[col - 1] = 0.4
    
    return probs

def initialize_board():
    board = [["_" for _ in range(COLS)] for _ in range(ROWS)]
    return board


def print_board(board):
    for row in board:
        print(" ".join(row))

def drop_chip(board, col, chip):
    """Drops the chip into the given column and returns the row where it was placed."""
    for row in range(len(board) - 1, -1, -1):  # Start from the bottom row and move upwards
        if board[row][col] == "_":
            board[row][col] = chip
            return row  # Return the row where the chip was placed
    return -1  # Column is full

def isempty(board, col):
    return board[0][col] == "_"

def get_empty_columns(board):
    return [col for col in range(COLS) if isempty(board, col)]

def calculate_score(board, player):
    """Calculates the score for a player by counting sequences of 4 consecutive chips."""
    score = 0

    # Check rows, columns, and diagonals for 4 consecutive chips
    for row in range(len(board)):
        for col in range(len(board[0]) - 3):
            if all(board[row][col + i] == player for i in range(4)): score += 1
    for col in range(len(board[0])):
        for row in range(len(board) - 3):
            if all(board[row + i][col] == player for i in range(4)): score += 1
    for row in range(len(board) - 3):
        for col in range(len(board[0]) - 3):
            if all(board[row + i][col + i] == player for i in range(4)): score += 1
    for row in range(len(board) - 3):
        for col in range(3, len(board[0])):
            if all(board[row + i][col - i] == player for i in range(4)): score += 1

    return score

# def evaluate_heuristic(board, player):
#     """Evaluate the board state using heuristic scoring."""
#     score = 0
#     opponent = "x" if player == "o" else "o"
    
#     # 1. Check connected pieces
#     # Horizontal
#     for row in range(len(board)):
#         for col in range(len(board[0]) - 3):
#             window = [board[row][col + i] for i in range(4)]
#             score += evaluate_window(window, player, opponent)

#     # Vertical
#     for row in range(len(board) - 3):
#         for col in range(len(board[0])):
#             window = [board[row + i][col] for i in range(4)]
#             score += evaluate_window(window, player, opponent)

#     # Diagonal (positive slope)
#     for row in range(len(board) - 3):
#         for col in range(len(board[0]) - 3):
#             window = [board[row + i][col + i] for i in range(4)]
#             score += evaluate_window(window, player, opponent)

#     # Diagonal (negative slope)
#     for row in range(3, len(board)):
#         for col in range(len(board[0]) - 3):
#             window = [board[row - i][col + i] for i in range(4)]
#             score += evaluate_window(window, player, opponent)

#     # Center control preference
#     center_col = len(board[0]) // 2
#     center_array = [board[row][center_col] for row in range(len(board))]
#     score += center_array.count(player) * 3

#     return score

# def evaluate_window(window, player, opponent):
#     """Evaluate a window of 4 positions."""
#     score = 0
    
#     if window.count(player) == 4:
#         score += 100
#     elif window.count(player) == 3 and window.count("_") == 1:
#         score += 5
#     elif window.count(player) == 2 and window.count("_") == 2:
#         score += 2

#     if window.count(opponent) == 3 and window.count("_") == 1:
#         score -= 4

#     return score



def count_candidate_points(board, player):
    """Counts the number of candidate points for the given player (potential 4-in-a-row opportunities)."""
    candidate_count = 0
    opponent = "x" if player == "o" else "o"
    
    # Horizontal candidate points
    for row in range(len(board)):
        for col in range(len(board[0]) - 3):
            window = [board[row][col + i] for i in range(4)]
            candidate_count += evaluate_window_for_candidates(window, player, opponent)

    # Vertical candidate points
    for col in range(len(board[0])):
        for row in range(len(board) - 3):
            window = [board[row + i][col] for i in range(4)]
            candidate_count += evaluate_window_for_candidates(window, player, opponent)
    
    # Diagonal candidate points (positive slope)
    for row in range(len(board) - 3):
        for col in range(len(board[0]) - 3):
            window = [board[row + i][col + i] for i in range(4)]
            candidate_count += evaluate_window_for_candidates(window, player, opponent)
    
    # Diagonal candidate points (negative slope)
    for row in range(3, len(board)):
        for col in range(len(board[0]) - 3):
            window = [board[row - i][col + i] for i in range(4)]
            candidate_count += evaluate_window_for_candidates(window, player, opponent)

    return candidate_count


def evaluate_window_for_candidates(window, player, opponent):
    """Evaluates a window of 4 cells for candidate points."""
    candidate_points = 0
    empty_count = window.count("_")
    
    # If there are any empty spots, check if the player or opponent can potentially complete the sequence
    if empty_count > 0:
        player_count = window.count(player)
        opponent_count = window.count(opponent)
        
        # If the window has the player's pieces and empty spaces, it's a candidate
        if player_count > 0 and opponent_count == 0:
            candidate_points += empty_count  # Each empty space is a candidate for the player
        
        # If the window has the opponent's pieces and empty spaces, it’s a potential threat (defensive candidate)
        if opponent_count > 0 and player_count == 0:
            candidate_points += empty_count  # Count as a candidate to block

    return candidate_points


def count_clusters(board, player):
    """Counts the number of clusters (2 or 3 consecutive pieces) for the given player."""
    cluster_count = 0
    
    # Horizontal clusters
    for row in range(len(board)):
        for col in range(len(board[0]) - 3):
            window = [board[row][col + i] for i in range(4)]
            cluster_count += evaluate_window_for_clusters(window, player)
    
    # Vertical clusters
    for col in range(len(board[0])):
        for row in range(len(board) - 3):
            window = [board[row + i][col] for i in range(4)]
            cluster_count += evaluate_window_for_clusters(window, player)
    
    # Diagonal clusters (positive slope)
    for row in range(len(board) - 3):
        for col in range(len(board[0]) - 3):
            window = [board[row + i][col + i] for i in range(4)]
            cluster_count += evaluate_window_for_clusters(window, player)
    
    # Diagonal clusters (negative slope)
    for row in range(3, len(board)):
        for col in range(len(board[0]) - 3):
            window = [board[row - i][col + i] for i in range(4)]
            cluster_count += evaluate_window_for_clusters(window, player)
    
    return cluster_count

def evaluate_window_for_clusters(window, player):
    """Evaluates a window of 4 cells for clusters (2 or 3 consecutive pieces)."""
    cluster_points = 0
    player_count = window.count(player)
    empty_count = window.count("_")
    
    # If there are 2 or 3 pieces in a row and empty spaces, it's a cluster
    if player_count == 2 and empty_count == 2:
        cluster_points += 1  # A potential 2-in-a-row with 2 empty spots
    elif player_count == 3 and empty_count == 1:
        cluster_points += 1  # A potential 3-in-a-row with 1 empty spot
    
    return cluster_points

def block_opponent_threats(board, opponent):
    """Counts the number of threats from the opponent (3-in-a-row with 1 empty space) that need to be blocked."""
    block_count = 0
    
    # Horizontal threats
    for row in range(len(board)):
        for col in range(len(board[0]) - 3):
            window = [board[row][col + i] for i in range(4)]
            block_count += evaluate_window_for_threats(window, opponent)
    
    # Vertical threats
    for col in range(len(board[0])):
        for row in range(len(board) - 3):
            window = [board[row + i][col] for i in range(4)]
            block_count += evaluate_window_for_threats(window, opponent)
    
    # Diagonal threats (positive slope)
    for row in range(len(board) - 3):
        for col in range(len(board[0]) - 3):
            window = [board[row + i][col + i] for i in range(4)]
            block_count += evaluate_window_for_threats(window, opponent)
    
    # Diagonal threats (negative slope)
    for row in range(3, len(board)):
        for col in range(len(board[0]) - 3):
            window = [board[row - i][col + i] for i in range(4)]
            block_count += evaluate_window_for_threats(window, opponent)
    
    return block_count

def evaluate_window_for_threats(window, opponent):
    """Evaluates a window of 4 cells for threats (3 consecutive opponent pieces and 1 empty space)."""
    threat_points = 0
    opponent_count = window.count(opponent)
    empty_count = window.count("_")
    
    # A threat is 3 opponent pieces with 1 empty space
    if opponent_count == 3 and empty_count == 1:
        threat_points += 1  # This is a potential win for the opponent, so we need to block
    
    return threat_points


def evaluate_heuristic(board, player):
    score = 0
    opponent = "x" if player == "o" else "o"
    
    # 1. Offensive sequences (count 4-in-a-row)
    score += calculate_score(board, player) * 10  # High weight for offensive sequences
    
    # 2. Defensive sequences (block opponent's 4-in-a-row)
    score -= calculate_score(board, opponent) * 10  # Negative weight for opponent's sequences
    
    # 3. Number of candidate points for both players
    score += count_candidate_points(board, player) * 2
    score -= count_candidate_points(board, opponent) * 2
    
    # 4. Control of the center columns (stronger for central control)
    center_col = len(board[0]) // 2
    score += 10 *(board[0][center_col] == player and 5)  # Weight for center column
    score -= board[0][center_col] == opponent and 5  # Weight for blocking center column
    
    # 5. Clustering of pieces (reward clusters of connected pieces)
    score += count_clusters(board, player) * 3
    score -= count_clusters(board, opponent) * 3
    
    # 6. Blocking opponent's potential 4-in-a-row
    score -= block_opponent_threats(board, opponent) * 15  # Higher penalty for blocking
    
    # 7. Early vs Late Game Strategy (adjust weights based on depth)
    if player == "o":  # AI is 'o', so we adjust weights for offensive vs. defensive
        score += 50 if not game_over(board) else 100  # Late game heavy defense
    
    return score

def AlphaBeta_Minimax(board, depth, maximizing_player, player, alpha, beta, column=None):
    global expanded_nodes
    expanded_nodes += 1
    if depth == 0 or game_over(board):
        if game_over(board):
            player_score = calculate_score(board, "x")
            ai_score = calculate_score(board, "o")
            if player == "x":
                value = 100000 if player_score > ai_score else -100000
            else:
                value = 100000 if ai_score > player_score else -100000
        else:
            value = evaluate_heuristic(board, player)
        
        # Ensure a consistent dictionary structure
        return {'value': value, 'children': [], 'column': column}

    valid_columns = get_empty_columns(board)
    children = []
    if maximizing_player:  
        max_eval = float("-inf")
        for col in valid_columns:
            child_board = copy.deepcopy(board)
            row = drop_chip(child_board, col, "o")  # Drop the chip
            child = AlphaBeta_Minimax(child_board, depth - 1, False, player, alpha, beta, column=col)
            board[row][col] = "_"  # Undo the move
            max_eval = max(max_eval, child['value'])
            children.append(child)
            alpha = max(alpha, child['value'])
            if beta <= alpha:
                print(f"Pruning at Depth {depth} (Maximizing Player):")
                print(f"Column considered: {col}, Alpha: {alpha}, Beta: {beta}")
                print(f"board state before pruning:{print_board(board)}")
                print("-----------------------------")
                break  
        return {'value': max_eval, 'children': children, 'column': column}
    else:  
        min_eval = float("inf")
        for col in valid_columns:
            child_board = copy.deepcopy(board)
            row = drop_chip(child_board, col, "x")  # Drop the chip
            child = AlphaBeta_Minimax(child_board, depth - 1, True, player, alpha, beta, column=col)
            board[row][col] = "_"  # Undo the move
            min_eval = min(min_eval, child['value'])
            children.append(child)
            beta = min(beta, child['value'])
            if beta <= alpha:
                print(f"Pruning at Depth {depth} (Minimizing Player):")
                print(f" Column considered: {col}, Alpha: {alpha}, Beta: {beta}")
                print(f" board state before pruning:{print_board(board)}")
                print("-----------------------------")
                break  
        return {'value': min_eval, 'children': children, 'column': column}

def minimax(board, depth, maximizing_player, player, column=None):
    global expanded_nodes
    expanded_nodes += 1
    if depth == 0 or game_over(board):
        if game_over(board):
            player_score = calculate_score(board, "x")
            ai_score = calculate_score(board, "o")
            if player == "x":
                value = 100000 if player_score > ai_score else -100000
            else:
                value = 100000 if ai_score > player_score else -100000
        else:
            value = evaluate_heuristic(board, player)
        
        # Ensure a consistent dictionary structure
        return {'value': value, 'children': [], 'column': column}
    valid_moves = get_empty_columns(board)
    children = []

    if maximizing_player:
        max_eval = float("-inf")
        for col in valid_moves:
            child_board = copy.deepcopy(board)
            row = drop_chip(child_board, col, player)
            child = minimax(child_board, depth - 1, False, player, column=col)
            board[row][col] = "_"  # Undo move
            max_eval = max(max_eval, child['value'])
            children.append(child)
        return {'value': max_eval, 'children': children, 'column': column}
    else:
        min_eval = float("inf")
        opponent = "x" if player == "o" else "o"
        for col in valid_moves:
            child_board = copy.deepcopy(board)
            row = drop_chip(child_board, col, opponent)
            child = minimax(child_board, depth - 1, True, player, column=col)
            board[row][col] = "_"  # Undo move
            min_eval = min(min_eval, child['value'])
            children.append(child)
        return {'value': min_eval, 'children': children, 'column': column}
    
def expected_minimax(board, depth, maximizing_player, player, column=None):
    global expanded_nodes
    expanded_nodes += 1
    if depth == 0 or game_over(board):
        if game_over(board):
            player_score = calculate_score(board, "x")
            ai_score = calculate_score(board, "o")
            if player == "x":
                value = 100000 if player_score > ai_score else -100000
            else:
                value = 100000 if ai_score > player_score else -100000
        else:
            value = evaluate_heuristic(board, player)
        
        # Ensure a consistent dictionary structure
        return {'value': value, 'children': [], 'column': column}

    valid_columns = get_empty_columns(board)
    children = []

    if maximizing_player:  # Computer's move (maximizing player)
        max_eval = float("-inf")
        best_col = None

        for col in valid_columns:
            # Calculate the probability distribution for the chosen column
            probs = get_probabilities(board, col)
            
            child_board = copy.deepcopy(board)
            row = drop_chip(child_board, col, "o")  
            child = expected_minimax(child_board, depth - 1, False, player, column=col)
            board[row][col] = "_" 
            children.append(child)

            expected_value = sum(prob * child['value'] for prob, child in zip(probs, children))
            max_eval = max(max_eval, expected_value)

            print(f"Maximizing Player (Depth {depth}):")
            print(f"Column: {col}, Expected Value: {expected_value}, Max Eval So Far: {max_eval}")
            print(f"Probs: {probs}")
            print("-----------------------------------")

        return {'value': max_eval, 'children': children, 'column': column}

    else:  # Human's move (minimizing player)
        min_eval = float("inf")
        best_col = None
        total_value = 0

        for col in valid_columns:
            probs = get_probabilities(board, col)
            
           
            child_board = copy.deepcopy(board)
            row = drop_chip(child_board, col, "x") 
            child = expected_minimax(child_board, depth - 1, True, player, column=col)
            board[row][col] = "_" 
            children.append(child)

            # Calculate the expected value by weighting each child value with its probability
            expected_value = sum(prob * child['value'] for prob, child in zip(probs, children))
            min_eval = min(min_eval, expected_value)

            print(f"Minimizing Player (Depth {depth}):")
            print(f"Column: {col}, Expected Value: {expected_value}, Min Eval So Far: {min_eval}")
            print(f"Probs: {probs}")
            print("-----------------------------------")

        return {'value': min_eval, 'children': children, 'column': column}

def game_over(board):
    """Checks if the game is over (either player has won or the board is full)."""
    for col in range(len(board[0])):
        if board[0][col] == "_":  # Check if any column is not full
            return False
    return True

def best_move(board, level, algo="minimax", show=False): 
    board_copy = copy.deepcopy(board)
    level = level - 1
    start = time.time()
    global expanded_nodes
    expanded_nodes = 0

    # Create root node representing current board state
    root_node = {
        'value': evaluate_heuristic(board, "o"),
        'children': [],
        'column': None
    }

    # Generate children for all valid moves
    for col in range(len(board[0])):
        if board_copy[0][col] == "_":
            board_after_move = copy.deepcopy(board_copy)
            row = drop_chip(board_after_move, col, "o")

            if algo == "minimax":
                result = minimax(board_after_move, level, False, "o", column=col)
            elif algo == "alpha_beta":
                result = AlphaBeta_Minimax(board_after_move, level, False, "o", float("-inf"), float("inf"), column=col)
            elif algo == "expected_minimax":
                result = expected_minimax(board_after_move, level, False, "o", column=col)
            
            # Create child node and its subtree
            child_tree = create_tree(board_after_move, level, False, "o", column=col)
            root_node['children'].append(child_tree)

            board_copy[row][col] = "_"  # Reset the move

    # Find best move from children
    best_col = -1
    best_value = float("-inf")
    move_scores = []

    for i, child in enumerate(root_node['children']):
        col = child['column']
        move_value = child['value']
        move_scores.append((col, move_value))

        if move_value > best_value:
            best_value = move_value
            best_col = col

    if show:
        print("Complete Game Tree:")
        print_tree(root_node, maximizing_player=True)
        show_tree(root_node)

    end = time.time()
    total_time = end - start
    print(f"Time taken: {total_time:.4f} seconds")
    print(f"Nodes expanded: {expanded_nodes + 1}")
    
    return (best_col, move_scores)


def show_menu(screen):
    # Initialize level_choice with a default value
    level_choice = 3  # Default level set to 3

    # Shift the menu slightly to the left by changing the `x` starting position
    menu_x = WIDTH // 9  # You can change this value to shift the menu further left or right

    screen.fill(WHITE)

    # Title
    title = FONT.render("Connect 4", True, BLACK)
    screen.blit(title, (menu_x, HEIGHT // 4))

    # Algorithm Options
    minimax_text = SMALL_FONT.render("1. Minimax", True, BLACK)
    alpha_beta_text = SMALL_FONT.render("2. Alpha-Beta", True, BLACK)
    expected_minimax_text = SMALL_FONT.render("3. Expected Minimax", True, BLACK)
    screen.blit(minimax_text, (menu_x, HEIGHT // 2 - 40))
    screen.blit(alpha_beta_text, (menu_x, HEIGHT // 2))
    screen.blit(expected_minimax_text, (menu_x, HEIGHT // 2 + 40))

    # Truncation Level (adjustable)
    level_text = SMALL_FONT.render(f"Level: {level_choice}", True, BLACK)
    screen.blit(level_text, (menu_x, HEIGHT // 2 + 80))

    # Instructions for changing level
    level_instructions = SMALL_FONT.render("Use LEFT/RIGHT arrows", True, BLACK)
    screen.blit(level_instructions, (menu_x, HEIGHT // 2 + 120))

    pygame.display.update()

    # Wait for user input (algorithm and truncation level)
    algorithm_choice = None
    while algorithm_choice is None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                # Click on algorithm options
                if HEIGHT // 2 - 40 < y < HEIGHT // 2:
                    algorithm_choice = "minimax"
                elif HEIGHT // 2 < y < HEIGHT // 2 + 40:
                    algorithm_choice = "alpha_beta"
                elif HEIGHT // 2 + 40 < y < HEIGHT // 2 + 80:
                    algorithm_choice = "expected_minimax"

            if event.type == pygame.KEYDOWN:
                # Number key selections for algorithm
                if event.key == pygame.K_1:
                    algorithm_choice = "minimax"
                elif event.key == pygame.K_2:
                    algorithm_choice = "alpha_beta"
                elif event.key == pygame.K_3:
                    algorithm_choice = "expected_minimax"
                
                # Arrow key adjustments for truncation level
                if event.key == pygame.K_RIGHT and level_choice < 10:
                    level_choice += 1
                elif event.key == pygame.K_LEFT and level_choice > 1:
                    level_choice -= 1
                
                # Update the level display after a key press
                level_text = SMALL_FONT.render(f"Level: {level_choice}", True, BLACK)
                screen.fill(WHITE)
                screen.blit(title, (menu_x, HEIGHT // 4))
                screen.blit(minimax_text, (menu_x, HEIGHT // 2 - 40))
                screen.blit(alpha_beta_text, (menu_x, HEIGHT // 2))
                screen.blit(expected_minimax_text, (menu_x, HEIGHT // 2 + 40))
                screen.blit(level_text, (menu_x, HEIGHT // 2 + 80))
                screen.blit(level_instructions, (menu_x, HEIGHT // 2 + 120))
                pygame.display.update()

    return algorithm_choice, level_choice



def start_game_gui():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Connect 4")

    # Show the menu for algorithm selection
    algorithm, level = show_menu(screen)
    print(f"Chosen Algorithm: {algorithm}, Level: {level}")

    board = initialize_board()
    game_over_flag = False
    human = "x"
    computer = "o"
    current_turn = "x"  # Human starts

    # Game loop
    while not game_over_flag:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # If it's the human's turn
            if current_turn == "x":
                if event.type == pygame.MOUSEMOTION:
                    pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, SQUARESIZE))
                    posx = event.pos[0]
                    pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE / 2)), RADIUS)

                pygame.display.update()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    posx = event.pos[0]
                    col = int(posx // SQUARESIZE)

                    if board[0][col] == "_":
                        row = drop_chip(board, col, current_turn)

                        if game_over(board):
                            game_over_flag = True
                        else:
                            current_turn = "o" if current_turn == "x" else "x"  # Switch turn

            # If it's the computer's turn
            elif current_turn == "o":
                pygame.draw.rect(screen, BLACK, (0, 0, WIDTH, SQUARESIZE))
                pygame.display.update()

                best_col, move_scores = best_move(board, level, algo=algorithm, show=True)  # Get best move and scores
                if best_col != -1:
                    row_placed = drop_chip(board, best_col, computer)  # Place computer's chip
                    # print(f"Computer placed chip in column {best_col}, row {row_placed}")
                    print(f"Move scores: {move_scores}")  # Optional: Display scores for all columns

                if game_over(board):
                    game_over_flag = True
                else:
                    current_turn = "x"  # Switch turn back to human

        # Draw the board after every update
        draw_board(board, screen)
    human_score = calculate_score(board, human)
    computer_score = calculate_score(board, computer)

    print("Game Over!")
    print(f"Final Score - Human: {human_score} | Computer: {computer_score}")

    if human_score > computer_score:
        print("Human wins!")
    elif computer_score > human_score:
        print("Computer wins!")
    else:
        print("It's a tie!")

def start_game(board, row, col):
    """Starts the Connect 4 game loop."""
    board = initialize_board(board, row, col)
    computer = "o"
    human = "x"
    turns = row * col  # Total number of turns

    print("Choose AI Algorithm:")
    print("1. Minimax")
    print("2. Alpha-Beta Pruning Minimax")
    print("3. Expected Minimax")
    choice = input("Enter your choice (1/2/3): ").strip()
    if choice == "1":
        algorithm = "minimax"
    elif choice == "2":
        algorithm = "alpha_beta"
    elif choice == "3":
        algorithm = "expected_minimax"
    else:
        print("Invalid choice. Defaulting to Minimax.")
        algorithm = "minimax"

    level = int(input("Enter truncation level: "))

    for turn in range(turns):
        print_board(board)  # Display board before the turn
        print(f"Turn {turn + 1}")

        if turn % 2 == 0:  # Human's turn
            print("Enter your play (column number): ")
            play_col = int(input())  # Get column number from human
            while play_col < 0 or play_col >= col or board[0][play_col] != "_":
                print("Invalid move. Enter a valid column number:")
                play_col = int(input())

            row_placed = drop_chip(board, play_col, human)  # Place human's chip
            print(f"Human placed chip in column {play_col}, row {row_placed}")
        else:  # Computer's turn
            print("Computer is thinking...")
            best_col, move_scores = best_move(board, level, algo=algorithm, show=True)  # Get best move and scores
            if best_col != -1:
                row_placed = drop_chip(board, best_col, computer)  # Place computer's chip
                print(f"Computer placed chip in column {best_col}, row {row_placed}")
                print(f"Move scores: {move_scores}")  # Optional: Display scores for all columns

        # Display board and calculate scores
        human_score = calculate_score(board, human)
        computer_score = calculate_score(board, computer)
        print(f"Human Score: {human_score} | Computer Score: {computer_score}")
        print("--------------------------------------------")

        # Check if game is over
        if game_over(board):
            print("Game over!")
            break

    # Final scores and winner
    human_score = calculate_score(board, human)
    computer_score = calculate_score(board, computer)

    print("Game Over!")
    print(f"Final Score - Human: {human_score} | Computer: {computer_score}")

    if human_score > computer_score:
        print("Human wins!")
    elif computer_score > human_score:
        print("Computer wins!")
    else:
        print("It's a tie!")
    
if __name__ == "__main__":
    # Initialize pygame
    

    # rows = 6  # Typical for Connect 4
    # cols = 7  # 7 columns for Connect 4
    # board: list[list] = []
    expanded_nodes= 0
    start_game_gui()
    # start_game(board, rows, cols)
