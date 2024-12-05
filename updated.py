import copy
import time

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

def initialize_board(board, row, col):
    for i in range(row):
        board.append(["_"] * col)
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
    return [col for col in range(cols) if isempty(board, col)]

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

def evaluate_heuristic(board, player):
    """Evaluate the board state using heuristic scoring."""
    score = 0
    opponent = "x" if player == "o" else "o"
    
    # 1. Check connected pieces
    # Horizontal
    for row in range(len(board)):
        for col in range(len(board[0]) - 3):
            window = [board[row][col + i] for i in range(4)]
            score += evaluate_window(window, player, opponent)

    # Vertical
    for row in range(len(board) - 3):
        for col in range(len(board[0])):
            window = [board[row + i][col] for i in range(4)]
            score += evaluate_window(window, player, opponent)

    # Diagonal (positive slope)
    for row in range(len(board) - 3):
        for col in range(len(board[0]) - 3):
            window = [board[row + i][col + i] for i in range(4)]
            score += evaluate_window(window, player, opponent)

    # Diagonal (negative slope)
    for row in range(3, len(board)):
        for col in range(len(board[0]) - 3):
            window = [board[row - i][col + i] for i in range(4)]
            score += evaluate_window(window, player, opponent)

    # Center control preference
    center_col = len(board[0]) // 2
    center_array = [board[row][center_col] for row in range(len(board))]
    score += center_array.count(player) * 3

    return score

def evaluate_window(window, player, opponent):
    """Evaluate a window of 4 positions."""
    score = 0
    
    if window.count(player) == 4:
        score += 100
    elif window.count(player) == 3 and window.count("_") == 1:
        score += 5
    elif window.count(player) == 2 and window.count("_") == 2:
        score += 2

    if window.count(opponent) == 3 and window.count("_") == 1:
        score -= 4

    return score

def AlphaBeta_Minimax(board, depth, maximizing_player, player, alpha, beta, column=None):
    global expanded_nodes
    expanded_nodes += 1
    if depth == 0 or game_over(board):
        if game_over(board):
            player_score = calculate_score(board, "x")
            ai_score = calculate_score(board, "o")
            if player == "x":
                return 100000 if player_score > ai_score else -100000
            else:
                return 100000 if ai_score > player_score else -100000
        return {'value': evaluate_heuristic(board, player), 'children': [], 'column': column}

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
                return 100000 if player_score > ai_score else -100000
            else:
                return 100000 if ai_score > player_score else -100000
        return {'value': evaluate_heuristic(board, player), 'children': [], 'column': column}
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
                return 100000 if player_score > ai_score else -100000
            else:
                return 100000 if ai_score > player_score else -100000
        return {'value': evaluate_heuristic(board, player), 'children': [], 'column': column}

    valid_columns = get_empty_columns(board)
    children = []

    if maximizing_player:  # Computer's move (maximizing player)
        max_eval = float("-inf")
        best_col = None

        for col in valid_columns:
            # Calculate the probability distribution for the chosen column
            probs = get_probabilities(board, col)
            
            # Simulate the maximizing player's move
            child_board = copy.deepcopy(board)
            row = drop_chip(child_board, col, "o")  # Drop the chip
            child = expected_minimax(child_board, depth - 1, False, player, column=col)
            board[row][col] = "_"  # Undo the move
            children.append(child)

            # Calculate the expected value by weighting each child value with its probability
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
            
            # Simulate the minimizing player's move
            child_board = copy.deepcopy(board)
            row = drop_chip(child_board, col, "x")  # Drop the chip
            child = expected_minimax(child_board, depth - 1, True, player, column=col)
            board[row][col] = "_"  # Undo the move
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

    best_col = -1
    best_value = float("-inf")
    global expanded_nodes
    expanded_nodes = 0 
    start = time.time()
    board_copy = copy.deepcopy(board)  
    move_scores = []  

    for col in range(len(board[0])):
        if board_copy[0][col] == "_":  
            row = drop_chip(board_copy, col, "o")  

            if algo == "minimax":
                result = minimax(board_copy, level, False, "o", column=col)
                move_value = result['value']  
            elif algo == "alpha_beta":
                result = AlphaBeta_Minimax(board_copy, level, False, "o", float("-inf"), float("inf"), column=col)
                move_value = result['value']  
            elif algo == "expected_minimax":
                result = expected_minimax(board_copy, level , False , "o",column=col)
                move_value = result['value'] 
            else:
                move_value = float("-inf")  

            game_tree = create_tree(board_copy, level, False, "o", column=col)
            if show:
                print(f"Game Tree for column: {col}")
                print_tree(game_tree, maximizing_player=True, column=col)
                print("-" * 30)
            
            move_scores.append((col, move_value))  
            board_copy[row][col] = "_"  

            if move_value > best_value:
                best_value = move_value
                best_col = col
    end = time.time()
    total_time = end - start
    print(f"Time taken: {total_time:.4f} seconds")
    print(f"Nodes expanded: {expanded_nodes}")
    return (best_col, move_scores)

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
    rows = 6  # Typical for Connect 4
    cols = 7  # 7 columns for Connect 4
    board: list[list] = []
    expanded_nodes= 0
    start_game(board, rows, cols)
