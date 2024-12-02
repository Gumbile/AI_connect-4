import copy  # To make a copy of the board

def initialize_board(board, row, col):
    """Initialize the board with empty spaces."""
    for i in range(row):
        board.append(["_"] * col)
    return board

def print_board(board):
    """Print the current board."""
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
    """check if column have empty slots"""
    return board[0][col] == "_"

def get_empty_columns(board):
    """Get valid columns for current board."""
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


def evaluate_sequence(seq, player):
    """Evaluate a seq of 4 for a given player."""
    score = 0
    opponent = "x" if player == "o" else "o"
    
    if seq.count(player) == 4:
        score += 1000  # Winning move
    elif seq.count(player) == 3 and seq.count("_") == 1:
        score += 100  # 3 in a row with one empty
    elif seq.count(player) == 2 and seq.count("_") == 2:
        score += 10  # 2 in a row with two empty

    if seq.count(opponent) == 3 and seq.count("_") == 1:
        score -= 100  # Block opponent's potential win

    return score

def evaluate_board(board, player):
    """Evaluate the board state for a given player."""
    score = 0

    # Center column score
    center_col = cols // 2
    center_array = [board[row][center_col] for row in range(rows)]
    count = center_array.count(player)
    score += count * 5  # Reward control of the center column
    # score for all directions(vertical,horizental,positive and negative diagonal)
    for col in range(cols):
        col_locations = [board[row][col] for row in range(rows)]
        for row in range(rows - 3):  
            seq = col_locations[row:row + 4]
            score += evaluate_sequence(seq, player)
    
    for row in range(rows):
        row_locations = board[row]
        for col in range(cols - 3):  
            seq = row_locations[col:col + 4]
            score += evaluate_sequence(seq, player)

    for row in range(rows - 3):
        for col in range(cols - 3):
            seq = [board[row + i][col + i] for i in range(4)]
            score += evaluate_sequence(seq, player)

    for row in range(3, rows):
        for col in range(cols - 3):
            seq = [board[row - i][col + i] for i in range(4)]
            score += evaluate_sequence(seq, player)

    return score


def AlphaBeta_Minimax(board, depth, maximizing_player, player, alpha,beta):
    """
    Minimax algorithm with Alpha-Beta Pruning.
    """
    valid_columns = get_empty_columns(board)
    if depth == 0 or game_over(board):
        return evaluate_board(board, player)

    if maximizing_player:  # Computer's move (maximizing player)
        max_eval = float("-inf")
        for col in valid_columns:
            row = drop_chip(board, col, "o")  # Drop the chip
            eval = AlphaBeta_Minimax(board, depth - 1, False, "o", alpha, beta)
            board[row][col] = "_"  # Undo the move
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cutoff
        return max_eval
    else:  # Human's move (minimizing player)
        min_eval = float("inf")
        for col in valid_columns:
            row = drop_chip(board, col, "x")  # Drop the chip
            eval = AlphaBeta_Minimax(board, depth - 1, True, "x",  alpha, beta)
            board[row][col] = "_"  # Undo the move
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_eval

def minimax(board, depth, maximizing_player, player):
    """Minimax algorithm with heuristic evaluation, truncated to depth K."""
    if depth == 0 or game_over(board):
        if game_over(board):
            player_score = calculate_score(board, "x")
            ai_score = calculate_score(board, "o")
            if player == "x":
                return 100000 if player_score > ai_score else -100000
            else:
                return 100000 if ai_score > player_score else -100000
        return evaluate_heuristic(board, player)

    valid_moves = [col for col in range(len(board[0])) if board[0][col] == "_"]
    
    if maximizing_player:
        max_eval = float("-inf")
        for col in valid_moves:
            row = drop_chip(board, col, player)
            eval = minimax(board, depth - 1, False, player)
            board[row][col] = "_"  # Undo move
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float("inf")
        opponent = "x" if player == "o" else "o"
        for col in valid_moves:
            row = drop_chip(board, col, opponent)
            eval = minimax(board, depth - 1, True, player)
            board[row][col] = "_"  # Undo move
            min_eval = min(min_eval, eval)
        return min_eval
    
def expected_minimax(board, depth, maximizing_player, player):
    if depth == 0 or game_over(board):
        return evaluate_board(board, player)

    if maximizing_player:
        expected_value = 0
        total_prob = 0
        for col in range(len(board[0])):
            if board[0][col] == "_":
                row = drop_chip(board, col, "o")
                prob = 0.6
                expected_value += prob * expected_minimax(board, depth - 1, False, "o")
                total_prob += prob
                board[row][col] = "_"

                if col > 0:
                    row = drop_chip(board, col - 1, "o")
                    prob_left = 0.2
                    expected_value += prob_left * expected_minimax(board, depth - 1, False, "o")
                    total_prob += prob_left
                    board[row][col - 1] = "_"

                if col < len(board[0]) - 1:
                    row = drop_chip(board, col + 1, "o")
                    prob_right = 0.2
                    expected_value += prob_right * expected_minimax(board, depth - 1, False, "o")
                    total_prob += prob_right
                    board[row][col + 1] = "_"

        return expected_value / total_prob if total_prob > 0 else expected_value

    else:
        min_eval = float("inf")
        for col in range(len(board[0])):
            if board[0][col] == "_":
                row = drop_chip(board, col, "x")
                eval = expected_minimax(board, depth - 1, True, "x")
                board[row][col] = "_"
                min_eval = min(min_eval, eval)
        return min_eval

def game_over(board):
    """Checks if the game is over (either player has won or the board is full)."""
    for col in range(len(board[0])):
        if board[0][col] == "_":  # Check if any column is not full
            return False
    return True

def best_move(board, level , algo="minimax", verbose=False):
    """
    Returns the best column for the computer to drop its chip by using a copy of the board.
    If verbose is True, prints possible moves with their heuristic values and board states.
    """
    best_col = -1
    best_value = float("-inf")

    board_copy = copy.deepcopy(board)  # Make a copy of the board to simulate moves
    move_scores = []  # To store scores for all possible moves

    for col in range(len(board[0])):
        if board_copy[0][col] == "_":  # Check if the column is not full
            row = drop_chip(board_copy, col, "o")  # Drop the chip temporarily

            # Call the selected algorithm
            if algo == "minimax":
                move_value = minimax(board_copy, level, False, "o")
            elif algo == "alpha_beta":
                move_value = AlphaBeta_Minimax(board_copy, level, False, "o", float("-inf"), float("inf"))
            elif algo == "expected_minimax":
                move_value = expected_minimax(board_copy, level , True, "o")
            else:
                move_value = float("-inf")  # Invalid algorithm
            
            move_scores.append((col, move_value, copy.deepcopy(board_copy)))  # Save the column, score, and board state

            board_copy[row][col] = "_"  # Undo the move

            # Keep track of the best move
            if move_value > best_value:
                best_value = move_value
                best_col = col

    # If verbose, print all possible moves, their heuristic values, and board states
    if verbose:
        print("\nAI possible moves:")
        for col, score, state in move_scores:
            print(f"Column: {col}, Heuristic: {score}")
            print_board(state)  # Print the board state for this move
            print("-" * 30)

    return best_col




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
    level = int(input("enter truncation level:"))

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
            col_computer = best_move(board, level , algo=algorithm, verbose=True)  # Use verbose mode to show heuristics and boards
            if col_computer != -1:
                row_placed = drop_chip(board, col_computer, computer)  # Place computer's chip
                print(f"Computer placed chip in column {col_computer}, row {row_placed}")



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

    start_game(board, rows, cols)
