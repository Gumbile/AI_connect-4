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

def evaluate_board(board, player):
    """Evaluates the board based on several strategic factors."""
    opponent = "x" if player == "o" else "o"
    score = 0
    
    # 1. Center control: Reward for placing chips in the center columns
    center_col = len(board[0]) // 2
    if board[0][center_col] == player:
        score += 10  # Reward center control
    for col in range(center_col - 1, center_col + 2):
        if 0 <= col < len(board[0]) and board[0][col] == player:
            score += 1  # Reward nearby center columns

    # 2. Block opponent's winning move (3-in-a-row with 1 open space)
    for row in range(len(board)):
        for col in range(len(board[0]) - 3):
            if all(board[row][col + i] == opponent for i in range(3)) and board[row][col + 3] == "_":
                score += 50  # High priority for blocking opponent's 3-in-a-row

    # 3. Creating winning opportunities (3-in-a-row with 1 empty space)
    for row in range(len(board)):
        for col in range(len(board[0]) - 3):
            if all(board[row][col + i] == player for i in range(3)) and board[row][col + 3] == "_":
                score += 20  # Reward for creating a potential winning move

    # 4. Opponent's 3-in-a-row threat (penalize if not blocked)
    for row in range(len(board)):
        for col in range(len(board[0]) - 3):
            if all(board[row][col + i] == opponent for i in range(3)) and board[row][col + 3] == "_":
                score -= 30  # Penalize if the AI doesn't block an opponent's threat

    return score

def AlphaBeta_Minimax(board, depth, maximizing_player, player, alpha,beta):
    """
    Minimax algorithm with Alpha-Beta Pruning.
    """
    if depth == 0 or game_over(board):
        return evaluate_board(board, player)

    if maximizing_player:  # Computer's move (maximizing player)
        max_eval = float("-inf")
        for col in range(len(board[0])):
            if board[0][col] == "_":  # Check if the column is not full
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
        for col in range(len(board[0])):
            if board[0][col] == "_":  # Check if the column is not full
                row = drop_chip(board, col, "x")  # Drop the chip
                eval = AlphaBeta_Minimax(board, depth - 1, True, "x",  alpha, beta)
                board[row][col] = "_"  # Undo the move
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cutoff
        return min_eval

def minimax(board, depth, maximizing_player, player):
    """Minimax algorithm with a better heuristic."""
    if depth == 0 or game_over(board):
        return evaluate_board(board, player)

    if maximizing_player:  # Computer's move (maximizing player)
        max_eval = float("-inf")
        for col in range(len(board[0])):
            if board[0][col] == "_":  # Check if the column is not full
                row = drop_chip(board, col, "o")  # Drop the chip
                eval = minimax(board, depth - 1, False, "o")
                board[row][col] = "_"  # Undo the move
                max_eval = max(max_eval, eval)
        return max_eval
    else:  # Human's move (minimizing player)
        min_eval = float("inf")
        for col in range(len(board[0])):
            if board[0][col] == "_":  # Check if the column is not full
                row = drop_chip(board, col, "x")  # Drop the chip
                eval = minimax(board, depth - 1, True, "x")
                board[row][col] = "_"  # Undo the move
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
    row = 6  # Typical for Connect 4
    col = 7  # 7 columns for Connect 4
    board: list[list] = []

    start_game(board, row, col)
