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

def game_over(board):
    """Checks if the game is over (either player has won or the board is full)."""
    for col in range(len(board[0])):
        if board[0][col] == "_":  # Check if any column is not full
            return False
    return True

def best_move(board, algo="minimax"):
    """Returns the best column for the computer to drop its chip by using a copy of the board."""
    best_col = -1
    best_value = float("-inf")

    board_copy = copy.deepcopy(board)  # Make a copy of the board to simulate moves
    for col in range(len(board[0])):
        if board_copy[0][col] == "_":  # Check if the column is not full
            row = drop_chip(board_copy, col, "o")  # Drop the chip temporarily
            
            # Call the selected algorithm
            if algo == "minimax":
                move_value = minimax(board_copy, 3, False, "o")
            elif algo == "alpha_beta":
                move_value = AlphaBeta_Minimax(board_copy, 3, False, "o",  float("-inf"), float("inf"))
            else:
                print("invalid algorithm")

            board_copy[row][col] = "_"  # Undo the move

            # Keep track of the best move
            if move_value > best_value:
                best_value = move_value
                best_col = col

    return best_col


def start_game(board, row, col):
    """Starts the Connect 4 game loop."""

    board = initialize_board(board, row, col)
    computer = "o"
    human = "x"
    turns = row * col  # Total number of turns
    
    print("Choose AI Algorithm:")
    print("1.Minimax")
    print("2.Alpha-Beta Pruning Minimax")
    choice = input("Enter your choice (1/2): ").strip() 
    if choice == "1" :
         algorithm = "minimax"
    else:
         algorithm = "alpha_beta"     

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
            col_computer = best_move(board, algo=algorithm)  # Use selected algorithm
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
