import copy  # To make a copy of the board


def initialize_board(board, row, col):
    for i in range(row):
        board.append([])
        for j in range(col):
            board[i].append("_")
    return board


def print_board(board):
    """Prints the current board."""
    for row in board:
        print(" ".join(row))


def drop_chip(board, col, chip):
    """Drops the chip into the given column and returns the row where it was placed."""
    for row in range(len(board) - 1, -1, -1):  # Start from the bottom row and move upwards
        if board[row][col] == "_":
            board[row][col] = chip
            return row  # Return the row where the chip was placed
    return -1  # Return -1 if the column is full (though this should not happen due to previous checks)



def calculate_score(board, player):
    """Calculates the score for a player by counting sequences of 4 consecutive chips."""
    score = 0

    # Check rows for 4 consecutive chips
    for row in range(len(board)):
        for col in range(len(board[0]) - 3):  # Check up to the 4th last column
            if all(board[row][col + i] == player for i in range(4)):
                score += 1

    # Check columns for 4 consecutive chips
    for col in range(len(board[0])):
        for row in range(len(board) - 3):  # Check up to the 4th last row
            if all(board[row + i][col] == player for i in range(4)):
                score += 1

    # Check diagonals (top-left to bottom-right) for 4 consecutive chips
    for row in range(len(board) - 3):
        for col in range(len(board[0]) - 3):
            if all(board[row + i][col + i] == player for i in range(4)):
                score += 1

    # Check diagonals (top-right to bottom-left) for 4 consecutive chips
    for row in range(len(board) - 3):
        for col in range(3, len(board[0])):
            if all(board[row + i][col - i] == player for i in range(4)):
                score += 1

    return score


def evaluate_board(board, player):
    """Evaluates the board with several factors like winning opportunities, blocking, etc."""
    opponent = "x" if player == "o" else "o"
    score = 0

    # 1. Center control - give a bonus for placing chips in the center
    center_col = len(board[0]) // 2
    if board[0][center_col] == player:
        score += 3  # Highly reward for taking center control

    # 2. Blocking the opponent (block 3-in-a-row with 1 open space)
    for row in range(len(board)):
        for col in range(len(board[0]) - 3):
            # Look for opponent's 3-in-a-row with an open space
            if (
                all(board[row][col + i] == opponent for i in range(3))
                and board[row][col + 3] == "_"
            ):
                score += 10  # Block opponent's move

    # 3. Winning move: 3-in-a-row for the player with 1 empty space
    for row in range(len(board)):
        for col in range(len(board[0]) - 3):
            if (
                all(board[row][col + i] == player for i in range(3))
                and board[row][col + 3] == "_"
            ):
                score += 100  # Winning move for the player

    # 4. Two-in-a-row opportunities
    for row in range(len(board)):
        for col in range(len(board[0]) - 2):
            if (
                all(board[row][col + i] == player for i in range(2))
                and board[row][col + 2] == "_"
            ):
                score += 5  # Setup for a two-in-a-row

    # 5. Threats: Count opponent's 3-in-a-row with an open space
    for row in range(len(board)):
        for col in range(len(board[0]) - 3):
            if (
                all(board[row][col + i] == opponent for i in range(3))
                and board[row][col + 3] == "_"
            ):
                score -= 10  # Opponent's threat

    # 6. Edge considerations: Avoiding placing chips in edges and corners
    for row in range(len(board)):
        for col in range(len(board[0])):
            if col == 0 or col == len(board[0]) - 1:
                if board[row][col] == player:
                    score -= 2  # Penalize edges for the player

    return score


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
    # Check if the board is full
    for col in range(len(board[0])):
        if board[0][col] == "_":
            return False
    return True


def best_move(board):
    """Returns the best column for the computer to drop its chip by using a copy of the board."""
    best_col = -1
    best_value = float("-inf")

    # Make a copy of the board to simulate moves
    board_copy = copy.deepcopy(board)

    # Run the minimax algorithm for each possible move
    for col in range(len(board[0])):
        if board_copy[0][col] == "_":  # Check if the column is not full
            row = drop_chip(board_copy, col, "o")  # Drop the chip temporarily on the copied board
            move_value = minimax(board_copy, 3, False, "o")  # Search to depth 3, assuming the computer's move
            board_copy[row][col] = "_"  # Undo the move on the copied board

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
            col_computer = best_move(board)  # Get the best column for the computer
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

    # Once the board is full, calculate final scores and declare the winner
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
    row = 7  # For Connect 4, typically 6 rows
    col = 7  # 7 columns for Connect 4
    board: list[list] = []

    # Start the game directly without threading
    start_game(board, row, col)
    # board = initialize_board(board, row, col)
    # board_copy = copy.deepcopy(board)
    # board_copy[0][0] = "b"
    # print_board(board_copy)
    # print("--------------------------------------------------------------------------------")
    # print_board(board)
