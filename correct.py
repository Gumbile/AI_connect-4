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

def game_over(board):
    """Checks if the game is over (either player has won or the board is full)."""
    # if calculate_score(board, "o") > 0 or calculate_score(board, "x") > 0:
    #     return True  # Someone has won
    # # Check if the board is full
    for col in range(len(board[0])):
        if board[0][col] == "_":
            return False
    return True


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

def minimax(board, depth, maximizing_player, player, K):
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
            eval = minimax(board, depth - 1, False, player, K)
            board[row][col] = "_"  # Undo move
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float("inf")
        opponent = "x" if player == "o" else "o"
        for col in valid_moves:
            row = drop_chip(board, col, opponent)
            eval = minimax(board, depth - 1, True, player, K)
            board[row][col] = "_"  # Undo move
            min_eval = min(min_eval, eval)
        return min_eval
def best_move(board, K):
    """Returns the best column for the computer to drop its chip using Minimax."""
    best_col = -1
    best_value = float("-inf")

    # Make a copy of the board to simulate moves
    board_copy = copy.deepcopy(board)

    # Run the minimax algorithm for each possible move
    for col in range(len(board[0])):
        if board_copy[0][col] == "_":  # Check if the column is not full
            row = drop_chip(board_copy, col, "o")  # Drop the chip temporarily
            move_value = minimax(board_copy, K, False, "o", K)  # Minimax search with depth K
            board_copy[row][col] = "_"  # Undo the move on the copied board
            # print(str(col) + "  " + "score :" +str(move_value) )
            # Keep track of the best move
            if move_value > best_value:
                best_value = move_value
                best_col = col

    return best_col

def start_game(board, row, col, K):
    """Starts the Connect 4 game loop with depth K for the AI."""
    board = initialize_board(board, row, col)
    computer = "o"
    human = "x"
    turns = row * col  # Total number of turns
    for turn in range(turns):
        print_board(board)  # Display board before the turn
        print(f"Turn {turn + 1}")

        if turn % 2 == 1:  # Human's turn
            print("Enter your play (column number): ")
            play_col = int(input())  # Get column number from human
            while play_col < 0 or play_col >= col or board[0][play_col] != "_":
                print("Invalid move. Enter a valid column number:")
                play_col = int(input())

            row_placed = drop_chip(board, play_col, human)  # Place human's chip
            print(f"Human placed chip in column {play_col}, row {row_placed}")
        else:  # Computer's turn
            print("Computer is thinking...")
            col_computer = best_move(board, K)  # Get the best column for the computer
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
    row = 6  # For Connect 4, typically 6 rows
    col = 7  # 7 columns for Connect 4
    board: list[list] = []

    # Start the game directly without threading
    start_game(board, row, col,3)
    # board = initialize_board(board, row, col)
    # board_copy = copy.deepcopy(board)
    # board_copy[0][0] = "b"
    # print_board(board_copy)
    # print("--------------------------------------------------------------------------------")
    # print_board(board)
