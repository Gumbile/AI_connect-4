import tkinter as tk
from random import randint

# Constants
ROW_COUNT = 6
COL_COUNT = 7
HUMAN_CHIP = "red"
COMPUTER_CHIP = "yellow"
EMPTY = "_"
CIRCLE_RADIUS = 30  # Radius for the circular chips
FALL_SPEED = 10  # Speed of the falling animation in ms

class Connect4Game:
    def __init__(self, root):
        self.root = root
        self.root.title("Connect 4")
        self.root.configure(bg="lightblue")

        self.board = [[EMPTY for _ in range(COL_COUNT)] for _ in range(ROW_COUNT)]
        self.current_turn = HUMAN_CHIP
        self.game_over = False  # To track the game status

        # Create the canvas to draw the board and chips
        self.canvas = tk.Canvas(self.root, width=COL_COUNT * (CIRCLE_RADIUS * 2 + 10), 
                                height=ROW_COUNT * (CIRCLE_RADIUS * 2 + 10), bg="blue")
        self.canvas.pack(padx=10, pady=10)

        # Create the score label
        self.score_label = tk.Label(self.root, text="Human: 0 | Computer: 0", font=("Arial", 14), bg="lightblue")
        self.score_label.pack()

        # Create the column buttons under the board for human to click
        self.create_column_buttons()

        # Create a reset button under the column buttons
        self.reset_button = tk.Button(self.root, text="Reset Game", font=("Arial", 14), bg="lightgreen", 
                                      command=self.reset_game)
        self.reset_button.pack(pady=10)

        # Draw the empty board (with circular placeholders for chips)
        self.draw_board()

    def create_column_buttons(self):
        """Creates the buttons below the board for human to click on."""
        self.buttons_frame = tk.Frame(self.root, bg="lightblue")
        self.buttons_frame.pack()

        self.column_buttons = []
        for col in range(COL_COUNT):
            button = tk.Button(self.buttons_frame, text=f"Col {col+1}", font=("Arial", 10), 
                               command=lambda col=col: self.human_turn(col))
            button.grid(row=0, column=col, padx=5, pady=5, ipadx=10)

    def draw_board(self):
        """Draws the game board on the canvas."""
        for row in range(ROW_COUNT):
            for col in range(COL_COUNT):
                x = col * (CIRCLE_RADIUS * 2 + 10) + CIRCLE_RADIUS + 5
                y = row * (CIRCLE_RADIUS * 2 + 10) + CIRCLE_RADIUS + 5
                self.canvas.create_oval(x - CIRCLE_RADIUS, y - CIRCLE_RADIUS,
                                        x + CIRCLE_RADIUS, y + CIRCLE_RADIUS, 
                                        fill="white", outline="black", width=2)

    def human_turn(self, col):
        """Handles the human player's turn."""
        if self.board[0][col] == EMPTY and not self.game_over:  # Only if the column is not full
            row = self.drop_chip(col, HUMAN_CHIP)
            if row != -1:
                self.animate_chip_drop(col, row, HUMAN_CHIP)  # Animate chip falling
                self.update_board()
                self.check_scores()  # Check and update scores after the human's move
                if not self.game_over:
                    self.current_turn = COMPUTER_CHIP  # Switch turn to computer
                    self.root.after(500, self.computer_turn)  # Slight delay before computer's turn

    def computer_turn(self):
        """Handles the computer's turn (random move)."""
        if self.current_turn == COMPUTER_CHIP and not self.game_over:
            available_cols = [col for col in range(COL_COUNT) if self.board[0][col] == EMPTY]
            if available_cols:
                col = randint(0, len(available_cols) - 1)
                row = self.drop_chip(available_cols[col], COMPUTER_CHIP)
                if row != -1:
                    self.animate_chip_drop(available_cols[col], row, COMPUTER_CHIP)  # Animate chip falling
                    self.update_board()
                    self.check_scores()  # Check and update scores after the computer's move
                    if not self.game_over:
                        self.current_turn = HUMAN_CHIP  # Switch turn back to human

    def drop_chip(self, col, chip):
        """Drops the chip in the given column and returns the row where it was placed."""
        for row in range(ROW_COUNT - 1, -1, -1):
            if self.board[row][col] == EMPTY:
                self.board[row][col] = chip
                return row
        return -1  # If the column is full, return -1

    def animate_chip_drop(self, col, row, chip):
        """Animates the chip dropping from the top to its final position."""
        x = col * (CIRCLE_RADIUS * 2 + 10) + CIRCLE_RADIUS + 5
        y_start = 0  # Start from the top of the canvas
        y_end = row * (CIRCLE_RADIUS * 2 + 10) + CIRCLE_RADIUS + 5  # Target y position

        # Create the chip as a circle
        chip_color = HUMAN_CHIP if chip == HUMAN_CHIP else COMPUTER_CHIP
        chip_oval = self.canvas.create_oval(x - CIRCLE_RADIUS, y_start - CIRCLE_RADIUS, 
                                            x + CIRCLE_RADIUS, y_start + CIRCLE_RADIUS, 
                                            fill=chip_color, outline="black", width=2)

        # Animate the chip falling
        def fall_animation():
            nonlocal y_start
            if y_start < y_end:
                y_start += 5  # Increase the fall speed here
                self.canvas.coords(chip_oval, x - CIRCLE_RADIUS, y_start - CIRCLE_RADIUS, 
                                   x + CIRCLE_RADIUS, y_start + CIRCLE_RADIUS)
                self.root.after(FALL_SPEED, fall_animation)
            else:
                self.canvas.coords(chip_oval, x - CIRCLE_RADIUS, y_end - CIRCLE_RADIUS, 
                                   x + CIRCLE_RADIUS, y_end + CIRCLE_RADIUS)
                self.board[row][col] = chip  # Finalize chip placement in the board

        fall_animation()

    def update_board(self):
        """Updates the canvas to reflect the current state of the game."""
        self.canvas.delete("chip")  # Remove all chips before redrawing
        for row in range(ROW_COUNT):
            for col in range(COL_COUNT):
                if self.board[row][col] == HUMAN_CHIP:
                    x = col * (CIRCLE_RADIUS * 2 + 10) + CIRCLE_RADIUS + 5
                    y = row * (CIRCLE_RADIUS * 2 + 10) + CIRCLE_RADIUS + 5
                    self.canvas.create_oval(x - CIRCLE_RADIUS, y - CIRCLE_RADIUS,
                                            x + CIRCLE_RADIUS, y + CIRCLE_RADIUS, 
                                            fill=HUMAN_CHIP, outline="black", width=2, tags="chip")
                elif self.board[row][col] == COMPUTER_CHIP:
                    x = col * (CIRCLE_RADIUS * 2 + 10) + CIRCLE_RADIUS + 5
                    y = row * (CIRCLE_RADIUS * 2 + 10) + CIRCLE_RADIUS + 5
                    self.canvas.create_oval(x - CIRCLE_RADIUS, y - CIRCLE_RADIUS,
                                            x + CIRCLE_RADIUS, y + CIRCLE_RADIUS, 
                                            fill=COMPUTER_CHIP, outline="black", width=2, tags="chip")

    def calculate_score(self, player):
        """Calculates the score for a player by counting sequences of 4 consecutive chips."""
        score = 0

        # Check rows for 4 consecutive chips
        for row in range(ROW_COUNT):
            for col in range(COL_COUNT - 3):  # Check up to the 4th last column
                if all(self.board[row][col + i] == player for i in range(4)):
                    score += 1

        # Check columns for 4 consecutive chips
        for col in range(COL_COUNT):
            for row in range(ROW_COUNT - 3):  # Check up to the 4th last row
                if all(self.board[row + i][col] == player for i in range(4)):
                    score += 1

        # Check diagonals (top-left to bottom-right) for 4 consecutive chips
        for row in range(ROW_COUNT - 3):
            for col in range(COL_COUNT - 3):
                if all(self.board[row + i][col + i] == player for i in range(4)):
                    score += 1

        # Check diagonals (top-right to bottom-left) for 4 consecutive chips
        for row in range(ROW_COUNT - 3):
            for col in range(3, COL_COUNT):
                if all(self.board[row + i][col - i] == player for i in range(4)):
                    score += 1

        return score

    def check_scores(self):
        """Check scores after each move and update the score label."""
        human_score = self.calculate_score(HUMAN_CHIP)
        computer_score = self.calculate_score(COMPUTER_CHIP)
        self.score_label.config(text=f"Human: {human_score} | Computer: {computer_score}")
        
        # Check if the board is full
        if all(self.board[0][col] != EMPTY for col in range(COL_COUNT)):
            self.game_over = True
            if human_score > computer_score:
                self.score_label.config(text="Human wins!")
            elif computer_score > human_score:
                self.score_label.config(text="Computer wins!")
            else:
                self.score_label.config(text="It's a tie!")

    def reset_game(self):
        """Resets the game to its initial state."""
        self.board = [[EMPTY for _ in range(COL_COUNT)] for _ in range(ROW_COUNT)]
        self.current_turn = HUMAN_CHIP
        self.game_over = False
        self.canvas.delete("chip")
        self.score_label.config(text="Human: 0 | Computer: 0")
        self.draw_board()


def main():
    root = tk.Tk()
    game = Connect4Game(root)
    root.mainloop()

if __name__ == "__main__":
    main()
