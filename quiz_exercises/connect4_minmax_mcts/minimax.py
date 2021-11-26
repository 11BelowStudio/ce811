# Connect 4 implementation for MCTS and Minimiz.
# University of Essex.
# M. Fairbank November 2021 for course CE811 Game Artificial Intelligence
# 
# Acknowedgements: 
# All of the graphics and some other code for the main game loop and minimax came from https://github.com/KeithGalli/Connect4-Python
# Some of the connect4Board logic and MCTS algorithm came from https://github.com/floriangardin/connect4-mcts 
# Other designs are implemented from the Millington and Funge Game AI textbook chapter on Minimax.
import random
from connect4Board import Board
import math
import numpy as np

from typing import Tuple, Union


def segment_evaluator(segment: np.ndarray) -> int:
    segment_sum = segment.sum()
    if segment_sum < 0:  # if any enemy pieces are there
        if segment_sum % 5 == 0:  # if it's just enemy pieces
            return segment_sum // 5  # return count of enemy pieces
        else:
            return 0  # if both players have pieces, this has a value of 0.
    return segment_sum  # return count of player pieces if no enemy pieces are present.


def static_evaluator(board, piece) -> int:
    # static evaluator function, to estimate how good the board is from the point of view of player "piece".
    # On entry, piece will be either 1 or 2.

    enemy = 1
    if piece == 1:
        enemy = 2

    grid: np.ndarray = board.grid.copy()
    # Note grid is a numpy integer array with 6 rows and 7 columns, so its "shape" is [6,7]
    # Each element of grid is either a 0 or a 1 or a 2 (for empty / player1 / player 2, respectively)
    # you can access elements like grid[2,3], which will return an integer, or grid[2,2:6], which will return a numpy array of shape [4].

    grid: np.ndarray = np.where(grid == enemy, -5, grid)
    grid: np.ndarray = np.where(grid == piece, 1, grid)

    score = 0

    for i in range(0, 6):  # for each row
        for j in range(0, 4):  # for each possible start position of a 4 in that row

            score += segment_evaluator(grid[i, j:j+4])

    for i in range(0, 7):  # for each column
        for j in range(0, 3):  # for each possible start position of a 4 in that column
            score += segment_evaluator(grid[j:j+4, i])

    for i in range(0, 4):  # for the first half of the columns (where we can start a diagonal from)
        for j in range(0, 3):  # for each possible start position of a 4 in that column
            score += segment_evaluator(np.array([grid[j + k][i + k] for k in range(4)]))
            score += segment_evaluator(np.array([grid[j + k][i + 3 - k] for k in range(4)]))

    return score



ab_pruning: bool = True

def minimax(board, current_depth, max_depth, player, alpha: int=-100000000, beta: int=100000000) -> Tuple[Union[int, None], int]:
    # This function needs to return (best_move,value), where value is the value of the board according to player=player
    # See Millington and Funge, "Game Artificial Intelligence" texbook, 2nd edition, chapter 8.2 for pseudocode

    print("Alpha: {}, Beta: {}".format(alpha, beta))

    if player==board.get_player_turn():
        maximiser = True # This means we are at the "maximiser" level of the game tree
    else:
        maximiser = False  # This means we are at the "minimiser" level of the game tree
    
    # deal with easy case (the end-point of the recursion)...
    is_terminal = board.is_game_over()
    if current_depth == max_depth or is_terminal:
        if is_terminal:
            opponent=3-player
            if board.get_victorious_player()==player:
                return None, +100000000
            elif board.get_victorious_player()==opponent:
                return None, -100000000
            else: # Game is over, no more valid moves
                return None, 0
        else: # Depth is zero
            return None, static_evaluator(board, player)


    # Use recursion to move down through the minimax levels and calculate the best_move and board value....            
    valid_moves = board.valid_moves()
    if maximiser:
        if ab_pruning:
            best_val: int = -100000001
            best_move = valid_moves[0]
            for m in valid_moves:
                val = minimax(
                    board.play(m), current_depth + 1, max_depth, player, alpha, beta
                )[1]
                best_val = max(best_val, val)
                if best_val == val:
                    best_move = m
                    alpha = max(alpha, best_val)
                    if beta <= alpha:
                        break

            return best_move, best_val
        else:

            best_move, mm = max(
                ((m, minimax(
                    board.play(m), current_depth + 1, max_depth, player, alpha, beta
                )) for m in valid_moves), key=lambda bm: bm[1][1]
            )
            return best_move, mm[1]

    else:  # Minimising player
        if ab_pruning:
            best_val: int = +100000001
            best_move = valid_moves[0]
            for m in valid_moves:
                val = minimax(
                    board.play(m), current_depth + 1, max_depth, player, alpha, beta
                )[1]
                best_val = min(best_val, val)
                if best_val == val:
                    best_move = m
                    beta = min(beta, best_val)
                    if beta <= alpha:
                        break

            return best_move, best_val
        else:

            best_move, mm = min(
                ((m, minimax(
                    board.play(m), current_depth + 1, max_depth, player, alpha, beta
                )) for m in valid_moves), key=lambda bm: bm[1][1]
            )
            return best_move, mm[1]

