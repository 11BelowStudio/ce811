# Connect 4 implementation for MCTS and Minimiz.
# University of Essex.
# M. Fairbank November 2021 for course CE811 Game Artificial Intelligence
# 
# Acknowedgements: 
# All of the graphics and some other code for the main game loop and minimax came from https://github.com/KeithGalli/Connect4-Python
# Some of the connect4Board logic and MCTS algorithm came from https://github.com/floriangardin/connect4-mcts 
# Other designs are implemented from the Millington and Funge Game AI textbook chapter on Minimax.
import numpy as np
import random

from connect4Board import Board

class MCTS_Node:

    def __init__(self, board, move, parent):
        self.board = board  # The Board object that this node represents
        self.parent = parent  # Parent Node
        self.move = move  # This is the move used to get to this Node from the parent node (an int from 0 to 6)
        self.games = 0  # How many times this node has had rollout games played through it
        self.wins_for_player_just_moved = 0   # How many rollout games through this node have won (for the player who has just played)
        self.children = None  # A list of child Nodes

    def set_children(self, children):
        self.children = children

    def get_ucb1_score(self):
        if self.games == 0:
            return None
        return (self.wins_for_player_just_moved/self.games) + np.sqrt(2*np.log(self.parent.games)/self.games)

    def select_best_move(self):
        """
        Select best move and advance
        :return:
        """
        if self.children is None:
            return None, None

        winners = [child for child in self.children if (child.is_terminal_node() and child.board.get_victorious_player()!=0)]
        if len(winners) > 0:
            return winners[0], winners[0].move

        child_win_rates = [child.wins_for_player_just_moved/child.games if child.games > 0 else 0 for child in self.children]
        best_child = self.children[np.argmax(child_win_rates)]
        return best_child, best_child.move
     
    def is_terminal_node(self):
        return self.board.is_game_over()

    def get_child_with_move(self, move):
        if self.children is None:
            raise Exception('No existing child')
        for child in self.children:
            if child.move == move:
                return child
        raise Exception('No existing child')

def random_play(board):
    # Play a random game starting at board state. Return winner (1 or 2 (or 0 if draw))
    while True:
        if board.is_game_over():
            return board.get_victorious_player() 
        moves = board.valid_moves()
        assert len(moves) > 0
        selected_move = random.choice(moves)
        board = board.play(selected_move)

def expand_mcts_tree_repeatedly(mcts_tree, tree_expansion_time_ms):
    import time
    start_time = int(round(time.time() * 1000))
    current_time = start_time
    while (current_time - start_time) < tree_expansion_time_ms:
        expand_mcts_tree_once(mcts_tree)
        current_time = int(round(time.time() * 1000))
    
def build_initial_blank_mcts_tree():
    # we've not started building our game tree at all yet.  Make a top-level node for it.
    mcts_tree = MCTS_Node(Board(), move=None,  parent=None)    
    expand_mcts_tree_once(mcts_tree)
    return mcts_tree


def expand_mcts_tree_once(mcts_node):
    # this is the main MCTS algorithm.  On entry, mcts_node should be the top-level node of the tree
    # This function modifies the mcts tree, expanding it by adding new children as appropriate, and 
    # updating the win/loss statistics of all nodes along the path chosen.
    
    # 1. Selection
    while mcts_node.children is not None:
        # we try to find children of this node
        assert (not mcts_node.is_terminal_node())
        # and try to find out which ones are terminal nodes
        terminal_state_children = [child for child in mcts_node.children if child.is_terminal_node()]
        if terminal_state_children != []:
            # if any are terminal nodes, we pick the first terminal child
            mcts_node = terminal_state_children[0]
        else:
            # if none of them are terminal nodes, we pick the one with the highest UCB1 value
            ucts = [child.get_ucb1_score() for child in mcts_node.children]  # Select highest ucb1
            if None in ucts:
                # if any don't have an UCB1 value, we pick a random one of those valueless nodes to look at next.
                mcts_node = random.choice([mcts_node.children[i] for i, x in enumerate(ucts) if x is None])
            else:  # we pick the child with the highest UCB1 value.
                mcts_node = mcts_node.children[np.argmax(ucts)]
    
    # 2. Expansion
    if not mcts_node.is_terminal_node():  # if the selected node is not a terminal node
        moves = mcts_node.board.valid_moves()  # we see what valid moves we can make from there.
        assert len(moves) > 0
        assert mcts_node.children is None
        # we find out what states can be reached from this node.
        successor_states = [(mcts_node.board.play(move), move) for move in moves]
        # and we then convert those states into MCTS tree nodes
        new_children = [MCTS_Node(board, move=move, parent=mcts_node) for (board, move) in successor_states]
        # and we indicate that they're the children of this MCTS node.
        mcts_node.set_children(new_children)

    # 3. Playout (also called "Simulation" or "Random Rollout")
    if mcts_node.is_terminal_node():  # if this node is actually a terminal node
        victorious_player = mcts_node.board.get_victorious_player()  # we just see who won from that.
        assert mcts_node.board.get_victorious_player() in [mcts_node.board.get_player_who_just_moved(), 0]
    else:  # if this node isn't a normal node
        mcts_node = random.choice(mcts_node.children)  # we pick a random child from it
        if mcts_node.is_terminal_node():  # if it's a terminal node, we just see which player won at that node.
            victorious_player = mcts_node.board.get_victorious_player()
            assert mcts_node.board.get_victorious_player() in [mcts_node.board.get_player_who_just_moved(), 0]
        else:  # if it's not a terminal node, we see who wins by playing a completely random game from that node
            victorious_player = random_play(mcts_node.board)

    # 4. Backpropagation
    while mcts_node is not None:  # until we are done back-propagating
        mcts_node.games += 1  # indicate that this node has been in another game.
        if victorious_player != 0 and mcts_node.board.get_player_who_just_moved() == victorious_player:
            # The AND in the line above ensures that, if this was a win for player 1,
            #   as we back propagate up the tree,
            #   only alternating layers receive win+=1.
            # If it was player 2 who won then the alternating would happen too,
            #   but be the other way around
            #   (e.g. hit the odd layers instead of the even ones, or vice-versa).
            # So this means the wins at each layer of the tree refer to the opposite player winning
            #   compared to the layer before.
            # This means that in the "selection" phase above, when it chooses the best uct score at each layer,
            #   it is choosing the best move for the player whose turn it is; rather like minimax.
            #   Clever stuff!
            # Also note we match mcts_node.board.get_player_who_just_moved() == victorious_player,
            # because that player is the one who's just won or lost;
            #   not the person whose turn it is to move next!
            mcts_node.wins_for_player_just_moved += 1
        mcts_node = mcts_node.parent  # and we move on to whatever node is the parent of this node.

    pass
