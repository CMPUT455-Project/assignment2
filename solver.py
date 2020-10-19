import board
from board_util import GoBoardUtil, EMPTY, BLACK, WHITE
from transposition import TranspositionTable
import numpy as np

class solver():
    def __init__(self, board, is_perfect):
        self.current_board = board
        self.depth = 6
        self.table = TranspositionTable()
        self.is_perfect = is_perfect
    
    def history_moves(self):
        return self.current_board.move

    def call_alpha_beta(self):
        proof_tree_depth_zero = {}
        result = self.alphabeta(self.current_board, -100, 100, self.depth, proof_tree_depth_zero)
        move = self.find_first_move(proof_tree_depth_zero)
        return result, move

    def alphabeta(self, state, alpha, beta, depth, proof_tree_depth_zero):
        # result = self.table.lookup(state.code(), depth)
        # # check on table
        # if result != None:
        #     # print(str(GoBoardUtil.get_twoD_board(state)))
        #     # print("get " + str(result))
        #     return result
            
        if depth == 0 or state.end_of_game():    
            result = state.staticallyEvaluateForToPlay()    
            return self.store_result(state, result, depth, 0)
        
        #heristic endgame checking
        end, HV = state.heuristic_end_of_game()
        if end:
            return self.store_result(state, HV, depth, 0)      

        #heuristic search a move
        move = state.heuristic_search_move(state.current_player)
        if move:
            moves = [move]
        else:
            if self.is_perfect:
                moves = state.get_empty_points()
            else:
                moves = state.try_to_play_quick_list(state.current_player)

        for m in moves:
            state.play_move_gomoku_auto_change_player(m)
            # print(str(GoBoardUtil.get_twoD_board(state)))
            value = -self.alphabeta(state, -beta, -alpha, depth - 1, proof_tree_depth_zero)
            if value > alpha:
                if (depth == self.depth):
                    proof_tree_depth_zero[value] = m
                alpha = value
            state.undo_move()
            if value >= beta: 
                # self.store_result(state, result, depth, 0)
                return beta   # or value in failsoft (later)
            # self.store_result(state, alpha, depth)
        return alpha

    def find_first_move(self, proof_tree_depth_zero):
        if proof_tree_depth_zero.get(1) != None:
            return proof_tree_depth_zero.get(1)
        else:
            return proof_tree_depth_zero.get(0)

    def store_result(self, state, result, depth, is_what): 
        self.table.store(state.code(), result, depth, is_what)
        # print("store " + str(state.code()) + " " + str(result))
        return result









            
                

        
        

    
    