import numpy as np
import random
import time
import sys
import os 
from BaseAI import BaseAI
from Grid import Grid

# TO BE IMPLEMENTED
# 
class PlayerAI(BaseAI):

    def __init__(self) -> None:
        # You may choose to add attributes to your player - up to you!
        super().__init__()
        self.pos = None
        self.player_num = None
    
    def getPosition(self):
        return self.pos

    def setPosition(self, new_position):
        self.pos = new_position 

    def getPlayerNum(self):
        return self.player_num

    def setPlayerNum(self, num):
        self.player_num = num

    def getMove(self, grid: Grid) -> tuple:
        """ 
        YOUR CODE GOES HERE

        The function should return a tuple of (x,y) coordinates to which the player moves.

        It should be the result of the ExpectiMinimax algorithm, maximizing over the Opponent's *Trap* actions, 
        taking into account the probabilities of them landing in the positions you believe they'd throw to.

        Note that you are not required to account for the probabilities of it landing in a different cell.

        You may adjust the input variables as you wish (though it is not necessary). Output has to be (x,y) coordinates.
        
        """

        alpha, beta = float('-inf'), float('inf')
        maximizing_player = True
        depth = 3
        optimal_result = self.mini_max_get_move(grid, depth, alpha, beta, maximizing_player)
        optimal_state = optimal_result[0]

        #if not optimal_state:
        #    return random.choice(grid.getAvailableCells())

        rows, cols = optimal_state.dim, optimal_state.dim
        for i in range(rows):
            for j in range(cols):
                if optimal_state.map[i][j] == self.player_num:
                    return (i, j)
        return None

    def getTrap(self, grid : Grid) -> tuple:
        """ 
        YOUR CODE GOES HERE

        The function should return a tuple of (x,y) coordinates to which the player *WANTS* to throw the trap.
        
        It should be the result of the ExpectiMinimax algorithm, maximizing over the Opponent's *Move* actions, 
        taking into account the probabilities of it landing in the positions you want. 
        
        Note that you are not required to account for the probabilities of it landing in a different cell.

        You may adjust the input variables as you wish (though it is not necessary). Output has to be (x,y) coordinates.
        
        """
        alpha, beta = float('-inf'), float('inf')
        maximizing_player = True
        depth = 3
        optimal_result = self.mini_max_get_trap(grid, depth, alpha, beta, maximizing_player)
        optimal_state = optimal_result[0]

        if not optimal_state:
            return random.choice(grid.getAvailableCells())

        rows, cols = optimal_state.dim, optimal_state.dim
        for i in range(rows):
            for j in range(cols):
                if grid.map[i][j] != optimal_state.map[i][j]:
                    return (i, j)
        return None

    def mini_max_get_move(self, grid, depth, alpha, beta, maximizing_player):
        player_pos = grid.find(self.player_num)
        player_valid_neighbors = grid.get_neighbors(player_pos, True)

        if not player_valid_neighbors:
            eval_result = (grid, self.OCLS(grid, self.player_num))
            #print("no move: score: ",eval_result[1])
            return eval_result

        if depth == 0:
            eval_result = (grid, self.OCLS(grid, self.player_num))
            #print("0: score: ",eval_result[1])
            return eval_result

        if maximizing_player:
            max_eval = (None, float('-inf'))
            for child in player_valid_neighbors:
                grid_copy = grid.clone()
                grid_copy.move(child, self.player_num)

                # Recursion
                eval_result = self.mini_max_get_move(grid_copy, depth - 1, alpha, beta, False)
                eval_grid = eval_result[0]
                eval_score = eval_result[1]
                if max_eval[1] < eval_score:
                    max_eval = eval_result

                #Alpha Beta Pruning
                if alpha < eval_score:
                    alpha = eval_score
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = (None, float('inf'))

            for child in player_valid_neighbors:
                grid_copy = grid.clone()
                grid_copy.move(child, self.player_num)

                # Recursion
                eval_result = self.mini_max_get_move(grid_copy, depth - 1, alpha, beta, True)
                eval_grid = eval_result[0]
                eval_score = eval_result[1]
                if eval_score < min_eval[1]:
                    min_eval = eval_result

                #Alpha Beta Pruning
                if eval_score < beta:
                    beta = eval_score
                if beta <= alpha:
                    break

            return min_eval

    def mini_max_get_trap(self, grid, depth, alpha, beta, maximizing_player):
        opponent_pos = grid.find(3 - self.player_num)
        opponent_valid_neighbors = grid.get_neighbors(opponent_pos, True)

        if not opponent_valid_neighbors:
            #eval_result = (grid, self.IS(grid, 3 - self.player_num)) ------->IS
            eval_result = (grid, self.OCLS(grid, 3 - self.player_num))
            return eval_result

        # Base case, evaluate leaf nodes.
        if depth == 0:
            #print("DEP = 0")
            #grid.print_grid()
            #print()
            #eval_result = (grid, self.IS(grid, 3 - self.player_num)) -------->IS
            eval_result = (grid, self.OCLS(grid, 3 - self.player_num))
            #print("Score: ", eval_result[1])
            return eval_result

        # find available positions for throwing traps
        opponent_pos = grid.find(3 - self.player_num)
        opponent_valid_neighbors = grid.get_neighbors(opponent_pos, True)
        #print("------DEBUG")
        #print("Current Depth: ", depth)
        #print("Current Valid Trap: ", opponent_valid_neighbors)
        #grid.print_grid()
        #print("------DEBUG DONE -------")

        #print(opponent_valid_neighbors)
        if maximizing_player:
            max_eval = (None, float('-inf'))

            for child in opponent_valid_neighbors:
                grid_copy = grid.clone()
                grid_copy.trap(child)

                # Recursion
                eval_result = self.mini_max_get_trap(grid_copy, depth - 1, alpha, beta, False)
                eval_grid = eval_result[0]
                eval_score = eval_result[1]
                if max_eval[1] < eval_score:
                    max_eval = eval_result

                #Alpha Beta Pruning
                if alpha < eval_score:
                    alpha = eval_score
                if beta <= alpha:
                    break
            return max_eval

        else:
            min_eval = (None, float('inf'))

            for child in opponent_valid_neighbors:
                grid_copy = grid.clone()
                grid_copy.trap(child)

                # Recursion
                eval_result = self.mini_max_get_trap(grid_copy, depth - 1, alpha, beta, True)
                eval_grid = eval_result[0]
                eval_score = eval_result[1]
                if eval_score < min_eval[1]:
                    min_eval = eval_result

                #Alpha Beta Pruning
                if eval_score < beta:
                    beta = eval_score
                if beta <= alpha:
                    break

            return min_eval

    def IS(self, grid, player_num):

        # find all available moves by Player
        player_moves = grid.get_neighbors(grid.find(player_num), only_available = True)

        # find all available moves by Opponent
        opp_moves = grid.get_neighbors(grid.find(3 - player_num), only_available = True)

        return len(player_moves) - len(opp_moves)

    '''
    The difference between the Player's sum of possible moves looking
    one step ahead and the Opponent's sum of possible moves looking one step ahead

    def OCLS(self, grid, player_num):
        sum = 0
        available_moves = grid.get_neighbors(grid.find(player_num), only_available = True)
        for potential_moves in available_moves:
            ava_for_potential_moves = grid.get_neighbors(potential_moves, only_available = True)
            sum += len(ava_for_potential_moves)

        return sum
    '''
    def OCLS(self, grid, player_num):
        sum = 0
        available_moves = grid.get_neighbors(grid.find(player_num), only_available = True)
        for potential_moves in available_moves:
            ava_for_potential_moves = grid.get_neighbors(potential_moves, only_available = True)
            sum += len(ava_for_potential_moves)

        op_sum = 0
        op_available_moves = grid.get_neighbors(grid.find(3 - player_num), only_available = True)
        for potential_moves in op_available_moves:
            ava_for_potential_moves = grid.get_neighbors(potential_moves, only_available = True)
            op_sum += len(ava_for_potential_moves)

        return sum - op_sum

