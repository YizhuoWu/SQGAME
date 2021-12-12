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
        # find all available moves
        available_moves = grid.get_neighbors(self.pos, only_available = True)

        states = [grid.clone().move(mv, self.player_num) for mv in available_moves]

        # find move with best IS score
        am_scores = np.array([self.AM(state, self.player_num) for state in states])

        new_pos = available_moves[np.argmax(am_scores)]

        return new_pos

    def getTrap(self, grid : Grid) -> tuple:
        """ 
        YOUR CODE GOES HERE

        The function should return a tuple of (x,y) coordinates to which the player *WANTS* to throw the trap.
        
        It should be the result of the ExpectiMinimax algorithm, maximizing over the Opponent's *Move* actions, 
        taking into account the probabilities of it landing in the positions you want. 
        
        Note that you are not required to account for the probabilities of it landing in a different cell.

        You may adjust the input variables as you wish (though it is not necessary). Output has to be (x,y) coordinates.
        
        """
        optimal_result = self.mini_max_get_trap(grid, depth = 5, maximizing_player = True)
        #print(optimal_result)
        optimal_state = optimal_result[0]
        #print(optimal_state)

        rows, cols = optimal_state.dim, optimal_state.dim
        for i in range(rows):
            for j in range(cols):
                if grid.map[i][j] != optimal_state.map[i][j]:
                    #print(111)
                    #grid.print_grid()
                    #print(222)
                    return (i, j)

        return None

    def mini_max_get_trap(self, grid, depth, maximizing_player):
        # Base case, evaluate leaf nodes.
        if depth == 0:
            eval_result = (grid, self.IS(grid, 3 - self.player_num))
            #print(eval_result)
            return eval_result

        # find available positions for throwing traps
        opponent_pos = grid.find(3 - self.player_num)
        opponent_valid_neighbors = grid.get_neighbors(opponent_pos, True)

        if not opponent_valid_neighbors:
            #print("op: ",opponent_pos)
            #print("ava op: ", opponent_valid_neighbors)
            #grid.print_grid()
            grid_copy = grid.clone()
            grid_copy.trap(random.choice(grid.getAvailableCells()))
            return (grid_copy, self.IS(grid_copy, 3 - self.player_num))

        #print(opponent_valid_neighbors)
        if maximizing_player:
            max_eval = (None, float('-inf'))

            for child in opponent_valid_neighbors:
                grid_copy = grid.clone()
                grid_copy.trap(child)

                # Recursion
                eval_result = self.mini_max_get_trap(grid_copy, depth - 1, False)
                eval_grid = eval_result[0]
                eval_score = eval_result[1]
                if max_eval[1] <= eval_score:
                    max_eval = eval_result
            return max_eval

        else:
            min_eval = (None, float('inf'))

            for child in opponent_valid_neighbors:
                grid_copy = grid.clone()
                grid_copy.trap(child)

                # Recursion
                eval_result = self.mini_max_get_trap(grid_copy, depth - 1, True)
                eval_grid = eval_result[0]
                eval_score = eval_result[1]
                if eval_score <= min_eval[1]:
                    min_eval = eval_result

            return min_eval

    def IS(self, grid, player_num):

        # find all available moves by Player
        player_moves = grid.get_neighbors(grid.find(player_num), only_available = True)

        # find all available moves by Opponent
        opp_moves = grid.get_neighbors(grid.find(3 - player_num), only_available = True)

        return len(player_moves) - len(opp_moves)

    def AM(self, grid : Grid, player_num):

        available_moves = grid.get_neighbors(grid.find(player_num), only_available = True)

        return len(available_moves)