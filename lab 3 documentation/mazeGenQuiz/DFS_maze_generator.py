'''
Program Created by Mahrad Pisheh Var, October 2020
@ University of Essex

Updated by M. Fairbank for CE811 Oct 2020, August 2021
'''

import random

from typing import List

class maze_cell(object):
    # Helper class to represent a single cell of the maze, plus
    # maintain details of which surrounding walls are present.
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.visited = False
        self.surrounding_walls = {'up': True, 'down': True, 'left': True, 'right': True}
        self.neighbour_directions = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

    def set_visited(self):
        self.visited = True

    def is_visited(self):
        return self.visited

    def out_of_grid(self, grid, dx, dy):
        maze_height = len(grid)
        maze_width = len(grid[0])
        if self.x + dx < 0 or self.x + dx >= maze_width:
            return True
        elif self.y + dy < 0 or self.y + dy >= maze_height:
            return True
        else:
            return False

    def get_unvisited_neighbours(self, grid) -> List["maze_cell"]:
        neighbours: List["maze_cell"] = []
        for _, dir1 in self.neighbour_directions.items():
            (dx, dy) = dir1
            if not self.out_of_grid(grid, dx, dy):
                chosen_neighbour = grid[self.y + dy][self.x + dx]
                if not chosen_neighbour.is_visited():
                    neighbours.append(chosen_neighbour)
        return neighbours

    def remove_wall(self, other):
        if other.x > self.x:
            other.surrounding_walls['left'] = False
            self.surrounding_walls['right'] = False
        elif other.x < self.x:
            other.surrounding_walls['right'] = False
            self.surrounding_walls['left'] = False
        elif other.y > self.y:
            other.surrounding_walls['up'] = False
            self.surrounding_walls['down'] = False
        elif other.y < self.y:
            other.surrounding_walls['down'] = False
            self.surrounding_walls['up'] = False

    def __str__(self):
        return ("(x" + str(self.x) + ",y" + str(self.y) + ")")


def build_maze_grid_dfs(n_rows, n_columns):
    grid: List[List[maze_cell]] = [[maze_cell(x, y) for x in range(n_columns)] for y in range(n_rows)]
    curr_cell: maze_cell = grid[0][0]  # always starts from top left corner
    stack: List[maze_cell] = []
    while True:
        curr_cell.set_visited()
        neighbours = curr_cell.get_unvisited_neighbours(grid)

        if len(neighbours) > 0:
            stack.append(curr_cell)
            x = random.choice(neighbours)
            curr_cell.remove_wall(x)
            curr_cell = x
        else:
            if len(stack) > 0:
                curr_cell = stack.pop()
            else:
                break
    return grid


def convert_maze(grid):
    # This function converts the maze structure, which is an nested list
    # of type maze_cell, into a nested list of 1s and 0s.
    # For example, the output of this function is something like
    # [[1,1,1,1,1],[1,0,1,0,1],[1,0,0,0,1],[1,1,1,1,1]]
    # The nested list of 1s and 0s is more communicable to other programs
    converted_grid = []
    ylength = len(grid) * 2 + 1
    xlength = len(grid[0]) * 2 + 1
    for y in range(ylength):
        converted_grid.append([1] * xlength)
    for y, row_of_cells in enumerate(grid):
        for x, cell in enumerate(row_of_cells):
            for i, direction in enumerate(cell.surrounding_walls):
                if cell.is_visited():
                    # they should all be visited
                    converted_grid[y * 2 + 1][x * 2 + 1] = 0
                if not (cell.surrounding_walls[direction]):
                    (dy, dx) = cell.neighbour_directions[direction]
                    converted_grid[y * 2 + 1 + dy][x * 2 + 1 + dx] = 0
    return converted_grid


def display_maze_graphically(converted_maze):
    import pygame
    pygame.init()
    display_cell_size = 15
    green = (0, 155, 0)
    brown = (205, 133, 63)
    display_width = len(converted_maze[0]) * display_cell_size
    display_height = len(converted_maze) * display_cell_size

    pygame.display.set_caption("Maze_viewer")
    gameDisplay = pygame.display.set_mode((display_width, display_height))
    gameDisplay.fill((brown))
    for y, row_of_cells in enumerate(converted_maze):
        for x, cell in enumerate(row_of_cells):
            if cell == 1:
                pygame.draw.rect(gameDisplay, green,
                                 pygame.Rect(x * display_cell_size, y * display_cell_size, display_cell_size - 1,
                                             display_cell_size - 1))
    pygame.display.flip()
    input("Press enter")


size_x = 6
size_y = 6
grid = build_maze_grid_dfs(size_y, size_x)
converted_maze = convert_maze(grid)
print("Generated Maze:")
print(converted_maze)

# This next display method displays in a slightly easier to read form:
print("\n")
print('\n'.join([''.join(str(x)) for x in converted_maze]))

# This next display method displays the maze graphically:
# To get graphical displayer to run, you need to have pygame installed.
# e.g. pip install pygame
# or on Ubuntu, sudo apt install python-pygame
# display_maze_graphically(converted_maze)

display_maze_graphically(converted_maze)