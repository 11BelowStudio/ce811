big_maze=[[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
 [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
 [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1], 
 [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], 
 [1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1], 
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1], 
 [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1], 
 [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1], 
 [1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1], 
 [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1], 
 [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1], 
 [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], 
 [1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1], 
 [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1], 
 [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1], 
 [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1], 
 [1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1], 
 [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], 
 [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1], 
 [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], 
 [1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1], 
 [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1], 
 [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1], 
 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1], 
 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
 
small_maze=[[1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1], 
[1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1], [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1], 
[1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1], [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1], 
[1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]


import numpy as np


def calculate_neighboring_nodes(node,maze):
    # for any node (y,x), calculates a dictionary of which nodes are neighbours in this maze array?
    maze_height=len(maze)
    maze_width=len(maze[0])
    directions=[(1,0),(-1,0),(0,1),(0,-1)]
    (y,x)=node
    result_neigbours={}
    for (dy,dx) in directions:
        neighbour_y=y+dy
        neighbour_x=x+dx
        # check if this potential neighbour goes off the edges of the maze:
        if neighbour_y>=0 and neighbour_y<maze_height and neighbour_x>=0 and neighbour_x<maze_width:
            # check if this potential neighbour is not a wall:
            if maze[neighbour_y][neighbour_x]==0:
                # we have found a valid neighbouring node that is not a wall
                result_neigbours[(neighbour_y,neighbour_x)]=1 # this says the distance to this neighbour is 1
                # Note that all neighbours in thisproblem are distance 1 away!
    return result_neigbours



def solve_maze(maze, start_node, end_node):
    # on entry, maze is a nested list (2d-array) of 1s and 0s indicating maze walls and corridors respectively.
    maze_height=len(maze)
    maze_width=len(maze[0])
    assert maze[start_node[0]][start_node[1]]==0 # start node must point to a zero of the maze array
    assert maze[end_node[0]][end_node[1]]==0 # end node must point to a zero of the maze array

    # identify all nodes of the graph.  A node is a tuple object (y,x) corresponding to the location of the node in the maze.
    nodes=[(y,x) for x in range(maze_width)for y in range(maze_height)  if maze[y][x]==0]
    print("nodes:",nodes) 
    # store all of each node's immediate neighbours in a dictionary for speedy reference:                
    neighbouring_nodes={node:calculate_neighboring_nodes(node,maze) for node in nodes} 
    # The above line builds our "graph" of nodes and connections.  
    # Each connection is length 1 - which is rather inefficient.  Maybe we 
    # could skip lots of nodes which form a long corridor? 
    # See https://www.youtube.com/watch?v=rop0W4QDOUI for a discussion of one way to do that.
    print("neighbouring_nodes1:",neighbouring_nodes)
                               

    # Now implement Dijkstra's algorithm
    open_nodes=[start_node] # seed the flood-fill search to expand from here.
    parentNodes={start_node: None} # These hold which node precedes this one on the best route.
    nodeGValues={start_node: 0} # These hold the distance of each node from the start node.
    closed_list = [] # List of nodes that are completely finished

    while len(open_nodes)>0:
        sorted_list_of_open_nodes_nodes=sorted(open_nodes, key = lambda n: nodeGValues[n]) # this sorts the open_nodes nodes list by gValues first.
        current_node = sorted_list_of_open_nodes_nodes[0] # this pulls off the first (i.e. shortest-distance) open_nodes item.
        current_distance=nodeGValues[current_node]
        closed_list.append(current_node)
        open_nodes.remove(current_node)
        #consider all of the neighbours of the current node:
        for neighbour_node, neighbour_distance in neighbouring_nodes[current_node].items():
            # TODO insert code here for the core part of Dijkstra's algorithm




    # Now dijkstra's algorithm is finished.
    # Summarise the gScores into a single grid numpy array
    array_gScores=np.zeros((maze_height, maze_width),np.int32)
    for n in nodes:
        (y,x)=n
        if n not in nodeGValues:
            raise Exception("Not a fully connected maze!")
        array_gScores[y,x]=nodeGValues[n] 

    # Indicate the shortest-path solution to this maze with a path of "*" symbols:
    current_node=end_node # work backwards from the end_node of this maze
    while current_node!=None:
        (y,x)=current_node
        maze[y][x]="*" # mark the optimal route with * symbols
        # Work backwards along chain of parent nodes...
        current_node=parentNodes[current_node]
        
    return [maze, array_gScores, closed_list] # solved_maze and gscores and closed_list
        


maze=small_maze
start_node=(0,1) # this must point to a zero!
maze_height=len(maze)
maze_width=len(maze[0])
end_node=(maze_height-2,maze_width-2) 
[solved_maze, array_gScores, closed_list]=solve_maze(maze, start_node, end_node)
#Using numpy array to print the maze in a more readable format:
print(str(np.array(solved_maze)).replace("'",""))
print("closed_list order",closed_list)
print("GScores", array_gScores)



def display_maze_graphically(converted_maze):
    import pygame   
    import pygame.freetype  # Import the freetype module.
    pygame.init()
    font_size=12
    GAME_FONT = pygame.freetype.SysFont(name=pygame.freetype.get_default_font(),size=font_size)
    display_cell_size=21
    green = (0,155,0)
    walls = (205,0,0)    
    brown = (205,133,63)    
    display_width = len(converted_maze[0])*display_cell_size
    display_height = len(converted_maze)*display_cell_size

    pygame.display.set_caption("Maze_viewer")
    gameDisplay = pygame.display.set_mode((display_width,display_height))
    gameDisplay.fill((brown))
    for y,row_of_cells in enumerate(converted_maze):
        for x, cell in enumerate(row_of_cells):
            if cell!=1:
                pygame.draw.rect(gameDisplay, green, pygame.Rect((x*display_cell_size, y*display_cell_size, display_cell_size-1,display_cell_size-1)))
            if cell not in [0,1]:
                GAME_FONT.render_to(gameDisplay, (x*display_cell_size+(display_cell_size-font_size)//2, y*display_cell_size+(display_cell_size-font_size)//2), str(cell), (0, 0, 0))
                
    pygame.display.flip()
    input("press enter")

#Uncomment the following lines to plot the result graphically.
# You might need to "pip install pygame" 
# On ubuntu try "sudo apt install python-pygame"
#display_maze_graphically(solved_maze)
