##########################################################################
# Spring 2020, NC State University
# Course     : CSC584 - Building Game AI
# Instructor : Dr. Chris Martens
##
# Name       : Vinay Kumar
# UnityID    : vkumar24@ncsu.edu
##
##================== Assignment-2 (Part-1) Solutions ============#########
##================== Part #2: Pathfinding ============####################
##########################################################################
# Following are the key_locations:
# castle
# tar_pit
# tavern
# cave
# tree
# forge
#
# Following are the key_characters:
# knight
# king_of_leighra
# rameses
# lady_lupa
# tree_spirit
# innkeeper
# blacksmith
##########################################################################

import json

width = 800
height = 800
cols = 8
rows = cols * height / width
grid = []

open_set = []
closed_set = []
start_node = None
goal_node = None
wid = 0
hgt = 0

heuristic = None
obstacle_percent = 0.4
obstacle_color = color(217, 217, 217)

best_cell = None
best_path = []
knight_start_pos = None
knight_img = None
knight_size = 40

is_goal_set = False
is_goal_path_found = False
best_path_coordinates = []
knight = None
idx = 0
do_animate_player = False

def setup():
    size(width, height)
    print(rows, cols)
    background(255)
    frameRate(500)
    init_map()
    init_astar()
    knight.show()
    

def draw():
    global goal_node
    global is_goal_set
    global best_path_coordinates
    global is_goal_path_found
    global idx
    global wid
    global hgt
    global do_animate_player
    
    if is_goal_set:
        run_astar(goal_node=goal_node)
        best_path.reverse()
        best_path_coordinates = [(cell.x*wid+wid/2, cell.y*hgt+hgt/2) for cell in best_path]

    
    if idx < len(best_path) and is_goal_path_found:
        do_animate_player = True
        
    if is_goal_path_found:
        reset_astar()
        do_animate_player = True
        print("best_path_coordinatesX = ", best_path_coordinates)

        
    if do_animate_player:
        refresh_map()
        knight.show()
        knight.update(best_path_coordinates, idx)
        idx += 1
        # print("idx = ", idx)
        if idx >= len(best_path):
            do_animate_player = False
            idx = 0
        for cell in best_path:
            cell.show(color(0, 0, 255))


#########################################################################################################################
class Kinematic():
    def __init__(self, pos=PVector(0.,0.), theta=0., vel=PVector(0.,0.), rot=0.):
        self.pos    = pos
        self.theta  = theta
        self.vel    = vel
        self.rot    = rot
        
class Player():
    def __init__(self, img_src, img_size, pos):
        self.img_src = img_src
        self.img = loadImage(self.img_src)
        self.img_size = img_size
        self.kinematic = Kinematic(pos)
        
    def show(self):
        with pushMatrix():
            translate(self.kinematic.pos.x, self.kinematic.pos.y)
            image(self.img, 0-self.img_size/2, 0-self.img_size/2, self.img_size, self.img_size)
            # arc(0, 0, 20, 20, 0, 2 * PI, OPEN)

        
    def update(self, path, idx):
        x = float(path[idx][0])
        y = float(path[idx][1])
        self.kinematic.pos = PVector(x, y)
        # print(self.kinematic.pos)
        
#########################################################################################################################
def refresh_map():
    background(255)
    with open("../data/data.json", "r") as read_file:
        data = json.load(read_file)
    data_keys = data.keys()

    locations = data["key_locations"]
    for loc_key in locations.keys():
        path_sprite = "../data/" + loc_key + ".png"
        # print(path_sprite)
        sprite = loadImage(path_sprite)
        image(sprite, locations[loc_key][0] - 25,
              locations[loc_key][1] - 25, 50, 50)
        fill(random(0, 255), random(0, 255), random(0, 255))
        arc(locations[loc_key][0], locations[
            loc_key][1], 5, 5, 0, 2 * PI, OPEN)

    obstacles = data["obstacles"]
    for obs_key in obstacles.keys():
        obs = obstacles[obs_key]
        fill(obstacle_color)
        stroke(obstacle_color)
        beginShape()
        for idx in obs:
            vertex(idx[0], idx[1])
        endShape(CLOSE)
    
def init_map():
    with open("../data/data.json", "r") as read_file:
        data = json.load(read_file)
    data_keys = data.keys()
    
    global knight_start_pos
    global knight_img
    global knight
    global knight_size

    
    knight_start_pos = data["knight_start"]
    pos = PVector(knight_start_pos[0], knight_start_pos[1])
    knight = Player("../data/knight.png", knight_size, pos)
    
    locations = data["key_locations"]
    for loc_key in locations.keys():
        path_sprite = "../data/" + loc_key + ".png"
        # print(path_sprite)
        sprite = loadImage(path_sprite)
        image(sprite, locations[loc_key][0] - 25,
              locations[loc_key][1] - 25, 50, 50)
        fill(random(0, 255), random(0, 255), random(0, 255))
        arc(locations[loc_key][0], locations[
            loc_key][1], 5, 5, 0, 2 * PI, OPEN)

    obstacles = data["obstacles"]
    for obs_key in obstacles.keys():
        obs = obstacles[obs_key]
        fill(obstacle_color)
        beginShape()
        for idx in obs:
            vertex(idx[0], idx[1])
        endShape(CLOSE)
        
####========================================================================================####

def init_astar():
    """ Initialization of A* algorithm"""
    
    ## Referencing the global variables, which are accessed during each draw() callback
    global heuristic
    global start_node
    global goal_node
    global grid
    global wid
    global hgt
    global open_set
    global closed_set
    print("Initializing A*: init_astar()")

    heuristic = heuristic_minimax()
    
    wid = width / cols
    hgt = height / rows
    
    ## creating the grid to represent the whole map, so that A* can be applied
    grid = [[Cell(i, j) for i in range(cols)] for j in range(rows)]

    knight_start_pos_x = floor(float(knight_start_pos[1]) / wid)
    knight_start_pos_y = floor(float(knight_start_pos[0]) / hgt)
    start_node = grid[knight_start_pos_x][knight_start_pos_y]
    goal_node = grid[2][7]

    start_node.obstacle = False
    goal_node.obstacle = False
    
    ## calculating the heuristic value for each node/cell in the grid
    for row in grid:
        for cell in row:
            cell.add_neighbors(grid)
            cell.h = heuristic.get_cost(cell, goal_node)

    ## add the start_node to the open_set so that it can be explored when run_astar() is called initially
    open_set.append(start_node)

def reset_astar():
    global start_node
    global goal_node
    global open_set
    global closed_set
    global grid
    global is_goal_path_found
    
    print("Resetting A*: reset_astar()")
    
    ### reset the parents of all cells in the grid
    for row in grid:
        for cell in row:
            cell.parent = None
        
    open_set = []                             ## reset the open_set
    closed_set = []                           ## reset the closed_set
    
    start_node = goal_node                    ## assign the reached goal_node as the next start_node
    goal_node = None                          ## reset the goal_node to NONE (it will re-assigned when the mouse id clicked)
    
    open_set.append(start_node)               ## add the new start_node to the open_set so that it can be explored when run_astar() is called initially

    is_goal_path_found = False
    
    
def run_astar(goal_node=None):
    global start_node
    global best_cell
    global best_path
    global is_goal_path_found
    global is_goal_set
    global open_set
    global closed_set
    
    # print("Running A*: run_astar()")
    
    goal_node.show(color(255, 0, 255))
    # print(goal_node)

    for c in closed_set:
        c.show(color(255, 0, 0), no_fill=True)
    for c in open_set:
        c.show(color(0, 255, 0), no_fill=True)
        
    ## sanity check to see if the goal_node is in the closed_set
    ## (this case can arise if the user click on a grid cell after it has been explored)
    ## becasue the goal cell now is already in the closed_set it will never be expanded and hence no solution
    ## this check prevents this bug
    if goal_node in closed_set:
        # print(goal_node in closed_set)
        closed_set.remove(goal_node)
        open_set.append(goal_node)

    if len(open_set) > 0:
        best_cell = open_set[0]
        for cell in open_set:
            if cell.f < best_cell.f:
                best_cell = cell
        if best_cell == goal_node:
            print("DONE!! GOAL REACHED!!")
            is_goal_path_found =  True
            is_goal_set = False
            
        # print(open_set)
        open_set.remove(best_cell)
        closed_set.append(best_cell)
        
        if not is_goal_path_found:
            for nbr in best_cell.neighbors:
                if nbr not in closed_set and not nbr.obstacle:
                    # if the nbr is not in closed_set
                    # (if the nbr is in closed_set then we don't do anything
                    # b'coz, we already has visited it by some other best path)
                    temp_g = best_cell.g + 1
                    if nbr in open_set:
                        # if the nbr is not in closed_set but in open_set
                        if nbr.g > temp_g:
                            nbr.g = temp_g
                            nbr.parent = best_cell
                    else:
                        # if the nbr is not in closed_set and also not in open_set
                        open_set.append(nbr)
                        nbr.g = temp_g
                        nbr.parent = best_cell
    
                    # print(">>>>>>>>>>>>>>>>>>>", nbr.f, nbr.g, nbr.h, "\n")
                    nbr.f = nbr.g + nbr.h
    else:
        ## all possible nodes have been explored and open_set is empty
        print("NO SOLUTION !!!")
        is_goal_path_found = False
        is_goal_set = False

    # update the best_path
    best_path = []
    temp = best_cell
    best_path.append(temp)
    while temp.parent:
        best_path.append(temp.parent)
        temp = temp.parent

    for c in best_path:
        c.show(color(0, 0, 255), no_fill=True)
        

def mousePressed():
    """this method is triggered whenever the mouse is pressed"""
    global goal_node  # referencs the global variable "goal_node"
    global grid
    global is_goal_set
    
    goal_x = floor(rows * float(mouseX) / height)
    goal_y = floor(cols * float(mouseY) / width)
    # grabs the x & y coordinates of the mouse when pressed
    goal_node = grid[goal_y][goal_x]
    is_goal_set = True
    
    ## calculating the heuristic value for each node/cell in the grid
    for row in grid:
        for cell in row:
            cell.h = heuristic.get_cost(cell, goal_node)

    print("goal_node = ", goal_x, goal_y)

class heuristic_eucledian():

    def __init__(self):
        pass

    def get_cost(self, a, b):
        return dist(a.x, a.y, b.x, b.y)

class heuristic_manhattan():

    def __init__(self):
        pass

    def get_cost(self, a, b):
        return abs(a.x - b.x) + abs(a.y - b.y)
    
class heuristic_minimax():

    def __init__(self):
        pass

    def get_cost(self, a, b):
        return max(abs(a.x-b.x), abs(a.y-b.y))
class Cell():

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.f = 0
        self.g = 0
        self.h = 0
        self.parent = None
        self.neighbors = []
        self.obstacle = False  # not an obstacle by default

        # Checking the color of the pixel at the center of the cell to decide
        # whether it lies within an obstacle or not
        self.center_x = self.x * wid + wid / 2
        self.center_y = self.y * hgt + hgt / 2
        self.cell_color = get(self.center_x, self.center_y)

        if self.cell_color == obstacle_color:
            self.obstacle = True
            # print(hgt, wid, self.x, self.y, self.cell_color, obstacle_color)
        else:
            self.obstacle = False
            # print(hgt, wid, self.x, self.y, self.cell_color, obstacle_color)


    def show(self, clr, no_fill=True):
        if self.obstacle:
            if no_fill:
                noFill()
            else:
                fill(color(0))
            stroke(0)
        else:
            if no_fill:
                noFill()
            else:
                fill(clr)
            stroke(clr)

        # stroke(0)
        # noStroke()
        rect(self.x * wid, self.y * hgt, wid, hgt)

    def add_neighbors(self, grid):
        if self.x > 0:
            self.neighbors.append(grid[self.y][self.x - 1])
        if self.x < cols - 1:
            self.neighbors.append(grid[self.y][self.x + 1])
        if self.y > 0:
            self.neighbors.append(grid[self.y - 1][self.x])
        if self.y < rows - 1:
            self.neighbors.append(grid[self.y + 1][self.x])
     
