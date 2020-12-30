#########################################################################################################################
# Spring 2020, NC State University
# Course     : CSC584 - Building Game AI
# Instructor : Dr. Chris Martens
##
# Name       : Vinay Kumar
# UnityID    : vkumar24@ncsu.edu
##
##================== Assignment-2 (Part-1) Solutions ============#########
##================== Part #3: Decision Making ============################
#########################################################################################################################
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
#########################################################################################################################
#########################################################################################################################

import json

width = 640
height = 480
cols = 64
rows = cols * height / width
grid = []

greet_king_reward = None
world_has_dict = None
world_wants_dict = None
gold_with_knight = 0

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
    create_ai()
    

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
    
    global data
    
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
    global data
    
    global knight_start_pos
    global knight_img
    global knight
    global knight_size
    
    global greet_king_reward
    global world_has_dict
    global world_wants_dict
    
    with open("../data/data.json", "r") as read_file:
        data = json.load(read_file)
        
    
    greet_king_reward = data["greet_king"]
    state_of_world = data["state_of_world"]
    
    world_has_dict = {}
    for item in state_of_world["Has"]:
        if item[0] not in world_has_dict:
            world_has_dict[item[0]] = [item[1]]
        else:
            world_has_dict[item[0]].append(item[1])
    
    world_wants_dict = {}
    for item in state_of_world["Wants"]:
        if item[0] not in world_wants_dict:
            world_wants_dict[item[0]] = [item[1]]
        else:
            world_wants_dict[item[0]].append(item[1])

    
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

    heuristic = heuristic_manhattan()
    
    wid = width / cols
    hgt = height / rows
    
    ## creating the grid to represent the whole map, so that A* can be applied
    grid = [[Cell(i, j) for i in range(cols)] for j in range(rows)]

    knight_start_pos_x = floor(float(knight_start_pos[1]) / wid)
    knight_start_pos_y = floor(float(knight_start_pos[0]) / hgt)
    start_node = grid[knight_start_pos_x][knight_start_pos_y]
    goal_node = grid[2][50]

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
                
            
#########################################################################################################################
#########################################################################################################################
############################################# GOAL ORIENTED ACTION PLANNING #############################################

#########################################################################################################################

class GameWorld():
    def __init__(self):
        self.game_planners = []                                              ### placeholder for all planners in the game
        self.game_plans = []                                                 ### the game plan
        
    def add_game_planner(self, game_planner):
        """Adds a planner to the game world"""
        self.game_planners.append(game_planner)
        
    def search(self):
        """Performs search of the plans for the whole game world"""
        self.game_plans = []
        for plnr in self.game_planners:
            self.game_plans.append(plnr.search())                            ### searching the plan for each planner and appending it to the final game plan
            
    def find_plan(self, verbose=False):
        """Finds plan for each planner and returns cost-sorted plans"""
        plans = {}
        for plan in self.game_plans:
            cost = 0
            for action in plan:
                cost += action["g_val"]                                      ### calculating the cost of each action and addingit to the final cost of the plan
                
            if cost in plans:
                plans[cost].append(plan)
            else:
                plans[cost] = [plan]
                
        sorted_plans = sorted(plans.keys())                                  ### sorting the plans based on the cost for each plan
        
        if verbose:
            i = 1
            for score in sorted_plans:
                for plan in plans[score]:
                    print("plan#", i)
                    for act in plan:
                        print("\t", act["name"])
                    i += 1
                    print("\nTotal cost: ", score, "\n")

        return [plans[k][0] for k in sorted_plans]
    
        
class Planner():
    def __init__(self, *args):
        self.start_state = None
        self.goal_state  = None
        self.vals ={arg: -1 for arg in args}
        self.actions = None
        
    def state(self, **kwargs):
        """Creates/Updates the state based on the kwargs"""
        updated_state = self.vals.copy()
        updated_state.update(kwargs)
        return updated_state
    
    def init_start_state(self, **kwargs):
        """Initializes the start state for the planner"""
        self.start_state = self.state(**kwargs)
        
    def init_goal_state(self, **kwargs):
        """Initializes the goal state for the planner"""
        self.goal_state = self.state(**kwargs)
        
    def set_actions(self, actions):
        self.actions = actions
        
    def search(self):
        return run_astar_planner(self.start_state,
                                 self.goal_state,
                                 {c: self.actions.conditions[c].copy() for c in self.actions.conditions},
                                 {r: self.actions.reactions[r].copy()  for r in self.actions.reactions},
                                 self.actions.weights.copy())
    
class Actions():
    def __init__(self):
        self.conditions = {}
        self.reactions  = {}
        self.weights    = {}
        
    def add_condition(self, idx, **kwargs):
        if not idx in self.weights:
            self.weights[idx] = 1
            
        if not idx in self.conditions:
            self.conditions[idx] = kwargs
            ### check for knights num_gold coins
            ###
            return
        else:
            ### check for knights num_gold coins
            ###
            self.conditions[idx].update(kwargs)
            
    def add_reaction(self, idx, **kwargs):
        if not idx in self.conditions:
            raise Exception("invalid idx. Cannot add reaction")
        if not idx in self.reactions:
            self.reactions[idx] = kwargs
            ### check for knights num_gold coins
            ###
            return
        else:
            ### check for knights num_gold coins
            ###
            self.reactions[idx].update(kwargs)
            
    def add_weight(self, idx, val):
        if not idx in self.conditions:
            raise Exception("invalid idx. Cannot assign weight")
        self.weights[idx] = val
        
    
def calc_dist(state1, state2):
    """Calculate the distance between two states"""
    checked_idx = []
    distance = 0
    
    for idx in state1.keys():
        if state1[idx] == -1:
            continue
        if not state1[idx] == state2[idx]:
            distance += 1
        checked_idx.append(idx)
    
    for idx in state2.keys():
        if idx in checked_idx:
            continue
        if state2[idx] == -1:
            continue
        if not state2[idx] == state1[idx]:
            distance += 1
    
    return distance
        
        
def validate_conditions(state1, state2):
    for idx in state2.keys():
        if state2[idx] == -1:
            continue
        if not state1[idx] == state2[idx]:
            return False
        ### write a condition to check for number_of_gold available to knight bcoz its integer not boolean
        
    return True


def create_node(path, state, name):
    path["node_id"] += 1
    path["nodes"][path["node_id"]] = {"name": name,
                                      "p_id": None,
                                      "id": path["node_id"],
                                      "state": state,
                                      "f_val": 0,
                                      "g_val": 0,
                                      "h_val": 0,
                                      }
    return path["nodes"][path["node_id"]]


def check_node(node, node_array):
    for _next in node_array.values():
        if node["state"]==_next["state"] and node["name"]==_next["name"]:
            return True
    return False

def run_astar_planner(start_state, goal_state, actions, reactions, weight_tbl):
    pth = {"nodes": {},
           "node_id": 0,
           "goal": goal_state,
           "actions": actions,
           "reactions": reactions,
           "weight_tbl": weight_tbl,
           "action_nodes": {},
           "open_set": {},
           "closed_set": {},
           }
    start_node = create_node(path=pth, state=start_state, name="start")
    start_node["g_val"] = 0
    start_node["h_val"] = calc_dist(state1=start_state, state2=goal_state)
    start_node["f_val"] = start_node["g_val"] + start_node["h_val"]
    pth["open_set"][start_node["id"]] = start_node
    
    for act in actions:
        pth["action_nodes"][act] = create_node(path=pth, state=actions[act], name=act)
        
    return find_best_astar_path(pth)
    
    
def find_best_astar_path(path):
    node = None
    open_set = path["open_set"]
    closed_set = path["closed_set"]
    while len(open_set) > 0:
        ## find the best node
        best = {"node": None,
                "f_val": 9999999
                }
        for _next in open_set.values():
            if not best["node"] or _next["f_val"] < best["f_val"]:
                best["node"] = _next["id"]
                best["f_val"] = _next["f_val"]
                
        if best["node"]:
            node = path["nodes"][best["node"]]
        else:
            return
        
        del open_set[node["id"]]                  ### remove the node from the open set
        
        ### check if the goal is found
        if validate_conditions(node["state"], path["goal"]):
            pth = []
            while node["p_id"]:
                pth.append(node)
                node = path["nodes"][node["p_id"]]
            pth.reverse()
            return pth
        
        ### add the node to the closed set
        closed_set[node["id"]] = node
        
        ### finding the neighbors
        neighbors = []
        for act_name in path["action_nodes"]:
            if not validate_conditions(node["state"], path["action_nodes"][act_name]["state"]):
                continue
            path["node_id"] += 1
            
            x_node = node.copy()
            x_node["state"] = node["state"].copy()
            x_node["id"] = path["node_id"]
            x_node["name"] = act_name
            
            for i in path["reactions"][act_name]:
                val = path["reactions"][act_name][i]
                if val == -1:
                    continue
                x_node["state"][i] = val
                
            path["nodes"][x_node["id"]] = x_node
            neighbors.append(x_node)
            
        for nbr in neighbors:
            g_val = node["g_val"] + path["weight_tbl"][nbr["name"]]
            in_openset = check_node(nbr, open_set)
            in_closedset = check_node(nbr, closed_set)
            
            if in_openset and g_val < nbr["g_val"]:
                del open_set[nbr]
            if in_closedset and g_val < nbr["g_val"]:
                del closed_set[nbr["id"]]
            if not in_openset and not in_closedset:
                nbr["g_val"] = g_val
                nbr["h_val"] = calc_dist(nbr["state"], path["goal"])
                nbr["f_val"] = nbr["g_val"] + nbr["h_val"]
                nbr["p_id"]  = node["id"]
                open_set[nbr["id"]] = nbr
        
    return []

#########################################################################################################################
## creating specialized condition-action-reaction states for the Game and using GOAP to reach goal-state

def create_ai():
    
    global greet_king_reward
    global world_has_dict
    global world_wants_dict
    global gold_with_knight
    
    print(world_has_dict)
    print(world_wants_dict)

    game_ai = GameWorld()
    knight_ai = Planner("at_start",                  ## The game starts here. Knight has nothing.
                        "at_castle",                 ## KingOfLeighra livs here
                        "at_cave",                   ## LadyLupa lives here
                        "at_forge",                  ## Blacksmith lives here
                        "at_tarpit",                 ## Rameses lives here
                        "at_tavern",                 ## Innkeeper lives here
                        "at_tree",                   ## TreeSpirit livs here
                        "has_ale",                   ## create()
                        "has_axe",                   ## has(Blacksmith at_forge)
                        "has_blade",                 ## has(Blacksmith at_forge)
                        "has_cheap_sword",            ## create(has_blade, has_wood), do()
                        "has_fenrir",                ## has(LadyLupa at_cave), do(dead_rameses)
                        "has_fire",                  ## create()
                        "has_gold",                  ## has(King at_castle), wants(Innkeeper), wants(Blacksmith)
                        "has_poisoned_fenrir",        ## create(has_fenrir, has_wolfsbane), do(dead_rameses at_tarPit)
                        "has_poisoned_sword",         ## create(has_cheapSword, has_wolfsbane), wants(knightAI), do(dead_rameses at_tarPit)
                        "has_water",                 ## has(Innkeeper at_tavern), wants(TreeSpirit at_tree)
                        "has_wolfsbane",             ## has(TreeSpirit at_tree), wants(LadyLupa at_cave)
                        "has_wood",                  ## create(has_axe, at_tree)"fight_rameses",
                        "dead_knight",               ## ENDGAME
                        "dead_rameses",              ## ENDGAME
                        "endgame",                   ## ENDGAME (dead_Knight or dead_rameses)
                    )
    
    knight_ai.init_start_state(at_start=True,
                            at_castle=False,
                            at_cave=False,
                            at_forge=False,
                            at_tarpit=False,
                            at_tavern=False,
                            at_tree=False,
                            has_ale=False,
                            has_axe=False,
                            has_blade=False,
                            has_cheap_sword=False,
                            has_fenrir=False,
                            has_fire=False,
                            has_gold=False,
                            has_poisoned_fenrir=False,
                            has_poisoned_sword=False,
                            has_water=False,
                            has_wolfsbane=False,
                            has_wood=False,
                            dead_knight=False,
                            dead_rameses=False,
                            endgame=False
                            )
    
    knight_ai.init_goal_state(dead_rameses=True)
    knight_actions = Actions()
    
    ## condition-reaction for trade-rules at the start
    knight_actions.add_condition("start",
                                at_start=True,
                                endgame=False)
    knight_actions.add_reaction("start",
                                at_start=False,
                                at_castle=True)
    
    ## create condition-reaction to go to castle if the knight doesnot have enough gold coins
    knight_actions.add_condition("go_to_castle_&_greet_king_to_get_gold",
                                at_castle=False,
                                has_gold=False)
    knight_actions.add_reaction("go_to_castle_&_greet_king_to_get_gold",
                                at_castle=False,
                                has_gold=True)
    
    # ## condition-reaction for trade-rules with King
    # knight_actions.add_condition("greet_king",
    #                             at_castle=True,
    #                             has_gold=False)
    # knight_actions.add_reaction("greet_king",
    #                             at_castle=False,
    #                             has_gold=True)
    
    ## condition-reaction for trade-rules with Blacksmith at forge
    wantable_items = ["1gold", "Axe", "Blade", "Water", "Ale", "Wolfsbane"]
    hasable_items = ["Axe", "Blade", "Water", "Ale", "Wolfsbane", "Fenrir"]
    key_players = []
    key_players_locations = []

    # for plyr in key_players:
    #     for h in hasable_items:
    #         for w in wantable_items:
    #             if h in world_has_dict[plyr] and w in world_wants_dict[plyr]:
    #                 knight_actions.add_condition("get"+h+"from"+plyr+"by giving"+w,
    #                                              =True,
    #                                              "has"+h=True, "has"+w=False)


    if "1gold" in world_wants_dict["Blacksmith"] and "Axe" in world_has_dict["Blacksmith"]:
        knight_actions.add_condition("go_to_forge_if_doesnot_have_axe",
                                        at_forge=False,
                                        has_gold=True, has_axe=False)
        knight_actions.add_reaction("go_to_forge_if_doesnot_have_axe",
                                        at_forge=True)

    if "1gold" in world_wants_dict["Blacksmith"] and "Blade" in world_has_dict["Blacksmith"]:
        knight_actions.add_condition("go_to_forge_if_doesnot_have_blade",
                                        at_forge=False,
                                        has_gold=True, has_blade=False)
        knight_actions.add_reaction("go_to_forge_if_doesnot_have_blade",
                                        at_forge=True)

    if "1gold" in world_wants_dict["Blacksmith"] and "Axe" in world_has_dict["Blacksmith"]:
        knight_actions.add_condition("get_axe_from_blacksmith",
                                    at_forge=True, has_axe=False, has_gold=True)
        knight_actions.add_reaction("get_axe_from_blacksmith",
                                    at_forge=False, has_axe=True)

    if "1gold" in world_wants_dict["Blacksmith"] and "Blade" in world_has_dict["Blacksmith"]:
        knight_actions.add_condition("get_blade_from_blacksmith",
                                    at_forge=True, has_blade=False, has_gold=True)
        knight_actions.add_reaction("get_blade_from_blacksmith",
                                    at_forge=False, has_blade=True)
    
    ## condition-reaction for trade-rules with LadyLupa at cave

    knight_actions.add_condition("go_to_cave",
                                    at_cave=False, has_wolfsbane=True)
    knight_actions.add_reaction("go_to_cave",
                                    at_cave=True)

    if "Wolfsbane" in world_wants_dict["Lady Lupa"] and "Fenrir" in world_has_dict["Lady Lupa"]:
        knight_actions.add_condition("get_fenrir_from_ladylupa",
                                    at_cave=True,
                                    has_wolfsbane=True)
        knight_actions.add_reaction("get_fenrir_from_ladylupa",
                                    at_cave=False,
                                    has_fenrir=True,
                                    has_wolfsbane=False)
    
    
    ## condition-reaction for trade-rules with Rameses at tarpit
    
    knight_actions.add_condition("go_to_tarpit_with_cheap_sword",
                                    at_tarpit=False, has_cheap_sword=True)
    knight_actions.add_reaction("go_to_tarpit_with_cheap_sword",
                                    at_tarpit=True)
    
    knight_actions.add_condition("go_to_tarpit_with_fenrir",
                                    at_tarpit=False, has_fenrir=True)
    knight_actions.add_reaction("go_to_tarpit_with_fenrir",
                                    at_tarpit=True)
    
    knight_actions.add_condition("go_to_tarpit_with_fire",
                                    at_tarpit=False, has_fire=True)
    knight_actions.add_reaction("go_to_tarpit_with_fire",
                                    at_tarpit=True)
    
    knight_actions.add_condition("go_to_tarpit_with_poisoned_fenrir",
                                    at_tarpit=False, has_poisoned_fenrir=True)
    knight_actions.add_reaction("go_to_tarpit_with_poisoned_fenrir",
                                    at_tarpit=True)
    
    knight_actions.add_condition("go_to_tarpit_with_poisoned_sword",
                                    at_tarpit=False, has_poisoned_sword=True)
    knight_actions.add_reaction("go_to_tarpit_with_poisoned_sword",
                                    at_tarpit=True)
    
    knight_actions.add_condition("fight_rameses_with_fenrir",
                                at_tarpit=True, has_fenrir=True)
    knight_actions.add_reaction("fight_rameses_with_fenrir",
                                at_tarpit=False, has_fenrir=False,
                                dead_rameses=True, endgame=True)
    
    knight_actions.add_condition("fight_rameses_with_fire",
                                at_tarpit=True, has_fire=True)
    knight_actions.add_reaction("fight_rameses_with_fire",
                                at_tarpit=False, has_fire=False,
                                dead_knight=True, dead_rameses=True, endgame=True)
                                
    knight_actions.add_condition("fight_rameses_with_cheap_sword",
                                at_tarpit=True, has_cheap_sword=True)
    knight_actions.add_reaction("fight_rameses_with_cheap_sword",
                                at_tarpit=False, has_cheap_sword=False,
                                dead_knight=True, endgame=True)
    
    knight_actions.add_condition("fight_rameses_with_poisoned_fenrir",
                                at_tarpit=True, has_poisoned_fenrir=True)
    knight_actions.add_reaction("fight_rameses_with_poisoned_fenrir",
                                at_tarpit=False, has_poisoned_fenrir=False,
                                dead_rameses=True, endgame=True)
    
    knight_actions.add_condition("fight_rameses_with_poisoned_sword",
                                at_tarpit=True, has_poisoned_sword=True)
    knight_actions.add_reaction("fight_rameses_with_poisoned_sword",
                                at_tarpit=False, has_poisoned_sword=False,
                                dead_rameses=True, endgame=True)
    

    ## condition-reaction for trade-rules with Innkeeper at tavern
    knight_actions.add_condition("go_to_tavern",
                                    at_tavern=False, has_gold=True)
    knight_actions.add_reaction("go_to_tavern",
                                    at_tavern=True)
    if "1gold" in world_wants_dict["Innkeeper"] and "Ale" in world_has_dict["Innkeeper"]:
        knight_actions.add_condition("get_ale_from_innkeeper",
                                    at_tavern=True, has_gold=True, has_ale=False)
        knight_actions.add_reaction("get_ale_from_innkeeper",
                                    at_tavern=False, has_ale=True)

    if "1gold" in world_wants_dict["Innkeeper"] and "Water" in world_has_dict["Innkeeper"]:
        knight_actions.add_condition("get_water_from_innkeeper",
                                    at_tavern=True, has_gold=True, has_water=False)
        knight_actions.add_reaction("get_water_from_innkeeper",
                                    at_tavern=False, has_water=True)
    
    
    ## condition-reaction for trade-rules with TreeSpirit at tree
    knight_actions.add_condition("go_to_tree_with_water",
                                    at_tree=False, has_water=True)
    knight_actions.add_reaction("go_to_tree_with_water",
                                    at_tree=True)

    knight_actions.add_condition("go_to_tree_with_axe",
                                    at_tree=False, has_axe=True)
    knight_actions.add_reaction("go_to_tree_with_axe",
                                    at_tree=True)
    
    if "Water" in world_wants_dict["Tree Spirit"] and "Wolfsbane" in world_has_dict["Tree Spirit"]:
        knight_actions.add_condition("get_wolfsbane_from_treespirit",
                                    at_tree=True, has_water=True)
        knight_actions.add_reaction("get_wolfsbane_from_treespirit",
                                    at_tree=False, has_water=False, has_wolfsbane=True)
    
    knight_actions.add_condition("get_wood_from_treespirit",
                                at_tree=True, has_axe=True)
    knight_actions.add_reaction("get_wood_from_treespirit",
                                at_tree=False, has_wood=True)
    
    ## Create different items from the available items.
    
    knight_actions.add_condition("create_cheap_sword",
                                has_blade=True, has_wood=True)
    knight_actions.add_reaction("create_cheap_sword",
                                has_blade=False, has_wood=False,
                                has_cheap_sword=True)
    
    knight_actions.add_condition("create_fire",
                                has_ale=True, has_wood=True)
    knight_actions.add_reaction("create_fire",
                                has_ale=False, has_wood=False,
                                has_fire=True)
    
    knight_actions.add_condition("create_poisoned_fenrir",
                                has_fenrir=True, has_wolfsbane=True)
    knight_actions.add_reaction("create_poisoned_fenrir",
                                has_fenrir=False, has_wolfsbane=False,
                                has_poisoned_fenrir=True)
    
    knight_actions.add_condition("create_poisoned_sword",
                                has_cheap_sword=True, has_wolfsbane=True)
    knight_actions.add_reaction("create_poisoned_sword",
                                has_cheap_sword=False, has_wolfsbane=False,
                                has_poisoned_sword=True) 
    ##---------------------##
    knight_ai.set_actions(knight_actions)
    
    # print(knight_ai.action_list.conditions, "\n")
    # print(knight_ai.action_list.reactions, "\n")
    # print(knight_ai.action_list.weights, "\n")
    
    game_ai.add_game_planner(knight_ai)
    game_ai.search()
    
    # print(game_ai.plans)
    game_ai.find_plan(verbose=True)

#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
#########################################################################################################################
