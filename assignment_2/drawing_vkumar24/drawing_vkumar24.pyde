####################################################################################################################
## Spring 2020, NC State University
## Course     : CSC584 - Building Game AI
## Instructor : Dr. Chris Martens
##
## Name       : Vinay Kumar
## UnityID    : vkumar24@ncsu.edu
##
##================== Assignment-2 (Part-1) Solutions ============###################################################
##================== Part #1: Drawing the Game Map ============#####################################################
####################################################################################################################
## Following are the key_locations:
##    castle
##    tar_pit
##    tavern
##    cave
##    tree
## Following are the key_characters:
##    knight
##    king_of_leighra
##    rameses
##    lady_lupa
##    tree_spirit
##    innkeeper
##    blacksmith
####################################################################################################################

import json
width = 640
height = 480
cols = 64
rows = cols*height/width
def setup():
    ## fullScreen()
    size(width, height)
    print(rows, cols)
    background(255)
    with open("../data/data.json", "r") as read_file:
        data = json.load(read_file)
    data_keys = data.keys()
    
    knight_start_pos = data["knight_start"]
    ## print(knight_start_pos)
    fill(0, 255, 0)
    knight = loadImage("../data/knight.png")
    print(knight)
    image(knight, knight_start_pos[0]-25, knight_start_pos[1]-25, 50, 50)
    arc(knight_start_pos[0], knight_start_pos[1], 5, 5, 0, 2*PI, OPEN)
    

    
    
    locations = data["key_locations"]
    # print(len(locations))
    for loc_key in locations.keys():
        path_sprite = "../data/"+loc_key+".png"
        print(path_sprite)
        sprite = loadImage(path_sprite)
        image(sprite, locations[loc_key][0]-25, locations[loc_key][1]-25, 50, 50)
        # print(loc_key)
        # print(locations[loc_key])
        fill(random(0,255), random(0,255), random(0,255), random(0,255))
        arc(locations[loc_key][0], locations[loc_key][1], 5, 5, 0, 2*PI, OPEN)
    
    obstacles = data["obstacles"]
    # print(data["obstacles"].keys())
    for obs_key in obstacles.keys():
        obs = obstacles[obs_key]
        fill(color(217, 217, 217))
        stroke(color(217, 217, 217))
        beginShape()
        for idx in obs:
            vertex(idx[0], idx[1])
        endShape(CLOSE)
    
    
    
    
    
    
    
    
    
    
    
