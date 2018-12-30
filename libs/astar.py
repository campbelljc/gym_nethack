# Author: Christian Careaga (christian.careaga7@gmail.com)
# A* Pathfinding in Python (2.7)
# Please give credit if used

from heapq import *

import numpy

from gym_nethack.misc import distance_pt as heuristic

#def heuristic(a, b):
#    return abs(b[0] - a[0]) + abs(b[1] - a[1]) # manhattan

#todo: if current node is visible (i.e. 0) but not in explored_set, make it cheaper (or make others more expensive)
def astar(array, start, goal, diag=True, explored_set=None):

    if diag:
        neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
    else:
        neighbors = [(0,1),(0,-1),(1,0),(-1,0)]

    closed_set = set() # set of nodes already evaluated
    came_from = {} # set of discovered nodes to be evaluated
    gscore = {start:0} # cost of going from start to start is 0
    fscore = {start:heuristic(start, goal)} # cost from start to goal starts at heuristic estimate
    open_heap = []

    heappush(open_heap, (fscore[start], start)) # add start node to open set along with best guess
    
    while open_heap: # while there are still nodes on the open heap

        current = heappop(open_heap)[1] # get the smallest node in the heap ([1]->node instead of fscore)

        if current == goal: # if reached goal 
            data = []
            while current in came_from: # reconstruct path
                data.append(current)
                current = came_from[current]
            return data # finished

        closed_set.add(current) # add current node to explored set
        for i, j in neighbors: # for each neighbor of the current node
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor) # get cost from start node to n
            if explored_set is not None and neighbor not in explored_set:
                if abs(i) == 1 and abs(j) == 1:
                    tentative_g_score += 0.5
                else:
                    tentative_g_score -= 0.5
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:
                    if array[neighbor[0]][neighbor[1]] == 1: # invalid cell (marked as impassable)
                        continue
                else: # invalid cell (goes past y limits)
                    continue
            else: # invalid cell (goes past x limits)
                continue
                
            if neighbor in closed_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue # explored already and we reached it cheaper than current gscore
                
            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in open_heap]:
                # we didn't discover this neighbor before or we can get to it cheaper than before
                came_from[neighbor] = current # update the parent list
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(open_heap, (fscore[neighbor], neighbor)) # add neighbor to the open set
                
    return False # couldn't find a path

'''Here is an example of using my algo with a numpy array,
   astar(array, start, destination)
   astar function returns a list of points (shortest path)

nmap = numpy.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,0,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,1,1,1,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,0,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,0,1,1,1,1,1,1,1,1,1,1,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,1,1,1,1,1,1,1,1,1,0,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    
print(astar(nmap, (0,0), (10,13)))
'''