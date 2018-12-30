import os, random
from copy import deepcopy
from collections import deque
from itertools import combinations

import ast, math
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from gym_nethack.nhdata import *
from gym_nethack.nhutil import Passage
from gym_nethack.policies.core import ParameterizedPolicy
from gym_nethack.misc import verboseprint, dfs, is_straight_line_adjacent, get_maximal_rectangle, distance_pt

class MapExplorationPolicy(ParameterizedPolicy):
    """Template map exploration policy."""
    def __init__(self, need_full_map=False):
        """Initialize a default map exploration policy (must be subclassed).
        
        Args:
            need_full_map: whether to keep exploring after done_exploring() returns True but before full map is explored. Necessary for statistics purposes to determine how much of the map we missed, if using a policy that doesn't guarantee full exploration (e.g., anything but greedy algorithm)."""
        super().__init__()
        self.NEED_FULL_MAP = need_full_map
        
    def done_exploring(self):
        """Returns a boolean indicating whether to stop exploring the current map (i.e., end the episode)."""
        raise NotImplementedError
    
class GreedyExplorationPolicy(MapExplorationPolicy):
    """Map exploration policy that always visits closest frontier to player until no frontiers remain."""
    name = 'greedy'
    
    def set_config(self, compute_optimal_path=False, get_food=False, show_graph=False, **args):
        """Set config.
        
        Args:
            compute_optimal_path: whether to compute the optimal exploration path after each episode, as detailed in "Exploration with Secret Discovery", J. Campbell & C. Verbrugge, IEEE Transactions on Games, 2018.
            get_food: whether to stop to pick up food in rooms; increases num. of actions taken, but better approximates a real player's exploration action total.
            show_graph: whether to show the room/corridor graph on screen.
        """
        
        self.compute_optimal_path = compute_optimal_path
        self.get_food = get_food
        self.show_graph = show_graph
        
        super().set_config(**args)
        
        if show_graph and not os.path.exists(self.env.savedir+"room_graphs"):
            os.makedirs(self.env.savedir+"room_graphs")
    
    def reset(self):
        """Reset policy state."""
        self.target = None
        self.frontier_list = []
        
        self.passages = []
        self.visited_rooms = []
        self.picked_up_food_positions = set()
        self.in_shop = False
        
        self.target_known = False
        self.explored_all_rooms = False
        self.finished_exploring = False
        
        self.first_room = None
        self.graph = nx.Graph()
        
        self.current_trajectory = deque()
        self.updated_frontiers = True
        self.grid_needs_updating = True
        
        self.exit_graph()
        self.init_graph()
        
        super().reset()
    
    def init_graph(self):
        """Initialize the room/corridor graph (uses matplotlib)."""
        if not self.show_graph:
            return
        # src: http://stackoverflow.com/questions/7229971/2d-grid-data-visualization-in-python
        plt.ion()
        plt.pause(0.0001)
        self.draw_graph()
    
    def draw_graph(self):
        """Draw the room/corridor graph (uses matplotlib)."""
        if not self.show_graph:
            return
        plt.close()
        
        G = self.graph
        pos=nx.get_node_attributes(self.graph, 'pos')
        self.rooms_fig = plt.figure(figsize=(8, 6))
        nx.draw_networkx_edges(self.graph, pos, alpha=0.4)
        nx.draw_networkx_nodes(self.graph, pos, node_size=100, cmap=plt.cm.Reds_r)
        axes = plt.gca()
        axes.set_xlim([0, 80])
        axes.set_ylim([-21, 0])
        plt.savefig(self.env.savedir+"room_graphs/game"+str(self.env.total_num_games)+"_geo.png")
        plt.show()
        
        plt.pause(0.0001)
    
    def exit_graph(self):
        """Close the currently displayed graph."""
        if not self.show_graph:
            return
        plt.close()
    
    def end_turn(self):
        """End the current turn."""
        self.draw_graph()
        
        self.updated_frontiers = False
        
        if not self.explored_all_rooms and self.done_exploring():
            self.explored_all_rooms = True
            if not self.env.stop_recording:
                self.env.actions_until_all_rooms_explored = self.env.total_actions_this_episode    
    
    def end_episode(self):
        """End the current episode."""
        if self.compute_optimal_path:
            assert self.explored_all_rooms
            self.compute_optimal_solution()
        
        super().end_episode()
    
    def process_and_check_target(self, target):
        """Check if the given target is valid.
        
        Args:
            target: position we currently want to visit"""
        
        self.target_known = target in self.frontier_list
        if self.target_known:
            verboseprint("Target known, so removing from expl list")
            self.frontier_list.remove(target)
        return True
    
    def need_new_target(self):
        """Check if we need a new target."""
        return True if self.target == None or self.target == self.env.nh.cur_pos else False
    
    def get_best_target(self, targets, consider_all=False):
        """Find the closest position in the targets list to the player.
        
        Args:
            targets: positions to consider (i.e., frontiers)
            consider_all: used in subclass methods.
        """
        if len(targets) == 0:
            return None
        dists = [len(self.env.nh.pathfind_to(x, override_target_traversability=True)) for x in targets]
        smallest_dist = min(dists)
        smallest_positions = [i for i, j in enumerate(dists) if j == smallest_dist]
        smallest_positions.reverse()
        return targets[smallest_positions[0]]
    
    def mark_explored(self, pos):
        """Add the given position to the explored positions list, and delete it from the frontier list if applicable.
        
        Args:
            pos: position that we want to mark as explored"""
        
        self.env.nh.mark_explored(pos)
        if self.target == pos:
            self.target = None
        
        to_delete = []
        for i, node in enumerate(self.frontier_list):
            if node == pos:
                to_delete.append(i)
        for i in to_delete:
            self.frontier_list.pop(i)
    
    def add_to_frontier_list(self, pos):
        """Add the given position to the frontier list, if we haven't visited it already.
        
        Args:
            pos: position that we want to add to the frontier list"""
        for ex in self.env.nh.explored: # if exit already explored, don't add
            if ex == pos:
                return
        
        for ex in self.frontier_list:
            if ex == pos:
                return
        
        self.frontier_list.append(pos)
    
    def new_passage_from_room_exit(self, room_centroid, exit):
        """Update the passages list with the given room centroid and exit.
        
        Args:
            room_centroid: center of the room associated with the room exit below.
            exit: the room exit that we want to make a passage from."""
        
        for passage in self.passages:
            if exit in passage.positions:
                # the given room exit is already associated with a passage.
                passage.connect_room(room_centroid)
                for room1, room2 in combinations(list(passage.connected_rooms), 2):
                    if not self.graph.has_edge(room1, room2):
                        self.graph.add_edge(room1, room2, visited=True)
                self.draw_graph()
                return False
        
        # exit did not already exist in a passage, so make a new passage.
        self.passages.append(Passage(room_centroid, exit))
        self.passages[-1].connected_room_openings.add((exit))
        return True
    
    def new_corridor(self, corr):
        """Update the passages list with the given corridor position.
        
        Args:
            corr: corridor position"""
        
        # it's not from a room, so it must connect to an already seen passage
        spots_adjacent_to_corr = self.env.nh.get_corridor_exits(pos=corr)
        adjacent_passage_indices = []
        for i, passage in enumerate(self.passages):
            for adjacent_spot in spots_adjacent_to_corr:
                if adjacent_spot in passage.positions:
                    adjacent_passage_indices.append(i)
                    break
        
        if len(adjacent_passage_indices) == 0:
            return False
        
        # add the new position to the first passage found
        first_ind = adjacent_passage_indices.pop(0)
        self.passages[first_ind].add_position(corr)
        
        if len(adjacent_passage_indices) > 0:
            # have to merge some passages.
            for merge_p in adjacent_passage_indices:
                self.passages[first_ind] = Passage.merge(self.passages[first_ind], self.passages[merge_p])
            adjacent_passage_indices.reverse()
            for merge_p in adjacent_passage_indices:
                self.passages.pop(merge_p)
                
            for room1, room2 in combinations(list(self.passages[first_ind].connected_rooms), 2):
                if not self.graph.has_edge(room1, room2):
                    self.graph.add_edge(room1, room2, visited=True)
            self.draw_graph()
        return True
    
    def first_turn_update(self):
        """Special preparation taken on first turn only."""
        self.target = None
        self.env.nh.update_pathfinding_grid()
    
    def compute_optimal_solution(self):
        """Compute the optimal exploration path length, as detailed in "Exploration with Secret Discovery", J. Campbell & C. Verbrugge, IEEE Transactions on Games, 2018. Requires GLNS solver."""
        self.env.nh.update_pathfinding_grid() # update distances between rooms
        
        assert len(self.visited_rooms) == self.env.total_num_rooms
        
        # create clusters
        clusters = [exits for centroid, exits in self.visited_rooms] # one cluster of exits per room
        clusters = [[self.first_room]] + clusters # add centroid of initial room as its own cluster
        starting_cluster_index = 0                # (since we must visit it.)
        if self.env.parse_items and self.get_food: # add each obtained food pos as its own cluster
            clusters.extend([[food_pos] for food_pos in self.picked_up_food_positions])
        dummy_rooms = [[(-1, -1)], [(-2, -2)]]
        clusters.extend(dummy_rooms) # dummy clusters (to transform shortest hamiltonian path > TSP)
        
        nodes = [item for sublist in clusters for item in sublist]
        num_nodes = len(nodes)

        # create matrix of distances between nodes.                
        matrix = [[9999 for k in range(num_nodes)] for j in range(num_nodes)]
        for i, p1 in enumerate(nodes[:-len(dummy_rooms)]):
            for j, p2 in enumerate(nodes[:-len(dummy_rooms)]):
                if i == j: continue
                matrix[i][j] = len(self.env.nh.pathfind_to(p2, initial=p1))
        
        # first dummy room
        for i in range(len(matrix)):
            matrix[i][-2] = 0
            matrix[-2][i] = 0
    
        # second dummy room
        matrix[-1][starting_cluster_index] = 0
        matrix[starting_cluster_index][-1] = 0
        
        if not os.path.exists(self.env.savedir + "/mats"):
            os.makedirs(self.env.savedir + "/mats")
        fname = self.env.savedir + '/mats/matrix' + str(self.env.total_num_games) + '.gtsp'
        with open(fname, 'w') as f:
            f.write("NAME: matrix\nTYPE: GTSP\n")
            f.write("DIMENSION: " + str(num_nodes) + "\nGTSP_SETS: " + str(len(clusters)) + "\n")
            f.write("EDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX\nEDGE_WEIGHT_SECTION\n")
            for mat_row in matrix:
                for i, sq in enumerate(mat_row):
                    f.write(str(sq))
                    if i < len(mat_row)-1:
                        if   100 <= sq < 1000:
                            f.write(" ")
                        elif 10 <= sq < 100:
                            f.write("  ")
                        elif 0 <= sq < 10:
                            f.write("   ")
                        f.write(" ")
                f.write("\n")
            f.write("GTSP_SET_SECTION:\n")
            for i, cluster in enumerate(clusters):
                f.write(str(i+1))
                for exit in cluster:
                    f.write(" " + str(nodes.index(exit)+1))
                f.write(" -1\n")
            f.write("EOF")
        
        path_found = False
        while not path_found:
            output_file = 'tour.txt'
            if os.path.exists(output_file):
                os.remove(output_file)
            os.system("./libs/glns/GLNScmd.jl " + fname + " -output=" + output_file)
            while not os.path.exists(output_file): pass
        
            cost = -1
            tour = None
            with open(output_file, 'r') as f:
                for line in f:
                    if 'Tour Cost' in line:
                        cost = int(line.split(": ")[1])
                    elif 'Tour' in line:
                        tour = ast.literal_eval(line.split(": ")[1])
                        tour = [int(x) for x in tour]
            assert cost > -1
            assert tour is not None
        
            tour.remove(num_nodes) # remove the two dummy nodes
            tour.remove(num_nodes-1) # (indices start at 1)
        
            if tour[0] == starting_cluster_index+1 or tour[-1] == starting_cluster_index+1:
                break
        
        self.env.opt_actions = cost
        os.rename(output_file, self.env.savedir + '/mats/tour' + str(self.env.total_num_games) + '.txt')

    def done_exploring(self):
        """Check if we are done exploring (i.e., if there are no more frontiers, and we are not currently travelling anywhere)."""
        return True if len(self.frontier_list) == 0 and self.need_new_target() and self.env.total_actions_this_episode > 1 else False
    
    def select_action(self, q_values, valid_action_indices):
        """Determine where to move next (greedily)."""
        
        verboseprint("Player: ", self.env.nh.cur_pos, "Target: ", self.target)
        verboseprint("Frontier list: ", self.frontier_list)
        
        if self.env.nh.cur_pos in self.frontier_list:
            self.frontier_list.remove(self.env.nh.cur_pos)
        
        # Use PICKUP command if standing on top of food.
        if self.env.parse_items and self.get_food and self.env.nh.cur_pos in self.env.nh.food_positions:
            verboseprint("Standing on top of food", self.env.nh.cur_pos)
            self.env.nh.food_positions.remove(self.env.nh.cur_pos)
            self.picked_up_food_positions.add(self.env.nh.cur_pos)
            self.mark_explored(self.env.nh.cur_pos)
            return CMD.PICKUP
        
        # Check if we reached the target we were moving towards (or if we don't have a target)
        if self.need_new_target():
            verboseprint("We reached the target (or no target yet)")
            if self.env.total_actions_this_episode == 0: # initial action
                # Take a random action if it's the first move, to get our bearings (literally).
                verboseprint("-> Initial move - taking random action")
                self.first_turn_update()
                random_exit = random.choice(self.env.nh.get_corridor_exits(diag=True))
                dx, dy = random_exit[0] - self.env.nh.cur_pos[0], random_exit[1] - self.env.nh.cur_pos[1]
                return self.env.ability_dirs.index((dx, dy))
            elif self.done_exploring() and not self.env.single:
                if self.env.nh.on_stairs():
                    verboseprint("Going down stairs.")
                    return CMD.DIR.DOWN
                return CMD.WAIT
            
            if self.grid_needs_updating:
                self.env.nh.update_pathfinding_grid()
                self.grid_needs_updating = False
            
            # Get a new target (greedily).
            to_explore = self.get_best_target(self.frontier_list, consider_all=self.finished_exploring)
            
            if to_explore == None:
                verboseprint("-> No targets left")
                if self.NEED_FULL_MAP and not self.finished_exploring:
                    # We finished exploring, but are still missing some of the map
                    # (e.g., if we did not use a greedy algorithm but some approximation)
                    # so continue exploring to check how much we missed, but stop the recording total.
                    self.env.stop_recording = True
                    self.finished_exploring = True
                    verboseprint("Finished exploring, but will continue.")
                    return self.select_action(q_values, valid_action_indices)
                else:
                    # Get ready for a new episode.
                    self.component_search_targets = dict()
                    self.frontier_list = []
                    self.target = None
                    self.good_targets_left = False
                    self.finished_exploring = True
                    return CMD.WAIT
            
            self.target = to_explore
            
            if not self.process_and_check_target(self.target) or self.target == self.env.nh.cur_pos:
                verboseprint("Something wrong with target - trying again")
                return self.select_action(q_values, valid_action_indices) # something was wrong with target so try again
            
            verboseprint("-> New target: ", self.target, ": <", self.env.nh.map[self.target[0]][self.target[1]], ">")
            self.current_trajectory = deque(self.env.nh.pathfind_to(self.target, explored_set=self.env.nh.explored, override_target_traversability=True))
        else:
            if self.env.nh.cur_pos == self.current_trajectory[0]:
                self.current_trajectory.popleft()
        next_square = self.current_trajectory[0]
        
        verboseprint("Next square is", next_square, "to get to", self.target)
        dx, dy = next_square[0] - self.env.nh.cur_pos[0], next_square[1] - self.env.nh.cur_pos[1]
        
        if abs(dx) > 1 or abs(dy) > 1:
            raise Exception("Invalid move: trying to move from " + str(self.env.nh.cur_pos) + " to " + str(self.target) + " (to try to reach" + str(next_square) + ", with trajectory:" + str(self.current_trajectory) + ")")
        
        return self.env.ability_dirs.index((dx, dy))
    
    def observe_action(self):
        """Parse NetHack map."""
        self.in_shop = False
        
        # check if blinded.
        if self.env.nh.stats['blind'] == 'Blind':
            verboseprint("Not observing this round (blinded).")
            return
        
        # check if we are in a shop.
        if self.env.parse_items and self.env.nh.in_room():
            r_i = self.env.nh.get_room()
            room_positions = self.env.nh.rooms[r_i].positions
            
            num_items_in_room = 0
            for pos in room_positions:
                if pos in self.env.nh.item_positions:
                    num_items_in_room += 1
            
            num_player_chars = self.env.nh.rooms[r_i].count_char('@') + (1 if self.env.nh.map[self.env.nh.cur_pos[0]][self.env.nh.cur_pos[1]] != '@' else 0)
            self.in_shop = num_player_chars >= 2 and num_items_in_room >= 1 and 'peaceful' in self.env.nh.top_line
            verboseprint("Num items in room:", num_items_in_room, ", @ chars:", num_player_chars, ", in shop:", self.in_shop)
            if self.in_shop:
                verboseprint("***IN SHOP***")
                for pos in room_positions:
                    if pos in self.frontier_list:
                        self.frontier_list.remove(pos)
                    if pos == self.target:
                        self.target = None
        
        # check if already visited this location.
        if self.env.nh.cur_pos in self.env.nh.explored:
            verboseprint("Not observing this round (already been here).")
            return
        
        self.updated_frontiers = True
        self.grid_needs_updating = True
        self.mark_explored(self.env.nh.cur_pos)
        
        if self.env.nh.in_room():
            r_i = self.env.nh.get_room()
            room_positions = self.env.nh.rooms[r_i].positions
            room_centroid = self.env.nh.rooms[r_i].centroid
            if not self.graph.has_node(room_centroid):
                # New room found.
                
                exits = self.env.nh.rooms[r_i].wall_openings
                verboseprint("In a new room, exits:", exits)
                self.visited_rooms.append((room_centroid, exits))
                
                if self.first_room is None:
                    self.first_room = room_centroid
                
                self.graph.add_node(room_centroid, pos=(room_centroid[1], -room_centroid[0])) # add the room to the graph
                self.draw_graph()
                
                # mark all room spaces as explored
                for pos in room_positions:
                    if self.env.parse_items and self.get_food and pos in self.env.nh.food_positions:
                        if pos not in self.frontier_list and pos != self.env.nh.cur_pos:
                            verboseprint("Adding item ", pos, ": <", self.env.nh.map[pos[0]][pos[1]],">")
                            self.add_to_frontier_list(pos)
                    else:
                        self.mark_explored(pos)
                
                # make a new passage for each exit
                if not self.in_shop:
                    for exit in exits:
                        self.env.nh.grid[exit[0]][exit[1]] = 0
                        if not self.graph.has_node(exit):
                            self.graph.add_node(exit, pos=(exit[1], -exit[0]), visited=True)
                        if self.new_passage_from_room_exit(room_centroid, exit):
                            # we found a new passage (or new exit):
                            self.add_to_frontier_list(exit)
        
        elif self.env.nh.at_room_opening():
            verboseprint("At room opening")
            exits = self.env.nh.get_corridor_exits()
            for exit in exits:
                if self.env.nh.at_room_opening(exit):
                    continue # don't jump from room opening to room opening - wait till we are in the room
                if self.env.nh.basemap_char(*exit) == '#':
                    self.new_corridor(exit)
                self.add_to_frontier_list(exit)
        
        elif self.env.nh.in_corridor():
            verboseprint("In corridor")
            exits = self.env.nh.get_corridor_exits()
            for exit in exits:
                self.new_corridor(exit)
                self.add_to_frontier_list(exit)
        
        else:
            adjacent = self.env.nh.get_chars_adjacent_to(*self.env.nh.cur_pos)
            self.draw_graph()
            if (adjacent.count(' ') + adjacent.count('')) == 3:
                raise Exception("In niche? Adjacent chars: " + str(adjacent))
            raise Exception("Unknown map structure! Player pos: " + str(self.env.nh.cur_pos) + " and adjacent: " + str(adjacent))

class SecretGreedyExplorationPolicy(GreedyExplorationPolicy):
    """Extension of greedy exploration algorithm to support searching for secret doors and corridors."""
    name = 'secgreedy'
    
    def __init__(self):
        """Initialize policy."""
        super().__init__(need_full_map=True)
    
    def set_config(self, **args):
        """Set config."""
        param_abbrvs = ['nspw']
        param_combos = [[0], [1], [5], [10], [15], [20], [25], [30]] # settings for "num searches per wall" parameter
        super().set_config(param_abbrvs=param_abbrvs, param_combos=param_combos, **args)
    
    def get_default_params(self):
        """Get the default parameters for the policy."""
        return [5] # NUM_SEARCHES_PER_WALL
    
    def set_params(self, params):
        """Set the current parameters for the policy.
        
        Args:
            params: policy parameters"""
        assert len(params) == 1
        self.NUM_SEARCHES_PER_WALL = params[0]
    
    def reset(self):
        """Prepare the environment for a new episode."""
        
        self.visited_nodes = set()
        self.visited_room_pos = set()
        
        self.target_is_wall = False
        self.cur_search_targets = []
        self.searched_targets = set()
        
        self.secret_grid = np.array([[0 for j in range(COLNO)] for i in range(ROWNO)]) # 1 -> secret
        
        super().reset()
    
    def end_episode(self):
        """Compute information about how many secret doors/corridors/rooms were discovered and how many were not, in the map for the current episode."""
        super().end_episode()
        
        total_num_rooms = self.env.total_num_rooms # this is taken from the NH bottom line (R: %d)
    
        non_secret_map_positions = dfs(start=self.env.nh.initial_player_pos,
                                        passable_func=lambda x, y: self.secret_grid[x][y] == 0 and self.env.nh.basemap_char(x, y) in PASSABLE_CHARS,
                                        neighbor_func=lambda x, y, diag: self.env.nh.get_neighboring_positions(x, y, diag),
                                        min_neighbors=0, diag=True)
    
        total_nonsecret_rooms = 0
        num_discovered_secret_rooms = 0

        for room in self.visited_room_pos:
            if room in non_secret_map_positions:
                verboseprint("Room",room,"in non secret map positions")
                total_nonsecret_rooms += 1
            else:
                verboseprint("Room",room,"NOT in non secret map positions")
                num_discovered_secret_rooms += 1
    
        total_secret_rooms = total_num_rooms - total_nonsecret_rooms
        percent_secret_discovered = (num_discovered_secret_rooms / total_secret_rooms) if total_secret_rooms > 0 else 1
    
        num_discovered_sdoors_scorrs = 0
        for i in range(ROWNO):
            for j in range(COLNO):
                if self.secret_grid[i][j] == 1:
                    num_discovered_sdoors_scorrs += 1
        percent_secret_sdoors_scorrs_explored = (num_discovered_sdoors_scorrs/self.env.total_sdoors_scorrs) if self.env.total_sdoors_scorrs > 0 else 1
    
        verboseprint("Num secret rooms discovered:", num_discovered_secret_rooms, "and total:", total_secret_rooms)
        verboseprint("Num secret spots discovered:", num_discovered_sdoors_scorrs, "and total:", self.env.total_sdoors_scorrs)
        
        self.env.num_discovered_secret_rooms = num_discovered_secret_rooms
        self.env.total_secret_rooms = total_secret_rooms
        self.env.num_discovered_sdoors_scorrs = num_discovered_sdoors_scorrs
        
    def process_and_check_target(self, target):
        """Check if the given target is valid.
        
        Args:
            target: position we currently want to visit"""
        
        if target not in self.frontier_list: # moving to a wall...
            self.target_is_wall = True
            verboseprint("Chose a wall target (need to search when there): ", target)
            if self.target == self.env.nh.cur_pos:
                return False # restart target choosing
        
        return super().process_and_check_target(target)
    
    def get_best_target(self, targets, consider_all=False):
        """Find the closest position in the targets list to the player, taking into account walls to search at.
        
        Args:
            targets: positions to consider (i.e., frontiers)
            consider_all: used in subclass methods.
        """
        valid_search_targets = [(pos, count) for (pos, count) in self.cur_search_targets if count < self.NUM_SEARCHES_PER_WALL]
        verboseprint("Cur search targets:",valid_search_targets)
        if len(valid_search_targets) > 0: # something to search.
            dists = [len(self.env.nh.pathfind_to(pos, override_target_traversability=True)) for (pos, count) in valid_search_targets]
            smallest_dist = min(dists)
            smallest_positions = [i for i, j in enumerate(dists) if j == smallest_dist]
            smallest_positions.reverse()
            
            best_wall = valid_search_targets[smallest_positions[0]][0]
            verboseprint("Best wall:", best_wall)
            
            # find adjacent open space to wall
            neighboring_positions = self.env.nh.get_neighboring_positions(*best_wall)
            traversable_neighbors = [neighbor for neighbor in neighboring_positions if self.env.nh.basemap_char(*neighbor) in PASSABLE_CHARS and neighbor in self.visited_nodes]
            
            # get traversable neighbor that touches highest amount of walls
            neighboring_wall_counts = []
            for tn in traversable_neighbors:
                neighbors = self.env.nh.get_neighboring_positions(*tn)
                neighboring_wall_counts.append(sum([1 for neighbor in neighbors if neighbor in valid_search_targets]))
            
            return traversable_neighbors[neighboring_wall_counts.index(max(neighboring_wall_counts))]
        
        return super().get_best_target(targets)
    
    def done_exploring(self):
        """Check if finished exploring: including if all search targets are above max num searches per wall."""
        valid_search_targets = [(pos, count) for (pos, count) in self.cur_search_targets if count < self.NUM_SEARCHES_PER_WALL]
        return True if len(valid_search_targets) == 0 and super().done_exploring() else False
    
    def select_action(self, q_values, valid_action_indices):
        """Determine where to move next (greedily)."""
        
        if self.target_is_wall and self.target == self.env.nh.cur_pos:
            verboseprint("*** SEARCHING (we reached target room-space adjacent to wall to be searched)")
            
            verboseprint("cur search targets:", self.cur_search_targets)
            for nx, ny in self.env.nh.get_neighboring_positions(*self.env.nh.cur_pos, diag=True):
                walls = [wall for (wall, count) in self.cur_search_targets]
                cur_char = self.env.nh.basemap_char(nx, ny)
                if (cur_char in WALL_CHARS or cur_char == ' ') and (nx, ny) in walls:
                    wall, count = self.cur_search_targets[walls.index((nx, ny))]
                    self.cur_search_targets[walls.index((nx, ny))] = (wall, count+1)
                    verboseprint("      ****** Increasing count of", nx, ny, "(tile:",cur_char,") (count:",count+1,")")
                    if count+1 >= self.NUM_SEARCHES_PER_WALL:
                        # remove this wall from the cur_search_targets and add to searched_targets
                        self.searched_targets.add((nx, ny))
                        self.cur_search_targets.pop(walls.index((nx, ny)))
                        verboseprint("cur search targets now:", self.cur_search_targets)
                
            self.target = None
            self.target_is_wall = False
            self.target_known = False
            
            return CMD.SEARCH
        
        return super().select_action(q_values, valid_action_indices)
    
    def observe_action(self):
        """Parse NetHack map."""
        
        super().observe_action()

        if self.env.stop_recording:
            return
        
        changed_walls = []
        if self.env.nh.in_room():
            r_i = self.env.nh.get_room()
            room_positions = self.env.nh.rooms[r_i].positions
            room_centroid = self.env.nh.rooms[r_i].centroid
            
            changed_walls = self.env.nh.get_uncovered_doors()
            verboseprint("Changed wall positions:", changed_walls)
                        
            if any(room_pos not in self.visited_nodes for room_pos in room_positions):
                for pos in room_positions:
                    self.visited_nodes.add(pos)
                self.visited_room_pos.add(self.env.nh.rooms[r_i].top_left_corner)
                for coord in self.env.nh.rooms[r_i].wall_positions:
                    is_wall = self.env.nh.basemap_char(*coord) in ['|', '-']
                    if not is_wall or coord in self.searched_targets or coord in [c[0] for c in self.cur_search_targets]:
                        continue
                    
                    neighbors = self.env.nh.get_neighboring_positions(*coord)
                    reject = False
                    for nx, ny in neighbors:
                        if self.env.nh.basemap_char(nx, ny) == ' ':
                            adjacent = self.env.nh.get_chars_adjacent_to(nx, ny)
                            if adjacent.count(' ') < 3:
                                reject = True
                                break
                    if reject:
                        verboseprint("Wall", coord, "has too much around it: ", adjacent)
                        continue
                    
                    self.cur_search_targets.append((coord, 0))
                    #verboseprint("Adding to cur search targets:",coord)
        
        elif self.env.nh.in_corridor() or self.env.nh.at_room_opening():
            if self.env.nh.next_to_dead_end():
                verboseprint("At dead end!")
                neighbors = self.env.nh.get_neighboring_positions(*self.env.nh.cur_pos)
                #verboseprint("Searched targets:",self.searched_targets)
                for x, y in neighbors:
                    if self.env.nh.basemap_char(x, y) == ' ' and (x, y) not in self.searched_targets and (x, y) not in [c[0] for c in self.cur_search_targets]:
                        self.cur_search_targets.append(((x, y), 0))
                        #verboseprint("Adding to cur search targets:",x,y)
            if self.env.nh.cur_pos in self.visited_nodes:
                changed_walls = self.env.nh.get_uncovered_doors()
                verboseprint("Changed wall positions:", changed_walls)
                verboseprint("In corridor/room opening, been here before and", len(changed_walls), "changed walls detected!")
            self.visited_nodes.add(self.env.nh.cur_pos)
        
        if len(changed_walls) > 0:
            self.update_needed = True
            self.grid_needs_updating = True
        
        for exit in changed_walls:
            x, y = exit
            # check that the exit is adjacent to a visited node...
            changed_wall_neighbors = self.env.nh.get_neighboring_positions(x, y, diag=True)
            if not any(neighbor == self.env.nh.cur_pos for neighbor in changed_wall_neighbors):
                verboseprint("The exit",exit,"is not next to our visited nodes")
                continue
            #input("NEW EXIT FOUND")
            if self.env.nh.basemap_char(x, y) in DOOR_CHARS: #and self.new_passage_from_room_exit(room_centroid, exit):
                verboseprint("******",exit, self.env.nh.basemap_char(x, y))
                self.secret_grid[x][y] = 1
                
                if self.env.nh.in_room():
                    r_i = self.env.nh.get_room()
                    room_positions = self.env.nh.rooms[r_i].positions
                    room_centroid = self.env.nh.rooms[r_i].centroid
                    
                    if self.new_passage_from_room_exit(room_centroid, exit):
                        # we found a new passage (or new exit)
                        self.add_to_frontier_list(exit)
                elif self.env.nh.in_corridor() or self.env.nh.at_room_opening():
                    if self.env.nh.basemap_char(*exit) == '#':
                        self.new_corridor(exit)
                    self.add_to_frontier_list(exit)
                                
                # if it was in cur search targets, then remove it
                walls = [wall for (wall, count) in self.cur_search_targets]
                if exit in walls:
                    verboseprint("Removing",exit,"from search targets: ", self.cur_search_targets[walls.index(exit)])
                    self.cur_search_targets.pop(walls.index(exit))
                    self.searched_targets.add(exit)

class OccupancyMapPolicy(GreedyExplorationPolicy):
    """Occupancy map exploration algorithm for NetHack.
    Described in the paper "Exploration with Secret Discovery", J. Campbell & C. Verbrugge, IEEE Transactions on Games, 2018."""
    
    name = 'occmap'
    
    def __init__(self):
        """Initialize policy."""
        
        self.GRIDWIDTH, self.GRIDHEIGHT = ROWNO, COLNO
        self.MAX_MANHATTAN_DIST = max(self.GRIDWIDTH ** 2, self.GRIDHEIGHT ** 2) * 2
        self.SINGLE_PROB = 1/(self.GRIDHEIGHT*self.GRIDWIDTH)
        
        def get_neighbors(x, y, diag=False, keep_outer=False):
            adjacent = []
            for dx, dy in DIRS_DIAG if diag else DIRS:
                px, py = x+dx, y+dy
                if px >= 0 and px < self.GRIDWIDTH and py >= 0 and py < self.GRIDHEIGHT:
                    adjacent.append((px, py))
                elif keep_outer:
                    adjacent.append((-1, -1))
            return adjacent
        
        self.grid_neighbors_with_outer = [[get_neighbors(x, y, keep_outer=True) for y in range(self.GRIDHEIGHT)] for x in range(self.GRIDWIDTH)]
        self.grid_neighbors_with_diag = [[get_neighbors(x, y, diag=True) for y in range(self.GRIDHEIGHT)] for x in range(self.GRIDWIDTH)]
        self.grid_neighbors = [[get_neighbors(x, y) for y in range(self.GRIDHEIGHT)] for x in range(self.GRIDWIDTH)]
        
        super().__init__()
    
    def set_config(self, **args):
        """Set config."""
        param_abbrvs = ['df', 'ef', 'bm', 'mr', 'dmn', 'ptm', 'ptmd', 'ptv', 'fr']
        super().set_config(param_abbrvs=param_abbrvs, **args)
    
    def get_default_params(self):
        """Set the default params for the algorithm, which will be used if not using grid search."""
        return [
            0.65,   # DIFFUSION_FACTOR
            1,      # EVAL_FACTOR
            0.35,   # BORDER_MULTIPLIER
            5,      # MINIMUM_ROOM_SIZE             # at least 3x3 square + 1 wall per side + 1 buffer per side
            7,      # DFS_MIN_NEIGHBORS
            0.35,   # PROB_THRESHOLD_MULTIPLIER     # the higher the threshold, the less frontiers will be visited
            0.45,   # PROB_THRESHOLD_MULTIPLIER_DFS
            False,  # PROB_THRESHOLD_VARIES
            2       # FRONTIER_RADIUS
        ]
    
    def set_params(self, params):
        """Set the current parameters for the policy.
        
        Args:
            params: policy parameters"""
        
        verboseprint("Setting params to", params)
        self.DIFFUSION_FACTOR = params[0]
        self.EVAL_FACTOR = params[1]
        self.BORDER_MULTIPLIER = params[2]
        self.BORDER_PROB = self.SINGLE_PROB*self.BORDER_MULTIPLIER
        self.MINIMUM_ROOM_SIZE = params[3]
        self.DFS_MIN_NEIGHBORS = params[4]
        self.PROB_THRESHOLD_MULTIPLIER = params[5]
        self.PROB_THRESHOLD_MULTIPLIER_DFS = params[6]
        self.PROB_THRESHOLD_VARIES = params[7]
        self.FRONTIER_RADIUS = params[8]
        assert 0 <= self.EVAL_FACTOR <= 1
    
    def reset(self):
        """Prepare for a new episode."""
        
        self.connected_components = []
        self.good_targets = []
        self.distances_to_player = {}
        self.update_needed = True
        self.good_targets_left = True
        self.new_criticals = set()
        self.past_criticals = set()
        
        self.visited_nodes = set()
        self.grid_probs = [[self.SINGLE_PROB for k in range(self.GRIDHEIGHT)] for j in range(self.GRIDWIDTH)]
        self.predicted_grid = np.array([[0 for j in range(COLNO)] for i in range(ROWNO)])
        
        # Reset occupancy map.
        for x in range(self.GRIDWIDTH):
            for y in range(self.GRIDHEIGHT):
                if x in [0, self.GRIDWIDTH-1] or y in [0, self.GRIDHEIGHT-1]:
                    self.grid_probs[x][y] = self.SINGLE_PROB*self.BORDER_MULTIPLIER
        
        super().reset()
    
    def init_graph(self):
        """Initialize the occupancy map graph (uses matplotlib)."""
        
        if not self.show_graph:
            return
            
        # src: http://stackoverflow.com/questions/7229971/2d-grid-data-visualization-in-python
        plt.ion()
        plt.pause(0.0001)
        self.fig = plt.figure()
        self.cmap2 = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', ['black', 'white'], 1024)
        self.im = plt.imshow(list(reversed(self.grid_probs)), interpolation='nearest', cmap = self.cmap2, origin='lower', vmin=0, vmax=self.SINGLE_PROB*1.2, extent=[0, self.GRIDHEIGHT-1, 1, self.GRIDWIDTH], aspect=2.25)
        #self.cb = plt.colorbar(self.im,cmap=self.cmap2)
        plt.axis('off')
        plt.autoscale(False)
        
    def draw_graph(self):
        """Draw the occupancy map graph (uses matplotlib)."""
        
        if not self.show_graph:
            return
                
        prob_threshold = self.get_prob_threshold()
        
        mxs, mys = [], []
        for i, row in enumerate(self.env.nh.map):
            for j, col in enumerate(row):
                if col in MONS_CHARS and (i, j) != self.env.nh.cur_pos and (i, j) not in self.env.nh.item_positions:
                    mxs.append(self.GRIDWIDTH-i)
                    mys.append(j)
        
        #if len(self.connected_components) < 2 or len(self.visited_rooms) < 2:
        #    return
        
        # src: http://stackoverflow.com/questions/12670101/matplotlib-ion-function-fails-to-be-interactive
        plt.clf()
        plt.axis('off')
        
        num_swalls = 0
        swalls = []
        plotted_frontiers = []
        if len(self.connected_components) > 0:
            tgt = [self.target] if self.target is not None else []
            best_component, _, best_frontiers_per_component = self.get_best_frontier(self.good_targets+tgt, self.connected_components, return_all=True)
            assert len(best_frontiers_per_component) == len(self.connected_components)
            
            # plot components and indicate which component has currently been selected                
            colors = cm.rainbow(np.linspace(0, 1, len(self.connected_components)))

            for component, frontier, color in zip(self.connected_components, best_frontiers_per_component, colors):
                if frontier is None: continue
                color = (color[0], color[1], color[2], color[3])
                xs, ys = [self.GRIDWIDTH - p[0] for p in component], [p[1] for p in component]
                
                if self.env.secret_rooms:
                    walls = self.get_walls_near_component(component, True)
                target_is_wall = self.env.secret_rooms and (self.target in self.env.nh.explored) # in walls or self.target in self.dead_end_walls or self.target_is_wall)
                #if target_is_wall:
                #    input("target is wall")
                
                mark = '+' if target_is_wall else 'x'
                alpha = 0.5
                plt.plot(ys, xs, color='none', markeredgecolor=color, markerfacecolor=color, markeredgewidth=1, marker=mark, markersize=10)
                
                wxs, wys = [], []
                if self.env.secret_rooms:
                    for wall, _ in walls:
                        if wall in self.dead_end_walls:
                            wxs.append(self.GRIDWIDTH - wall[0])
                            wys.append(wall[1])
                            #num_swalls += 1
                
                if target_is_wall:
                    swalls.append(([frontier[0][1]], [self.GRIDWIDTH - frontier[0][0]], color, True))
                    num_swalls += 1
                    for wall, _ in walls:
                        if wall not in self.dead_end_walls:
                            wxs.append(self.GRIDWIDTH - wall[0])
                            wys.append(wall[1])
                            num_swalls += 1
                else:
                    fx, fy = self.GRIDWIDTH - frontier[0][0], frontier[0][1]
                    plotted_frontiers.append(frontier[0])
                    plt.plot([fy], [fx], marker='v', color=color, markersize=20)
                swalls.append((wys, wxs, color, False))
                
        else:
            verboseprint("Graph: no components!")
        
        # plot the occ map probabilities in greyscale
        self.im = plt.imshow(list(reversed(self.grid_probs)), interpolation='nearest', cmap = self.cmap2, origin='lower', vmin=0, vmax=self.SINGLE_PROB*1.2, extent=[0, self.GRIDHEIGHT-1, 1, self.GRIDWIDTH], aspect=2.25)
        
        remaining_frontiers = [f for f in self.frontier_list if f not in plotted_frontiers]
        exs, eys = [], []
        eixs, eiys = [], []
        ixs, iys = [], []
        #if num_swalls < 3: return
        
        for wys, wxs, color, selected in swalls:
            plt.plot(wys, wxs, 's', color=color, markersize=5)
            print(list(zip(wxs, wys)))
        
        # plot player position
        px, py = self.env.nh.cur_pos
        self.player_pt = plt.plot([py], [self.GRIDWIDTH-px], marker='o', color='blue', markersize=10)
        
        # plot current target
        if self.target is not None:
            tx, ty = self.target
            self.target_pt = plt.plot([ty], [self.GRIDWIDTH-tx], marker='v', color='blue', markersize=10)
        
        for (ex, ey) in remaining_frontiers:
            if self.env.parse_items and (ex, ey) in self.env.nh.item_positions:
                ixs.append(self.GRIDWIDTH-ex)
                iys.append(ey)
            elif self.good_position((ex, ey), prob_threshold):
                exs.append(self.GRIDWIDTH-ex)
                eys.append(ey)
            else:
                eixs.append(self.GRIDWIDTH-ex)
                eiys.append(ey)
        
        # plot the frontiers and indicate whether they are interesting or not
        self.explore_pts = plt.plot(eys, exs, 'v', color='white', markersize=10)
        self.explore_i_pts = plt.plot(eiys, eixs, 'v', color='white', markersize=10)
        self.explore_item_pts = plt.plot(iys, ixs, 'v', color='cyan', markersize=10)
        self.monster_pts = plt.plot(mys, mxs, 'o', color='magenta', markersize=10)
        
        plt.pause(0.0001)
        save = '' # input("save? (Y/N) >")
        if save == 'Y':
            plt.savefig(self.env.savedir+'/viz.svg', format='svg', dpi=1000)
        elif save == 'r': # get new map
            self.target = None
            self.frontier_list = []
    
    def dfs_threshold_prob(self, start, prob_threshold):
        """Do a DFS on the current unexplored area of the map. Only visit positions that are above the given probability threshold.
        
        Args:
            start: position to start at for the DFS
            prob_threshold: positions must be above this threshold value to be visited by DFS"""
        
        return dfs(start, lambda x, y: self.grid_probs[x][y] > prob_threshold or (x, y) in self.new_criticals, lambda x, y, diag: self.grid_neighbors_with_diag[x][y], self.DFS_MIN_NEIGHBORS, diag=True)
    
    def get_prob_threshold(self, dfs=False):
        """Get the probability threshold.
        
        Args:
            dfs: use different threshold multiplier if we are getting the threshold for DFS search versus threshold for regular component/cell validation."""
        
        threshold_prob = (1/(self.GRIDHEIGHT*self.GRIDWIDTH))
        threshold_prob *= self.PROB_THRESHOLD_MULTIPLIER if not dfs else self.PROB_THRESHOLD_MULTIPLIER_DFS
        
        if self.PROB_THRESHOLD_VARIES:
            total_cells = self.GRIDHEIGHT*self.GRIDWIDTH
            num_high_prob_cells = 0
            for i in range(self.GRIDWIDTH):
                for j in range(self.GRIDHEIGHT):
                    if self.grid_probs[i][j] <= threshold_prob:
                        num_high_prob_cells += 1
            map_exploration_percentage = num_high_prob_cells/total_cells # between 0 and 1
            
            # translate to range 0 to SINGLE_PROB
            mep_norm = self.SINGLE_PROB * map_exploration_percentage
            
            return mep_norm
        else:
            return threshold_prob
        
    def get_connected_components(self, update=False):
        """Get the connected components (in terms of graph theory) of unexplored space in the current NetHack map.
        
        Args:
            update: whether to recalculate or return previously cached components"""
        if not update:
            return self.connected_components
        
        dfs_threshold = self.get_prob_threshold(dfs=True)
        visited_cells = set()
        
        cell_groupings = []
        for x in range(self.GRIDWIDTH):
            for y in range(self.GRIDHEIGHT):
                if ((self.grid_probs[x][y] > dfs_threshold and self.env.nh.basemap_char(x, y) not in PASSABLE_CHARS) or ((x, y) in self.new_criticals)) and (x, y) not in visited_cells:
                    connected_cells = self.dfs_threshold_prob((x, y), dfs_threshold)
                    visited_cells.update(connected_cells)
                    cell_groupings.append(connected_cells)
        
        connected_components = []
        for connected_cells in cell_groupings:
            connected_cells_copy = [c for c in connected_cells]
            
            if len(connected_cells) < 2: continue
            
            rect_cells = []
            contained_criticals = [cr for cr in self.new_criticals if cr in connected_cells]
            if len(contained_criticals) > 0:
                best_ll, best_ur, best_area, positions = get_maximal_rectangle(self.GRIDWIDTH, self.GRIDHEIGHT, contained_criticals)
                connected_components.append(positions)
                connected_cells.difference_update(positions)
            
            found_components = []
            while len(connected_cells) > 2:
                best_ll, best_ur, best_area, positions = get_maximal_rectangle(self.GRIDWIDTH, self.GRIDHEIGHT, connected_cells)
                width, height = best_ll[1] - best_ur[1], best_ur[0] - best_ll[0]
                connected_cells.difference_update(positions)
                if best_area >= self.MINIMUM_ROOM_SIZE**2 and min(width, height) >= self.MINIMUM_ROOM_SIZE:
                    #verboseprint("Component approved:", positions)
                    verboseprint("Comp area:", best_area, ",", width, "x", height, "len positions:", len(positions))
                    rect_cells.extend(list(positions))
                    found_components.append(positions)
            connected_components.extend(found_components)
            if len(found_components) == 0 and len(connected_cells_copy) >= self.MINIMUM_ROOM_SIZE**2:
                connected_components.append(connected_cells_copy)
        
        #Component = namedtuple('Component', 'cells disjoint best_frontier search_targets')
        #for comp in connected_components:
        #    verboseprint("Component:", comp)
        #    assert len(comp) > 0
        return connected_components
    
    def get_dist_to_component(self, component, position):
        """Get the Manhattan distance from the given position to the closest cell of the given component.
        
        Args:
            component: list of positions (tuples)
            position: tuple representing position"""
        
        best_dist = self.MAX_MANHATTAN_DIST
        best_cell = (-1, -1)
        for cell in component:
            dist = distance_pt(cell, position)
            if dist < best_dist:
                best_dist = dist
                best_cell = cell
        return best_dist, best_cell
    
    def get_frontier_near_component(self, component, frontiers, frontier_dists_to_player):
        """Get the frontier closest to both the given component and to the player.
        
        Args:
            component: list of positions (tuples)
            frontiers: list of frontiers to evaluate
            frontier_dists_to_player: distance to player for each frontier"""
        
        if len(frontiers) == 0: return None
        
        # check, for each frontier, if it is dfs-adjacent to the component
        # return best and total player-frontier-component distances.
        
        dists = []
        closest_cells = []
        for i, frontier in enumerate(frontiers):
            dist_frontier_cell, closest_cell = self.get_dist_to_component(component, frontier)
            closest_cells.append(closest_cell)
            if any(cr in component for cr in self.new_criticals):
                verboseprint("F", frontier, "cc", closest_cell, "CRIT. dist:", dist_frontier_cell)
                dists.append(dist_frontier_cell)
                continue
            
            # check if we can go in straight line.
            path = self.env.pathfind_through_unexplored_to(closest_cell, initial=frontier)
            
            if path is None or len(path) > 20:
                verboseprint("F", frontier, "cc", closest_cell, "path:", "none" if path is None else "length: " + str(len(path)))
                dists.append(math.inf)
                continue
            
            straight = is_straight_line_adjacent(frontier, path)
            if not straight:
                verboseprint("F", frontier, "cc", closest_cell, "path not straight:", path)
                dists.append(math.inf)
            else:
                verboseprint("F", frontier, "cc", closest_cell, "path good. d-player:", frontier_dists_to_player[i], "d-comp:", len(path))
                dists.append(frontier_dists_to_player[i]) # + len(path))
                
        smallest_dist = min(dists)
        verboseprint("BF:", frontiers[dists.index(smallest_dist)], "cc", closest_cells[dists.index(smallest_dist)], "(dist:", smallest_dist, ")")
        if smallest_dist == math.inf:
            return None
        
        return frontiers[dists.index(smallest_dist)], smallest_dist
    
    def get_evaluation_for_cells(self, component, frontier, sum_dists, sum_probs):
        """Evaluate component based on distance from closest frontier node to player and summed cell probability.
        
        Args:
            component: list of cells to evaluate
            frontier: tuple of (position, distance)
            sum_dists: sum of distances between player and frontiers associated with each valid component
            sum_probs: sum of cell probabilities for all valid components"""
        
        if frontier is None:
            return -math.inf
        
        frontier_pos, dist = frontier
        
        if any(cr in component for cr in self.new_criticals):
            return math.inf
        
        prob = sum([self.grid_probs[x][y] for x, y in component])
        norm_prob = prob / sum_probs # bigger val better
        if norm_prob > 1.01:
            raise Exception("NORM PROB > 1: " + norm_prob + "," + sum_prob + "," + total_prob)
        
        norm_dist = dist / (sum_dists+1) # smaller val better
        if norm_dist > 1:
            raise Exception("NORM DIST > 1: ", norm_dist + "," + dist + "," + sum_dists)
        
        return ((1 - self.EVAL_FACTOR)*norm_prob) + (self.EVAL_FACTOR*(1 - norm_dist))
    
    def get_distance_to_player(self, target):
        """Get distance to player, from cache if available.
        
        Args:
            target: position (tuple)"""
        if target not in self.distances_to_player:
            self.distances_to_player[target] = len(self.env.nh.pathfind_to(target, override_target_traversability=True))
        if self.env.parse_items and target in self.env.nh.item_positions:
            return (self.distances_to_player[target])/4
        return self.distances_to_player[target]
    
    def get_best_frontier(self, good_targets, connected_components, return_all=False):
        """Find the best component, and then find the best frontier associated with it.
        
        Args:
            good_targets: frontiers that have been evaluated and passed utility check
            connected_components: list of components
            return_all: whether to return best frontier for all components (to show on graph), or just best frontier for best component"""
        frontier_dists_to_player = [self.get_distance_to_player(frontier) for frontier in good_targets]
        best_frontiers = [self.get_frontier_near_component(component, good_targets, frontier_dists_to_player) for component in connected_components]
        
        verboseprint("Best frontiers:", best_frontiers)
        
        sum_dists = sum([f[1] for f in best_frontiers if f is not None])
        
        # use evaluation function to determine best component (by max value)
        best_component = None
        best_eval = -math.inf
        for component, frontier in zip(connected_components, best_frontiers):
            if frontier is None: continue
            
            def get_sum_stats(best_frontiers, connected_components):
                return [sum(map(sum, self.grid_probs))]
            
            eval_val = self.get_evaluation_for_cells(component, frontier, sum_dists, *get_sum_stats(best_frontiers, connected_components))
            if eval_val > best_eval:
                best_eval = eval_val
                best_component = (component, frontier)
        
        #verboseprint("Component eval vals: ", eval_vals)
        
        if best_eval == -math.inf:
            return (None, None, best_frontiers) if return_all else None
        
        if return_all:
            return best_component[0], best_component[1][0], best_frontiers
        return best_component[1][0]
    
    def good_position(self, pos, prob_threshold):
        """Check if frontier is interesting enough to visit.
        
        Args:
            prob_threshold: probability threshold value"""
                
        #if self.FRONTIER_CHECK_TYPE == 0:
        #    #gx, gy = pos #self.map_to_grid(*pos, float=True)
        #    cells = self.get_surrounding_cells(gx, gy)
        #    if self.FRONTIER_RADIUS_EXTEND:
        #        for fx, fy in [(gx+1, gy), (gx-1, gy), (gx, gy+1), (gx, gy-1)]:
        #            cells.extend(self.get_surrounding_cells(fx, fy))
        #        cells = list(set(cells))
        
        if self.env.parse_items and pos in self.env.nh.item_positions:
            return True # must go to item
        
        if pos in self.new_criticals:
            return True # part of a room we have observed but not yet visited, so definitely visit it.
        
        positions_to_surround = set([pos])
        for i in range(self.FRONTIER_RADIUS): # if FR>0, get the neighbors of (gx, gy) so we analyze their neighbors later
            positions = deepcopy(positions_to_surround)
            for x, y in positions:
                for dx, dy in DIRS_DIAG:
                    positions_to_surround.update([(x+dx, y+dy)])
        
        cells = set() # get surrounding positions
        for x, y in positions_to_surround:
            for dx, dy in DIRS:
                if self.env.nh.in_range(x+dx, y+dy):
                    cells.update([(x+dx, y+dy)])
        
        if any(c in self.new_criticals for c in cells):
            return True # part of a room we have observed but not yet visited, so definitely visit it.
        
        cell_probs = [self.grid_probs[x][y] for x, y in cells]
        if any(prob > prob_threshold for prob in cell_probs):
            return True # if one of the neighboring cells has a prob. higher than the threshold, visit it.
        
        return False
    
    def no_more_targets(self, targets):
        """Check if there are any more targets left.
        
        Args:
            list of frontiers"""
        return True if len(targets) == 0 else False
    
    def first_turn_update(self):
        """Special preparation taken on first turn only."""
        self.target = None
        self.update_caches(self.frontier_list)
        self.env.nh.update_pathfinding_grid()
    
    def update_caches(self, targets, prob_threshold=None):
        """Update validated frontier and component caches.
        
        Args:
            targets: current list of frontiers
            prob_threshold: probability threshold value"""
        verboseprint("Updating caches...")
        if prob_threshold is None:
            prob_threshold = self.get_prob_threshold()
        self.good_targets = [p for p in targets if self.good_position(p, prob_threshold)]
        self.distances_to_player.clear()
        self.connected_components = self.get_connected_components(update=True)
    
    def normalize_and_diffuse(self, p_culled):
        """Normalize occupancy map probabilities and run diffusion as described by D. Isla."""
        
        # for all n in Visited, P(t)(n) = 0
        for gx, gy in self.visited_nodes:
            self.grid_probs[gx][gy] = 0
        for gx, gy in self.env.nh.concrete_positions:
            self.grid_probs[gx][gy] = 0
        
        # for all n in Hidden, P(t)(n) = P(t-1)(n)/(1 - Pculled)
        if p_culled > 0:
            for x in range(self.GRIDWIDTH):
                for y in range(self.GRIDHEIGHT):
                    if (x, y) in self.visited_nodes:
                        continue # we only want hidden nodes, not visited
                    self.grid_probs[x][y] = self.grid_probs[x][y] / (1 - p_culled)
        
        # diffusion step:
        # P(t)(n) = (1-A) * P(t)(n) + (A/4) * sum over all n' neighbors(n): P(t)(n')
        old_probs = deepcopy(self.grid_probs)
        for x in range(1, self.GRIDWIDTH-1):
            for y in range(1, self.GRIDHEIGHT-1):
                if (x, y) in self.env.nh.concrete_positions: continue
                if (x, y) in self.new_criticals: continue
                neighbor_probs = [old_probs[i][j] for i, j in self.grid_neighbors[x][y]]
                #neighbor_probs = [old_probs[i][j] if (i, j) != (-1, -1) else self.SINGLE_PROB*self.BORDER_MULTIPLIER for i, j in self.grid_neighbors_with_outer[x][y]] # if (i, j) not in self.new_criticals]
                self.grid_probs[x][y] = ((1 - self.DIFFUSION_FACTOR)*old_probs[x][y]) + (self.DIFFUSION_FACTOR/len(neighbor_probs))*sum(neighbor_probs)
        
        for gx, gy in self.visited_nodes:
            self.grid_probs[gx][gy] = 0
    
    def get_best_target(self, targets, consider_all=False):
        """Find the closest position in the targets list to the player.
        
        Args:
            targets: positions to consider (i.e., frontiers)
            consider_all: whether to consider all frontiers, or just ones that have passed the utility check (good_position())
        """
        if consider_all:
            return super().get_best_target(targets) # use greedy alg
        elif self.env.parse_items and self.env.nh.in_room():
            item_targets = [t for t in targets if t in self.env.nh.item_positions and t in self.env.nh.rooms[self.env.nh.get_room()].positions]
            if len(item_targets) > 0:
                # get closest target
                dist = (1000, None)
                for t in item_targets:
                    t_dist = max(abs(self.env.nh.cur_pos[0]-t[0]), abs(self.env.nh.cur_pos[1]-t[1]))
                    if t_dist < dist[0]:
                        dist = (t_dist, t)
                return dist[1]
        
        prob_threshold = self.get_prob_threshold()
        if self.update_needed:
            self.update_caches(targets, prob_threshold) # updates self.good_targets, self.connected_components
            self.update_needed = False
        if self.no_more_targets(self.good_targets):
            verboseprint("No more good targets!")
            return None
                
        connected_components = self.get_connected_components()
        if len(connected_components) == 0:
            verboseprint("No more components!")
            return None
        frontier = self.get_best_frontier(self.good_targets, connected_components)
        if frontier is None:
            verboseprint("No applicable frontier to best component!")
            return None
        return frontier
    
    def done_exploring(self):
        """Returns a boolean indicating whether to stop exploring the current map (i.e., end the episode)."""        
        return True if (len(self.frontier_list) == 0 or not self.good_targets_left) and self.need_new_target() and self.env.total_actions_this_episode > 1 else False
    
    def observe_action(self):
        """Parse NetHack map and update occupancy map accordingly."""
        
        self.in_shop = False
        
        # check if blinded.
        if self.env.nh.stats['blind'] == 'Blind':
            verboseprint("Not observing this round (blinded).")
            return
        
        # check if we are in a shop.
        if self.env.parse_items and self.env.nh.in_room():
            r_i = self.env.nh.get_room()
            room_positions = self.env.nh.rooms[r_i].positions
            
            num_items_in_room = 0
            for pos in room_positions:
                if pos in self.env.nh.item_positions:
                    num_items_in_room += 1
            
            num_player_chars = self.env.nh.rooms[r_i].count_char('@') + (1 if self.env.nh.map[self.env.nh.cur_pos[0]][self.env.nh.cur_pos[1]] != '@' else 0)
            self.in_shop = num_player_chars >= 2 and num_items_in_room >= 1 and 'peaceful' in self.env.nh.top_line
            verboseprint("Num items in room:", num_items_in_room, ", @ chars:", num_player_chars, ", in shop:", self.in_shop)
            if self.in_shop:
                verboseprint("***IN SHOP***")
                for pos in room_positions:
                    if pos in self.frontier_list:
                        self.frontier_list.remove(pos)
                    if pos == self.target:
                        self.target = None
        
        # check if already visited this location.
        if self.env.nh.cur_pos in self.env.nh.explored:
            verboseprint("Not observing this round (already been here).")
            return
                
        self.updated_frontiers = True
        self.grid_needs_updating = True
        self.mark_explored(self.env.nh.cur_pos)
        
        x, y = self.env.nh.cur_pos
        adjacent = self.env.nh.get_chars_adjacent_to(x, y)
        
        self.new_criticals.update([pos for pos in self.env.nh.critical_positions if pos not in self.past_criticals])
        
        if self.env.nh.at_room_opening():
            verboseprint("At room opening")
            if any(shkname in self.env.nh.top_line for shkname in SHOPKEEPER_NAMES):
                verboseprint("Entrance to shop - skipping observe")
                
            else:
                exits = self.env.nh.get_corridor_exits(diag=False)
                verboseprint("Exits detected:", exits)
                for exit in exits:
                    if self.env.nh.at_room_opening(exit):
                        verboseprint("Skipping exit", exit, "(it's also a RO)")
                        continue # don't jump from room opening to room opening - wait till we are in the room
                    if self.env.nh.basemap_char(*exit) == '#':
                        self.new_corridor(exit)
                    if self.env.nh.basemap_char(*exit) == '@': # likely a shop
                        verboseprint("Skipping exit", exit, "(shopkeeper)")
                        continue
                    #verboseprint("Adding exit(2): <", self.env.nh.map[exit[0]][exit[1]],">")
                    self.add_to_frontier_list(exit)
        
        elif self.env.nh.in_corridor():
            verboseprint("In corridor")
            exits = self.env.nh.get_corridor_exits()
            for exit in exits:
                self.new_corridor(exit)
                #verboseprint("Adding exit(3): <", self.env.nh.map[exit[0]][exit[1]],">")
                self.add_to_frontier_list(exit)
        
        elif self.env.nh.in_room():
            r_i = self.env.nh.get_room()
            room_positions = self.env.nh.rooms[r_i].positions
            room_centroid = self.env.nh.rooms[r_i].centroid
            
            if not self.graph.has_node(room_centroid):
                exits = self.env.nh.rooms[r_i].wall_openings
                verboseprint("In a new room, exits:", exits)
                self.visited_rooms.append((room_centroid, exits))
                self.past_criticals.update(room_positions)
                self.past_criticals.update(exits)
                self.past_criticals.update(self.env.nh.rooms[r_i].wall_positions)
                self.past_criticals.update(self.env.nh.rooms[r_i].corners)
                self.new_criticals.difference_update(self.past_criticals)
                self.env.nh.concrete_positions.difference_update(self.env.nh.rooms[r_i].wall_positions)
                
                if self.first_room is None:
                    self.first_room = room_centroid
                
                self.graph.add_node(room_centroid, pos=(room_centroid[1], -room_centroid[0])) # add the room to the graph
                self.draw_graph()
                
                # mark all room spaces as explored
                for pos in room_positions:
                    if self.env.parse_items and self.get_food and pos in self.env.nh.food_positions:
                    #if self.parse_items and not self.in_shop and pos in self.env.nh.item_positions:
                        if pos not in self.frontier_list and pos != self.env.nh.cur_pos:
                            verboseprint("Adding item ", pos, ": <", self.env.nh.map[pos[0]][pos[1]],">")
                            self.add_to_frontier_list(pos)
                    else:
                        self.mark_explored(pos)
                
                if len(exits) >= 8:
                    verboseprint("High num. of exits...:", exits)
                    #input("")
                
                # make a new passage for each exit
                if not self.in_shop:
                    for exit in exits:
                        self.env.nh.grid[exit[0]][exit[1]] = 0
                        if not self.graph.has_node(exit):
                            self.graph.add_node(exit, pos=(exit[1], -exit[0]), visited=True)
                        if self.new_passage_from_room_exit(room_centroid, exit):
                            # we found a new passage (or new exit):
                            #verboseprint("Adding exit(1): <", self.env.nh.map[exit[0]][exit[1]],">")
                            self.add_to_frontier_list(exit)
        
        else:
            adjacent = self.env.nh.get_chars_adjacent_to(*self.env.nh.cur_pos)
            verboseprint(adjacent)
            if (adjacent.count(' ') + adjacent.count('') + adjacent.count('-') + adjacent.count('|')) == 3 and adjacent.count('.') == 1:
                verboseprint("In niche")
                sys.exit(1)
            else:
                self.draw_graph()
                raise Exception("???? Player pos:", self.env.nh.cur_pos)
        
        self.new_criticals.difference_update(self.visited_nodes)
        
        p_culled = 0
        for (rx, ry) in self.new_criticals:
            p_culled += self.grid_probs[rx][ry] - (self.SINGLE_PROB*1.2)
            self.grid_probs[rx][ry] = (self.SINGLE_PROB*1.2)
        
        if self.env.nh.in_room():
            r_i = self.env.nh.get_room()
            for rx, ry in self.env.nh.rooms[r_i].positions:
                if (rx, ry) not in self.visited_nodes:
                    p_culled += self.grid_probs[rx][ry]                
                    self.visited_nodes.add((rx, ry))
                    self.env.nh.base_map[rx][ry] = '.'
                    #input("")
        
        elif self.env.nh.in_corridor() or self.env.nh.at_room_opening():
            gx, gy = self.env.nh.cur_pos
            if (gx, gy) not in self.visited_nodes:
                p_culled = self.grid_probs[gx][gy]
                self.visited_nodes.add((gx, gy))
        
        else:
            raise Exception("Couldn't determine location type.")
        
        if p_culled != 0: # we changed the probability of one or more cells in the occupancy map
            self.update_needed = True
            self.normalize_and_diffuse(p_culled)
        
        if self.updated_frontiers:
            self.update_needed = True
    