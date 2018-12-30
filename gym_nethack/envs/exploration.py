import os, random
from collections import namedtuple

import numpy as np

from libs import astar

from gym_nethack.nhdata import *
from gym_nethack.misc import verboseprint
from gym_nethack.envs.base import Terminals, Goals, NetHackRLEnv

TurnRec = namedtuple('ExplFoodRec', 'turn_num num_squares_explored calculated_food_level entered_new_room')
ExplRec = namedtuple('ExplRec', 'actions_this_game all_rooms_explored actions_until_all_rooms_explored num_rooms_explored total_num_rooms num_secret_rooms_explored total_num_secret_rooms num_secret_spots_explored total_num_secret_spots turn_records opt_actions')

class NetHackExplEnv(NetHackRLEnv):
    """Environment for NetHack exploration."""
    def __init__(self, nhinfo=None):
        """Initialize NetHack exploration environment.
        
        Args:
            nhinfo: NetHackInfo object to be used (in cases of multiple environments like Level). If None (default), it is created in set_config().
        """
        super().__init__(nhinfo)
        self.records['expl'] = []
                        
        self.abilities = DIRS_DIAG# ['move left', 'move down', 'move right', 'move up']
        self.ability_dirs = DIRS_DIAG # [(0, -1), (1, 0), (0, 1), (-1, 0)]
        self.ability_cmds = DIRS_CMDS #[CMD.DIR.W, CMD.DIR.S, CMD.DIR.E, CMD.DIR.N]
    
    def get_savedir_info_list(self):
        """Get the strings that should form the save directory name."""
        return [
            *super().get_savedir_info_list(),
            self.test_policy.name if self.test_policy is not None else '',
            'secret' if self.secret_rooms else 'nonsecret'
        ]
    
    def set_config(self, proc_id, test_policy=None, num_episodes=200, num_episodes_per_combo=200, max_num_actions_per_episode=5000, dataset='fixed', secret_rooms=False, name='exploration', **args):
        """Set config.
        
        Args:
            proc_id: process ID of this environment, to be matched with the argument passed to the daemon launching script.
            num_episodes: number of total episodes to run for.
            max_num_actions_per_episode: max number of (legal) actions that can be taken in an episode
            dataset: whether the maps are 'fixed' (same set of maps, i.e., same starting RNG seed) or 'random' (always different)
            secret_rooms: whether to enable generation of secret doors & corridors in NetHack maps
            name: used for record folder name
        """
        assert dataset in ['fixed', 'random']
        self.dataset = dataset
        self.secret_rooms = secret_rooms
        self.test_policy = test_policy
        
        super().set_config(proc_id, name=name, max_num_episodes=num_episodes, max_num_actions_per_episode=max_num_actions_per_episode, **args)
        
    def get_game_params(self):
        """Parameters to pass to NetHack on the creation of a new game. (Will be saved in the NH options file.)"""
        
        if self.dataset is 'fixed':
            seed = 1525485787+self.total_num_games
            #if self.dataset == 'test':
            #    seed += self.num_episodes
        elif self.dataset is 'random':
            seed = -1
        
        return {
            'proc_id': self.proc_id,
            'create_items': self.parse_items,
            'secret_rooms': self.secret_rooms,
            'seed': seed
        }
    
    def reset(self):
        """Prepare the environment for a new episode."""
        assert self.policy is not None # should be set in keras-rl/core.py DQNAgent::fit()/test().
        self.policy.reset()
                
        self.explored_rooms = set()

        self.turn_records = []
        self.stop_recording = False
        self.calculated_food_level = 900
        self.actions_until_all_rooms_explored = -1
        self.opt_actions = -1
        
        self.num_discovered_secret_rooms = -1
        self.num_discovered_sdoors_scorrs = -1
        
        self.total_num_rooms = -1
        self.total_sdoors_scorrs = -1
        self.total_secret_rooms = -1 # this one is calculated at episode end in secret greedy policy::end_episode
        
        self.pathfind2_distances = {}
        
        return super().reset()
        
    def process_msg(self, msg, slim_charset=False):
        """Processes the map screen outputted by NetHack."""
        
        super().process_msg(msg, parse_monsters=False)
        
        self.total_num_rooms = self.nh.stats['rooms']
        self.total_sdoors_scorrs = self.nh.stats['sdoor']
        
        if self.nh.in_room() and self.nh.explored_current_room():
            self.mark_room_explored()
        
        self.nh.update_pathfinding_grid()
        
        verboseprint("Rooms:", str(len(self.explored_rooms)) + "/" + str(self.total_num_rooms), self.explored_rooms)
        assert len(self.explored_rooms) <= self.total_num_rooms
    
    def get_status(self, msg):
        """Check if we are done exploring or not."""
        status = Terminals.OK
        goal_reached = None
        if self.policy.done_exploring():
            verboseprint("Success (finished exploring)")
            status = Terminals.SUCCESS
            goal_reached = Goals.SUCCESS
        return status, goal_reached
    
    def end_turn(self):
        """End the current turn, observe map and store a Turn Record. (Turn = observe state & take action.)"""
        self.policy.observe_action()
        self.policy.end_turn()
        
        if not self.stop_recording:
            self.calculated_food_level -= 1 # once per movement, generally
        if self.stop_recording:
            self.total_actions_this_episode -= 1 # since we will be adding +1 in base.py::NetHackRLEnv::take_action()
            
        assert self.nh.num_explored_squares > 0
        self.turn_records.append(TurnRec(self.total_actions_this_episode, self.nh.num_explored_squares, self.calculated_food_level, False)) # TODO: last 2 arguments
    
    def end_episode(self):
        """End the current episode, storing a record about the episode."""
        self.policy.end_episode()
        
        assert len(self.turn_records) > 0
        assert self.total_actions_this_episode > 0
        assert self.total_num_rooms > 0
        
        # Store record.
        self.records['expl'].append(ExplRec(self.total_actions_this_episode, len(self.explored_rooms) == self.total_num_rooms, self.actions_until_all_rooms_explored, len(self.explored_rooms), self.total_num_rooms, self.num_discovered_secret_rooms, self.total_secret_rooms, self.num_discovered_sdoors_scorrs, self.total_sdoors_scorrs, self.turn_records, self.opt_actions))
        
        super().end_episode()
    
    def get_command_for_action(self, action):
        """Return the direction CMD for the given action index."""
        return self.ability_cmds[action]
    
    def pathfind_through_unexplored_to(self, target, initial):
        """A* pathfinding from initial to target, where A* can visit any position that has *NOT* been explored.
        
        Args:
            target: target position to pathfind to.
            initial: position to start pathfinding from. If None, use current player position.
        """
        
        if (initial, target) not in self.pathfind2_distances:
            overwritten_chars = {}
            #for x, y in override_targets:
            #    overwritten_chars[(x, y)] = self.grid[x][y]
            #if override_target_traversability:
            
            inverse_grid = np.array([[1 for j in range(COLNO)] for i in range(ROWNO)])
            for x in range(ROWNO): #self.GRIDWIDTH):
                for y in range(COLNO): #self.GRIDHEIGHT):
                    if self.nh.basemap_char(x, y) == ' ':
                        inverse_grid[x][y] = 0
            
            overwritten_chars[target] = inverse_grid[target[0]][target[1]]
            inverse_grid[target[0]][target[1]] = 0
            #for x, y in override_targets:
            #    self.grid[x][y] = 0
            path = astar.astar(inverse_grid, initial, target, diag=False)
            
            if type(path) is bool:
                #verboseprint("Error: could not pathfind from", initial, "to", target, "! (target on map:", self.nh.map[target[0]][target[1]], " and basemap:", self.nh.base_map[target[0]][target[1]], ")")
                self.pathfind2_distances[(initial, target)] = None
                return None
            path.reverse() # path[0] should be next to start node.
            self.pathfind2_distances[(initial, target)] = path
        
        return self.pathfind2_distances[(initial, target)]
    
    def mark_room_explored(self):
        """Mark the current room as explored by adding its top left corner position to the explored rooms list."""
        if self.stop_recording:
            return
        
        r_i = self.nh.get_room()
        if self.nh.rooms[r_i].top_left_corner not in self.explored_rooms:
            self.explored_rooms.add(self.nh.rooms[self.nh.get_room()].top_left_corner)
