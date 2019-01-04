import os #, time
from copy import deepcopy

import numpy as np
import gym, zmq, dill
from gym import utils, spaces

from libs import astar

from gym_nethack.conn import *
from gym_nethack.nhutil import *
from gym_nethack.nhdata import *
from gym_nethack.misc import VERBOSE
#from gym_nethack.nhdaemon import spawn_daemon

class Terminals: OK, PLAYER_DIED, MONSTER_DIED, IMPOSSIBLE_ACTION, TIME_EXCEEDED, CONN_ERROR, SUCCESS = range(0, 7)
class Goals: SUCCESS, LOSS, TIME_EXCEEDED, CONN_ERROR = range(0, 4)

class NetHackInfo(object):
    """Stores NetHack game state, and contains methods for processing/parsing screen output and for item & map information."""
    def __init__(self, parse_items=True):
        """Initialize info object.
        
        Args:
            parse_items: whether to keep or discard items in the parsed NetHack map
        """
        super().__init__()
        self.parse_items = parse_items
    
    def reset(self):
        """Reset all map- and level-dependent variables."""
        self.observed = False
        
        self.rooms = []
        self.room_openings = set()
        self.corridors = set()
        self.map = None
        self.base_map = None
        self.top_line = ""
        self.num_explored_squares = 0
        self.pathfind_distances = {}
        
        self.explored = set()
        self.grid = np.array([[1 for j in range(COLNO)] for i in range(ROWNO)]) # 1 -> impassable
        
        self.initial_player_pos = None
        self.prev_prev_pos = None
        self.prev_pos = None
        self.cur_pos = None
        self.stats = {}
        self.attributes = {}
        self.player_has_lycanthropy = False
        self.in_fog = False
        
        self.monster_positions = []
        self.critical_positions = []
        self.concrete_positions = []
        
        if self.parse_items:
            self.inventory = []
            self.ammo_positions = []
            self.item_positions = set()
            self.food_positions = set()
    
    def process_msg(self, socket, message, update_base=True, parse_monsters=True, parse_ammo=False):
        """Processes the map screen outputted by NetHack.
        
        Args:
            socket: the socket connected to the NetHack process (needed to send/rcv inventory message)
            message: the message outputted by NetHack
            update_base: whether to update our record of the map with new information gleaned or not
            parse_monsters: whether to keep or discard monsters in the parsed NetHack map
            parse_ammo: whether to keep or discard ammo in the parsed NetHack map
        """
        self.prev_prev_pos = self.prev_pos
        self.prev_monster_positions = deepcopy(self.monster_positions)
        self.prev_map = self.map
        self.prev_pos = self.cur_pos
        
        self.base_map, self.map, attmsg, sttmsg, self.top_line, self.cur_pos, self.monster_positions, self.ammo_positions, new_items, new_food, self.back_glyph, self.critical_positions, self.concrete_positions, self.num_explored_squares = unpack_msg(message, self.base_map, parse_ammo=parse_ammo, update_base=update_base, parse_monsters=parse_monsters)
        if VERBOSE:
            # only call verboseprint if VERBOSE specified to omit computation time of join() call
            verboseprint(''.join(item for innerlist in self.base_map for item in innerlist))
            verboseprint(''.join(item for innerlist in self.map for item in innerlist))
            verboseprint("Top: " + self.top_line)
        
        if self.parse_items:
            self.len_prev_inventory = len(self.inventory)
            self.inventory = get_inventory(socket)
            self.equipped_armor_types = []
            self.num_equipped_rings = 0
            for inven_item, _, _, matched_item, _ in self.inventory:
                if 'being worn' in inven_item:
                    self.equipped_armor_types.append(matched_item.type)
                if 'hand' in inven_item and 'ring' in inven_item:
                    self.num_equipped_rings += 1
            
            self.item_positions.update(new_items)
            self.food_positions.update(new_food)

        if self.back_glyph in ROOM_OPENING_GLYPHS:
            self.room_openings.add((self.cur_pos))
        elif self.back_glyph in CORRIDOR_GLYPHS:
            self.corridors.add((self.cur_pos))
        
        if not self.observed:
            self.observed = True
            self.prev_pos = self.cur_pos
            self.initial_player_pos = self.cur_pos
        
        self.prev_attributes, self.attributes = update_attrs(attmsg, self.attributes)
        self.prev_stats, self.stats = update_stats(sttmsg, self.stats)
                
        if 'Were' in self.attributes['role_title'] or 'feel feverish' in self.top_line:
            self.player_has_lycanthropy = True
        elif self.player_has_lycanthropy and 'feel purified' in self.top_line:
            self.player_has_lycanthropy = False # TODO: other methods of removing
        
        if 'laden with moisture' in self.top_line or self.count_char_on_map('.') == 0:
            self.in_fog = True
        if self.in_fog and ('destroy the fog' in self.top_line or self.count_char_on_map('.') > 3):
            self.in_fog = False
    
    def get_cur_weapon(self):
        """Returns the current weapon object wielded by the player, and whether it is cursed or not."""
        for inven_item, _, _, weap_obj, _ in self.inventory:
            if wielding(inven_item):
                cursed = 'cursed ' in inven_item
                return weap_obj, cursed
        return None, False
    
    def get_inven_char_for_item(self, item):
        """Returns the inventory character mapped to a particular item, assuming the player has it in the inventory."""
        for _, inven_char, stripped_inven_item, _, _ in self.inventory:
            if item_match(item.full_name, stripped_inven_item):
                return inven_char
        raise Exception("Couldn't match" + str(item) + "to anything in inventory: \n" + str(self.inventory))
        
    def in_range(self, x, y):
        """Returns true if the given x,y coordinate is within the map bounds."""
        return x >= 0 and x < ROWNO and y >= 0 and y < COLNO
    
    def basemap_char(self, x, y):
        """Returns the basemap character at the given map coords."""
        return self.base_map[x][y] if self.in_range(x, y) else ''
    
    def char_under_player(self):
        """Returns the character under the player."""
        x, y = self.cur_pos
        assert x != -1 and y != -1
        return self.base_map[x][y]
    
    def get_room(self):
        """Returns the list index for the current room object, creating one if necessary."""
        for i, room in enumerate(self.rooms):
            if self.cur_pos in room.positions:
                return i
        # room does not yet exist.
        self.rooms.append(Room(self))
        return -1
    
    def get_uncovered_doors(self):
        """Return the coordinates which in the last turn were revealed to be doors."""
        if self.prev_map is None: return []
        walls = []
        for i, row in enumerate(self.base_map):
            for j, cur_char in enumerate(row):
                old_char = self.prev_map[i][j]
                if cur_char != old_char and cur_char in DOOR_CHARS and old_char != '@':
                    walls.append((i, j))
        return walls
    
    def get_corridor_exits(self, pos=None, diag=True):
        """Return the corridors adjacent to the given position.
        
        Args:
            pos: the position around which to look for corridors. If none, the player's current position is used.
            diag: whether to consider diagonal tiles
        """
        if pos == None:
            pos = self.cur_pos
        x, y = pos
        dirs = DIRS_DIAG if diag else DIRS
        exits = []
        for dx, dy in dirs:
            if dx == 0 and dy == 0: continue
            if self.basemap_char(x+dx, y+dy) in PASSABLE_CHARS:
                exits.append((x+dx, y+dy))
        return exits
    
    def get_chars_adjacent_to(self, x, y, diag=False):
        """Returns the list of basemap tiles adjacent to the given map coordinate."""
        adjacent = [self.basemap_char(x-1, y), self.basemap_char(x+1, y), self.basemap_char(x, y-1), self.basemap_char(x, y+1)]
        if diag:
            adjacent.extend([self.basemap_char(x-1, y-1), self.basemap_char(x-1, y+1), self.basemap_char(x+1, y-1), self.basemap_char(x+1, y+1)])
            
        adjacent = list(filter(((-1, -1)).__ne__, adjacent))
        return adjacent
    
    def get_neighboring_positions(self, x, y, diag=True):
        """Returns the list of in-range coordinates adjacent to the given map coordinate."""
        dirs = DIRS_DIAG if diag else DIRS
        return [(x+dx, y+dy) for dx, dy in dirs if self.in_range(x+dx, y+dy)]
    
    def count_char_on_map(self, char):
        """Return the number of appearances of the given char on the map."""
        c = 0
        for i, row in enumerate(self.map):
            for j, col in enumerate(row):
                if col == char:
                    c += 1
        return c
    
    def find_char_on_base_map(self, char):
        """Return the map coordinate at which the first instance of the given char appears, or None if it does not."""
        for i, row in enumerate(self.base_map):
            for j, col in enumerate(row):
                if col == char:
                    return (i, j)
        return None
    
    def on_stairs(self):
        """Return true if the player is standing on top of the staircase."""
        return True if '>' in self.char_under_player() or 'stair' in self.top_line else False
    
    def in_room(self):
        """Return true if the player is in a room."""
        x, y = self.cur_pos
        adjacent = self.get_chars_adjacent_to(x, y)
        return True if (self.char_under_player() in ROOM_CHARS and (adjacent.count('.') + adjacent.count('>') + adjacent.count('<') + adjacent.count('^')) >= 2 and self.back_glyph not in ROOM_OPENING_GLYPHS) or adjacent.count('.') == 4 else False
    
    def in_corridor(self):
        """Return true if the player is in a corridor."""
        x, y = self.cur_pos
        adjacent = self.get_chars_adjacent_to(x, y)
        return True if self.cur_pos in self.corridors or (self.char_under_player() in CORRIDOR_CHARS and (adjacent.count('#') + adjacent.count('`') + adjacent.count(' ') + adjacent.count('^')) >= 1) or (adjacent.count('#') + adjacent.count(' ') == 4) else False # or (self.char_under_player() == '.' and (adjacent.count('#') + adjacent.count('`') + adjacent.count(' ') + adjacent.count('^')) >= 2) else False
    
    def at_intersection(self):
        """Return true if the player is at the intersection of two or more corridors."""
        x, y = self.cur_pos
        adjacent = self.get_chars_adjacent_to(x, y, diag=True)
        return True if self.in_corridor() and (adjacent.count('#') + adjacent.count('`') + adjacent.count('^')) > 2 else False
    
    def at_room_opening(self, pos=None):
        """Return true if the player (or the given position) is at a room opening."""
        if pos == None:
            pos = self.cur_pos
        #return True if self.back_glyph in ROOM_OPENING_GLYPHS else False
        return True if pos in self.room_openings or (self.basemap_char(*pos) in ['#', '.', '+'] and self.get_chars_adjacent_to(*pos).count('|') + self.get_chars_adjacent_to(*pos).count('-') == 2) else False
    
    def next_to_dead_end(self):
        """Return true if the player (or the given position) is at a dead-end in a corridor."""
        x, y = self.cur_pos
        adjacent = self.get_chars_adjacent_to(x, y)
        if self.in_corridor():
            # count up the number of adjacent traversable squares.
            adjacent_traversable_squares = adjacent.count('#') + adjacent.count('+') + adjacent.count('.')
            return adjacent_traversable_squares <= 1
            #and adjacent.count('#') <= 1 and (adjacent.count(' ') == 3 or (adjacent.count('|') + adjacent.count('-') + adjacent.count('.') + adjacent.count('+')) == 0):
            #return True
        elif self.at_room_opening() and adjacent.count('#') == 0:
            return True
        else:
            return False
    
    def explored_current_room(self):
        """Return true if the player has already explored the current room."""
        # assume we are already in a room
        found, x, y = self.rooms[self.get_room()].find_char(' ')
        return (not found, x, y)
    
    def is_player_invisible(self):
        """Return true if the player is currently invisible."""
        x, y = self.cur_pos
        return not (self.map[x][y] == '@')
    
    def pathfind_to(self, target, initial=None, full_path=True, explored_set=None, override_target_traversability=False, override_targets=[]):
        """A* pathfinding from initial to target, where A* can visit any position that has been explored.
        
        Args:
            target: target position to pathfind to.
            initial: position to start pathfinding from. If None, use current player position.
            full_path: return entire trajectory if True, else return first position from initial.
            explored_set: if not None, increase A* heuristic score of non-explored tiles over explored tiles (e.g., if walking through a diagonal corridor, prefer to visit each square instead of moving diagonally, so we don't miss any branching corridor).
            override_target_traversability: pathfind to target even if it is not traversable by the player (e.g., solid wall).
            override_targets: override traversability of all targets in this list (see above parameter)
        """
        if initial == None:
            initial = self.cur_pos

        if (initial, target) not in self.pathfind_distances:
            overwritten_chars = {}
            for x, y in override_targets:
                overwritten_chars[(x, y)] = self.grid[x][y]
            if override_target_traversability:
                overwritten_chars[target] = self.grid[target[0]][target[1]]
                self.grid[target[0]][target[1]] = 0
            for x, y in override_targets:
                self.grid[x][y] = 0
            path = astar.astar(self.grid, initial, target, explored_set=explored_set)
            for (x, y) in overwritten_chars:
                self.grid[x][y] = overwritten_chars[(x, y)]
            
            if type(path) is bool:
                verboseprint("Error: could not pathfind from", initial, "to", target, "! (target on map:", self.map[target[0]][target[1]], " and basemap:", self.base_map[target[0]][target[1]], ")")
                raise Exception
            path.reverse() # path[0] should be next to start node.
            
            self.pathfind_distances[(initial, target)] = path
        
        path = self.pathfind_distances[(initial, target)]
        return path if full_path else path[0]
    
    def update_pathfinding_grid(self):
        """Update the pathfinding grid, setting a 0 if the position is traversable and 1 otherwise."""
        for i in range(ROWNO):
            for j in range(COLNO):
                self.grid[i][j] = 0 if self.base_map[i][j] in PASSABLE_CHARS else 1
    
    def mark_explored(self, pos):
        """Add the given position to the explored positions list.
        
        Args:
            pos: position that we want to mark as explored"""
        self.explored.add((pos))
    
    def mark_all_explored(self):
        """Mark all traversable positions in the map observed so far as explored, then update the pathfinding grid."""
        for i in range(ROWNO):
            for j in range(COLNO):
                if self.base_map[i][j] in PASSABLE_CHARS:
                    self.explored.add((i, j))
        self.update_pathfinding_grid()

class NetHackEnv(gym.Env, utils.EzPickle):
    """Basic NetHack environment. Must be subclassed. Contains statistics saving/loading methods and NetHack process management."""
    def __init__(self, nhinfo):
        """Initialize basic NetHack environment.
        Note: Actual step code is only found in subclasses.
        
        Args:
            nhinfo: NetHackInfo object to be used (in cases of multiple environments like Level). If None (default), it is created in set_config().
        """
        
        super().__init__()

        self.socket = None
        self.context = zmq.Context()

        self.records = {}
        #self.fname_infos = []
        self.total_num_games = 0
                
        self.single = nhinfo is None # if only this environment will be running, i.e., not Level.
        self.nh = nhinfo
    
    def load_records(self):
        """Load the saved records found at self.savedir/\*_records.dll"""
        for record_type in self.records:
            filename = self.savedir + record_type + "_records.dll"
            if os.path.exists(filename):
                try:
                    with open(filename, 'rb') as finput:
                        self.records[record_type] = dill.load(finput)
                        verboseprint("Existing recs found:", len(self.records[record_type]))
                except:
                    pass
    
    def save_records(self):
        """Save records to self.savedir/\*_records.dll, creating directories if necessary."""
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        for record_type in self.records:
            filename = self.savedir + record_type + "_records.dll"
            with open(filename, 'wb') as output:
                dill.dump(self.records[record_type], output)
    
    def close(self):
        """Save records."""
        self.save_records()
        #if self.daemon_socket is not None:
        #    self.daemon_socket.send("exit".encode())
        super().close()
    
    def get_savedir_info_list(self):
        """Get the strings that should form the save directory name."""
        return [
            self.name,
            str(self.proc_id),
            #self.policy.name
        ]
    
    def set_config(self, proc_id, num_procs, name, parse_items, **args):
        """Set config and connect to the NetHack launcher daemon.
        
        Args:
            proc_id: process ID of this environment, to be matched with the argument passed to the daemon launching script.
            num_procs: number of processes to run in parallel - used if grid search is running
            name: to be used for the record folder name
            parse_items: whether to handle items in the environment or not
        """
        
        self.name = name
        self.proc_id = proc_id
        self.num_procs = num_procs
        
        self.savedir = '_'.join(self.get_savedir_info_list()) + '/'
        self.basedir = deepcopy(self.savedir)
        if not os.path.exists(self.savedir):
            os.makedirs(self.savedir)
        self.load_records()
        
        self.parse_items = parse_items
        if self.nh is None:
            self.nh = NetHackInfo(parse_items)
        
        #spawn_daemon(self.proc_id)
        #time.sleep(2)
        if self.single:
            verboseprint("Connecting to daemon...")
            self.daemon_socket = self.context.socket(zmq.REQ)
            self.daemon_socket.connect("tcp://localhost:" + str(5555-self.proc_id-1))
            self.daemon_socket.send("test".encode())
            self.daemon_socket.recv()
            verboseprint("Connected")
    
    def reset(self):
        """Prepare the environment for a new map.
        Kills the current NetHack process and launches a new one."""

        while True:
            global log_str
            log_str = ""
            
            self.nh.reset()
            
            if self.socket is not None:
                kill_nh(self.socket)
                self.socket.close()
                self.socket = None
            
            if self.num_procs == 1:
                os.system("killall nethack > /dev/null 2>&1")
                os.system("rm nethack-3.6.0/game/*lock* > /dev/null 2>&1")
            
            launch_nh(self.daemon_socket)
            self.socket = self.context.socket(zmq.REP)
            self.socket.RCVTIMEO = 2000
            self.socket.bind("tcp://*:" + str(5555 + self.proc_id))
        
            # get observation
            message = rcv_msg(self.socket)
            self.process_msg(message)
            
            break
    
    def start_episode(self):
        return True
    
    def start_turn(self):
        pass
    
    def end_turn(self):
        pass
    
    def end_episode(self):
        """End the current episode, incrementing the game counter by one and calling to save_records() every 100 games."""
        if not self.single:
            return
        
        self.total_num_games += 1
        if self.total_num_games % 100 == 0:
            self.save_records()

class NetHackRLEnv(NetHackEnv):
    """Basic NetHack RL env with core step() and take_action() methods. Must be subclassed."""
    def __init__(self, nhinfo=None):
        """Initialize basic RL NetHack environment.
        
        Args:
            nhinfo: NetHackInfo object to be used (in cases of multiple environments like Level). If None (default), it is created in set_config().
        """
        super().__init__(nhinfo)
    
    def set_config(self, proc_id, action_size=1, state_size=1, max_num_actions=-1, max_num_episodes=-1, max_num_actions_per_episode=200, policy=None, **args):
        """Set config.
        
        Args:
            proc_id: process ID of this environment, to be matched with the argument passed to the daemon launching script.
            action_size: number of discrete actions that can be taken
            state_size: size of state vector
            max_num_actions: max number of actions that can be taken before exiting (***TODO***)
            max_num_episodes: max number of episodes to take before exiting (for level env.) (TODO -- but used for keras-rl)
            max_num_actions_per_episode: max number of (legal) actions that can be taken in an episode
        """
        
        self.action_space = spaces.Discrete(action_size)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(state_size,), dtype=np.float32)
        self.max_num_actions = max_num_actions
        self.max_num_episodes = max_num_episodes
        self.max_num_actions_per_episode = max_num_actions_per_episode
        #self.policy = policy
        
        super().set_config(proc_id, **args)
    
    def end_episode(self):
        """End the current episode."""
        if not self.single:
            return
        
        super().end_episode()
        
        if self.total_num_games == self.max_num_episodes:
            self.save_records()
        
    def get_game_params(self):
        """Parameters to pass to NetHack on the creation of a new game. (Will be saved in the options file.)"""
        return {
            'proc_id': self.proc_id
        }
    
    def reset(self):
        """Prepare the environment for a new episode."""
        self.total_actions_this_episode = 0
        self.last_action = None
        if self.single:
            save_nh_conf(**self.get_game_params())
            super().reset() # launch nh
            
            status = self.start_episode()
            assert status
            
            self.state = self.get_state()
            return self.state, self.get_valid_action_indices()
    
    def step(self, action):
        """Take the given action, receive the message output from NetHack and return the new state."""
        self.start_turn()
        status = Terminals.OK
        try:
            # Try to take the action.
            message = self.take_action(action)
            #assert not self.last_action_impossible
            if message is None:
                message = rcv_msg(self.socket)
        except zmq.error.Again:
            print("Error when sending action, process", self.proc_id)
            message = ""
            status = Terminals.CONN_ERROR
            self.goal_reached = Goals.CONN_ERROR
        
        if "paniclog" in message or "***dir***" in message:
            raise Exception("Unexpected message received from NetHack: " + message)
        
        if self.should_end_episode():
            verboseprint("Game went too long, terminating...")
            status = Terminals.TIME_EXCEEDED
            self.goal_reached = Goals.TIME_EXCEEDED
        
        if status is Terminals.OK:
            status, self.goal_reached = self.get_status(message)
                
        if status is Terminals.OK:
            self.process_msg(message)
            self.state = self.get_state()
        
        # Get reward for the given status.
        reward = self.get_reward(status)
        
        self.end_turn()
        
        # Check if episode is over.
        episode_over = status is not Terminals.OK
        if episode_over:
            assert self.goal_reached is not None       
            self.end_episode()
        #else:
        #    assert self.action_took_effect()
        
        valid_action_indices = self.get_valid_action_indices() if not episode_over else np.array([])
        return self.state, reward, episode_over, {}, valid_action_indices
    
    def process_msg(self, msg, update_base=True, parse_monsters=True, parse_ammo=False):
        """Processes the map screen outputted by NetHack."""
        if self.single:
            self.nh.process_msg(self.socket, msg, update_base=update_base, parse_monsters=parse_monsters, parse_ammo=parse_ammo)
    
    def get_status(self, msg):
        """Process the message returned by NetHack to check if it is a terminal state. Must be implemented in subclass."""
        raise NotImplementedError
    
    def take_action(self, action):
        """Send the action to NetHack."""
        
        self.last_action = action
        
        action = self.process_action(action)
        assert action is not None #self.last_action_impossible = True
        
        verboseprint("Sending", action)
        message = send_msg(self.socket, action)
        #self.last_action_impossible = False
        self.total_actions_this_episode += 1
        return message
    
    def process_action(self, action):
        """Do any preprocessing required on the action selected, e.g., get the CMD object from the abilities list."""
        cmd = action
        if isinstance(action, np.int64) or isinstance(action, int):
            cmd = self.get_command_for_action(action)
            verboseprint("Command:", cmd, "from action", self.abilities[action])
        
        return cmd
    
    def should_end_episode(self):
        """Check if we should end the current episode."""
        return self.total_actions_this_episode > self.max_num_actions_per_episode
    
    def get_valid_action_indices(self):
        """Get the indices of valid actions (according to the abilities list/action space). Should be implemented in subclass, if there are illegal actions in the action space. Currently returns all actions as valid."""
        return np.array([i for i in range(self.action_space.n)])
    
    def get_state(self):
        """Return state passed to RL agent. Should be implemented in subclass."""
        return np.array(self.observation_space.shape)
    
    def get_reward(self, status):
        """Return reward for the given status. Should be implemented in subclass."""
        return 0
    
    def set_test(self):
        """Change environment from training to test mode, if required."""
        pass
