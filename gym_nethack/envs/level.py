from collections import namedtuple

from gym_nethack.nhutil import *
from gym_nethack.nhdata import *
from gym_nethack.misc import verboseprint
from gym_nethack.envs.base import Terminals, Goals, NetHackRLEnv
from gym_nethack.envs import NetHackCombatEnv, NetHackExplEnv

Game = namedtuple('Game', 'goal_reached actions game_number final_clvl final_ac final_dlvl final_score final_inventory num_combat_acts num_expl_acts num_combat_encounters')

class NetHackLevelEnv(NetHackRLEnv):
    """NetHack level environment (exploration + combat)."""
    def __init__(self):
        """Initialize level environment."""
        super().__init__()
        self.records['level'] = []
        self.terminate = True
        self.new_level = False
    
    def get_savedir_info_list(self):
        """Get the strings that should form the save directory name."""
        
        return [
            *super().get_savedir_info_list(),
            'dataset=' + self.dataset,
            'secret' if self.secret_rooms else 'nonsecret'
        ]
    
    def set_config(self, proc_id, dataset='fixed', secret_rooms=False, num_episodes=100, **args):
        """Set config.
        
        Args:
            proc_id: process ID of this environment, to be matched with the argument passed to the daemon launching script.
            dataset: whether the maps are 'fixed' (same set of maps, i.e., same starting RNG seed) or 'random' (always different)
            secret_rooms: whether or not to enable secret door/corridor generation
            num_episodes: number of total episodes to run for.
        
        Other arguments are passed to the base, combat, and exploration env set_config() methods.
        """
        
        self.dataset = dataset
        self.secret_rooms = secret_rooms
        
        # Call the base environment set_config() method.
        super().set_config(proc_id, 1, 1, parse_items=True, max_num_episodes=num_episodes, max_num_actions_per_episode=3000, **args)
        
        # Initialize and set config for the exploration and combat envs.
        assert self.nh is not None
        self.expl = NetHackExplEnv(self.nh)
        self.expl.get_savedir_info_list = self.get_savedir_info_list # save info in same directory
        self.expl.set_config(proc_id, parse_items=True, secret_rooms=secret_rooms, **args)
        self.combat = NetHackCombatEnv(self.nh)
        self.combat.get_savedir_info_list = self.get_savedir_info_list # save info in same directory
        self.combat.set_config(proc_id, **args)
        
        # Rewire some methods and variables for integration purposes.
        self.expl.daemon_socket = self.daemon_socket
        self.combat.daemon_socket = self.daemon_socket
    
    def get_game_params(self):
        """Parameters to pass to NetHack on the creation of a new game. (Will be saved in the NH options file.)"""
        
        if self.dataset is 'fixed':
            seed = 1525485787+self.total_num_games
        elif self.dataset is 'random':
            seed = -1
        
        return {
            'proc_id': self.proc_id,
            'create_items': True,
            'create_mons':  True,
            'secret_rooms': self.secret_rooms,
            'seed': seed
        }
    
    def reset(self):
        """Prepare the environment for a new episode. (Call reset() on combat and exploration envs.)"""
        
        self.combat.reset()
        
        if not self.terminate:
            if self.new_level:
                self.new_level = False
                self.in_combat = True # assume monster is present at start(?)
                self.safe_monster_positions = []
                self.nh.reset()
                self.expl.reset()
                self.policy.reset()
                verboseprint("New level, resetting map data structures...")
                return self.combat.state, [self.abilities.index("wait")]
            else:
                # killed monster, but keep current NH process running.
                verboseprint("Continuing current ep on monster death...")
                return self.combat.state, self.combat.get_valid_action_indices()
        
        self.safe_monster_positions = []
        self.in_combat = True # assume monster is present at start(?)
                
        self.terminate = False
        self.new_level = False
                
        self.num_combat_actions = 0
        self.num_combat_encounters_this_game = 0
        self.num_expl_actions = 0
        
        self.expl.reset()
        s, v = super().reset()
        self.policy.reset()
        
        # We updated the socket variable during our base::reset(), so reflect changes in the envs.
        self.expl.socket = self.socket
        self.combat.socket = self.socket
        self.abilities = self.combat.abilities

        return s, v
    
    def start_episode(self):
        """Start a new episode (level), creating a record for it."""
        self.records['level'].append(Game(-1, [], -1, -1, 1000, 0, 0, [], 0, 0, 0))
        self.records['level'][-1] = self.records['level'][-1]._replace(actions=[])
        self.records['level'][-1] = self.records['level'][-1]._replace(game_number = self.total_num_games)
        
        return super().start_episode()
    
    def end_episode(self):
        """End the current episode, updating the record."""

        if self.goal_reached is Goals.SUCCESS:
            if self.policy.prev_in_combat:
                # don't end episode -- we just killed a monster
                self.num_combat_actions += self.combat.total_actions_this_episode
                self.combat.goal_reached = self.goal_reached
                self.combat.end_episode()
            else:
                # don't end episode -- we entered a new level
                self.expl.end_episode()
            return
        
        assert len(self.records['level']) > 0
        self.records['level'][-1] = self.records['level'][-1]._replace(goal_reached = self.goal_reached)
        self.records['level'][-1] = self.records['level'][-1]._replace(final_dlvl = int(self.nh.stats['dlvl']))
        self.records['level'][-1] = self.records['level'][-1]._replace(final_clvl = int(self.nh.stats['exp']))
        self.records['level'][-1] = self.records['level'][-1]._replace(final_ac = int(self.nh.stats['ac']))
        self.records['level'][-1] = self.records['level'][-1]._replace(final_score = int(self.nh.attributes['sc']))
        self.records['level'][-1] = self.records['level'][-1]._replace(final_inventory = self.nh.inventory)
        self.records['level'][-1] = self.records['level'][-1]._replace(num_combat_acts = self.num_combat_actions)
        self.records['level'][-1] = self.records['level'][-1]._replace(num_expl_acts = self.num_expl_actions)
        self.records['level'][-1] = self.records['level'][-1]._replace(num_combat_encounters = self.num_combat_encounters_this_game)
        
        super().end_episode()
        
        self.num_expl_actions += self.expl.total_actions_this_episode
        self.expl.end_episode()
        
        # Check if we have finished all the episodes.
        if len(self.records['level']) >= self.max_num_episodes:
            self.close()
            raise Exception("Finished.")
    
    def start_turn(self):
        """Start the current turn, calling the appropriate env. method."""
        super().start_turn()
        if self.in_combat:
            self.combat.start_turn()
        else:
            self.expl.start_turn()
    
    def end_turn(self):
        """End the current turn, calling the appropriate env. method."""
        super().end_turn()
        if self.in_combat and self.expl.total_actions_this_episode > 0:
            self.combat.end_turn()
            self.records['level'][-1].actions.append(self.combat.abilities[self.last_action])
        else:
            if self.last_action is not None:
                self.expl.total_actions_this_episode += 1
            self.expl.end_turn()
        self.policy.end_turn()
    
    def process_msg(self, msg):
        """Process the message returned by NetHack."""
        super().process_msg(msg, parse_ammo=False, update_base=not self.nh.in_fog)

        self.expl.process_msg(msg)
        self.combat.process_msg(msg)
    
    def get_command_for_action(self, action):
        """Translate the given action (of type integer -- an index into the self.abilities list) into a command that can be passed to NetHack, of type CMD."""

        if self.expl.total_actions_this_episode == 0:
            return CMD.WAIT
        
        if self.in_combat:
            return self.combat.get_command_for_action(action)
        else:
            return self.expl.get_command_for_action(action)
    
    def get_status(self, msg):
        """Check for a terminal state (death), or terminal state for one of the combat or exploration environments (monster died, or level finished)."""
        
        status, goal_reached = self.combat.get_status(msg) # check for player death
        if status is Terminals.PLAYER_DIED:
            self.terminate = True # start a new level when reset() is next called.
        
        if status is Terminals.OK:
            # we didn't die.
            
            if self.in_combat:
                self.process_msg(msg)
                self.policy.exploration_policy.observe_action()
                self.state = self.get_state()
                
                # check for monster death (not handled by combat env since NH won't send monster death signal when doing level env. since it's harder to detect due to multiple monsters possibly being present)
                if len(self.nh.monster_positions) < len(self.nh.prev_monster_positions) and not self.nh.in_fog:
                    status, goal_reached = Terminals.MONSTER_DIED, Goals.SUCCESS
        
            elif self.last_action == CMD.DIR.DOWN:
                # We reached a new level. Reset the exploration environment.
                verboseprint("Entering new level.")
                status, goal_reached = Terminals.SUCCESS, Goals.SUCCESS
                self.new_level = True
        
        return status, goal_reached
    
    def get_state(self):
        """Return state passed to RL agent."""
        if self.in_combat:
            self.combat.state = self.combat.get_state()
            return self.combat.state
        else:
            return self.expl.get_state()
    
    def get_valid_action_indices(self):
        """Get the indices of valid actions (according to the abilities list/action space)."""
        if self.in_combat:
            return self.combat.get_valid_action_indices()
        else:
            return self.expl.get_valid_action_indices()
    
    def get_reward(self, status):
        """Return reward for the given status."""
        assert status is not Terminals.MONSTER_DIED or self.in_combat
        return self.combat.get_reward(status)
    
    def monster_present(self):
        """Check if a monster is present within 6 squares of us."""
        
        self.combat.next_square_to_monster = None
        
        if 'laden with moisture' in self.nh.top_line or self.nh.in_fog:
            verboseprint("In fog, so a monster is surrounding us.")
            return True
        
        # if combat env. has not found any monsters during its parsing, return False
        if self.combat.cur_monster_pos is None:
            return False
        try:
            # get a trajectory to the monster.
            trajectory_to_monster = self.nh.pathfind_to(self.combat.cur_monster_pos, explored_set=self.nh.explored, override_target_traversability=True)
        except:
            # not possible to move there, so don't try.
            verboseprint("Monster present but can't find a path to it")
            return False
        self.combat.next_square_to_monster = trajectory_to_monster[0]
                
        # If combat env. has detected a monster during parsing, it is an unsafe monster (i.e., hostile) and within 6 spaces of us, then we enter into combat mode.
        return True if (len(self.combat.cur_monsters) > 0 and self.combat.cur_monster_pos is not None and self.combat.cur_monster_pos not in self.safe_monster_positions) and len(trajectory_to_monster) <= 6 else False
