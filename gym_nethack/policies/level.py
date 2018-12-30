from collections import deque

from gym_nethack.nhdata import CMD
from gym_nethack.misc import verboseprint
from gym_nethack.policies.core import Policy

class LevelPolicy(Policy):
    """Policy that can explore a level and enter combat with monsters, by using a different policy for combat and exploration."""
    name = 'level'
    
    def set_config(self, combat_policy, combat_policy_params, exploration_policy, exploration_policy_params, **args):
        """Set config.
        
        Args:
            combat_policy: policy to use during combat
            combat_policy_params: parameters to pass to combat policy
            exploration_policy: policy to use during exploration
            exploration_policy_params: parameters to pass to exploration policy"""
        self.combat_policy = combat_policy()
        self.combat_policy.set_config(**combat_policy_params)
        self.combat_policy.agent = self.agent
        self.combat_policy.env = self.env.combat
        self.combat_policy.env.policy = self.combat_policy
        self.exploration_policy = exploration_policy()
        self.exploration_policy.set_config(**exploration_policy_params)
        self.exploration_policy.agent = self.agent
        self.exploration_policy.env = self.env.expl
        self.exploration_policy.env.policy = self.exploration_policy
        
        self.NUM_COMBAT_ACTIONS_PEACEFUL = 200
    
    def reset(self):
        """Reset policy state to default."""
        self.prev_in_combat = False
        self.monster_is_present = False
        self.moving_to_exit = False
    
    def end_turn(self):
        """End the current turn."""
        self.prev_in_combat = self.env.in_combat
        self.monster_was_present = self.monster_is_present
        self.monster_is_present = self.env.monster_present()
        self.env.in_combat = self.monster_is_present and not self.exploration_policy.in_shop
    
    def set_trajectory_to_possible_exit(self):
        """Try to find the down stairs ('>') on the NetHack map. If it is not directly visible, look under monsters and dungeon features. Update exploration policy to trajectory towards the position."""
        assert not self.moving_to_exit
        
        self.env.nh.update_pathfinding_grid()
        
        exit_pos = self.env.nh.find_char_on_base_map('>')            
        if exit_pos is None:
            # can't reach exit from here, must explore more.
            verboseprint("Can't find path to exit")
        
            possibilities = []

            # add monster tiles to frontier
            for i, row in enumerate(self.env.nh.map):
                for j, col in enumerate(row):
                    if col in MONS_CHARS:
                        possibilities.append((i, j))
    
            # add masking dungeon features
            for i, row in enumerate(self.env.nh.map):
                for j, col in enumerate(row):
                    if col in ['}', '_']:
                        possibilities.append((i, j))
            
            self.exploration_policy.frontier_list.extend(possibilities)
        
            frontier_paths_to_player = [(self.env.nh.pathfind_to(frontier, override_target_traversability=True), frontier) for frontier in self.exploration_policy.frontier_list]
            frontier_dists_to_player = [(len(path), frontier) for path, frontier in frontier_paths_to_player if len(path) > 0]
            
            exit_pos = min(frontier_dists_to_player)[1]
        
        self.exploration_policy.target = exit_pos
        self.exploration_policy.current_trajectory = deque(self.env.nh.pathfind_to(self.exploration_policy.target, explored_set=self.env.nh.explored, override_target_traversability=True))
    
    def select_action(self, **kwargs):
        """Return an action to be taken, using either the exploration or combat policy depending on whether we are in combat or not."""
        if self.env.expl.total_actions_this_episode == 0:
            return CMD.WAIT
        
        if self.env.in_combat:
            if not self.prev_in_combat:
                verboseprint("\n***Switching to combat from exploration. Monster:", self.env.combat.cur_monsters)
                self.env.num_combat_encounters_this_game += 1
                self.combat_policy.env.start_episode()
                self.combat_policy.env.save_encounter_info()
            
            verboseprint("In combat this step")
            if self.combat_policy.env.total_actions_this_episode > self.NUM_COMBAT_ACTIONS_PEACEFUL and not self.env.nh.in_fog:
                # Spent too long on this monster, so decide it is peaceful.
                verboseprint("Combat actions exceeded max - marking this monster as safe")
                self.env.safe_monster_positions.append(self.combat_policy.env.cur_monster_pos)
            
            # query the combat policy
            return self.combat_policy.select_action(**kwargs)
        
        else:
            if self.prev_in_combat:
                # Update the current trajectory, since we have just come out of combat.
                verboseprint("\n***Switching to exploration from combat")
                if self.exploration_policy.target is not None and not self.env.nh.in_fog:
                    verboseprint("Updating stale trajectory")
                    self.exploration_policy.current_trajectory = deque(self.env.nh.pathfind_to(self.exploration_policy.target, explored_set=self.env.nh.explored, override_target_traversability=True))
                        
            verboseprint("Exploring this step")
            
            if self.env.nh.stats['blind'] == 'Blind' or self.env.nh.stats['conf'] == 'Conf' or self.env.nh.stats['stun'] == 'Stun' or self.env.nh.stats['hallu'] == 'Hallu':
                # Use WAIT action if we have a bad status effect.
                verboseprint("Waiting for bad status effect to go away.")
                return self.combat_policy.select_action(valid_action_indices=[self.env.abilities.index("wait")])
            
            if self.exploration_policy.done_exploring() and not self.env.nh.on_stairs():
                # Done exploring, so we have to go stairs to go to next level.
                
                verboseprint("Time to go to exit!")
                if not self.moving_to_exit:
                    self.set_trajectory_to_possible_exit()
                    self.moving_to_exit = True
                
            return self.exploration_policy.select_action(**kwargs)
