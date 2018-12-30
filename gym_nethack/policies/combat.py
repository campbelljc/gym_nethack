import random

from gym_nethack.policies.core import Policy
from gym_nethack.nhdata import *

class ApproachAttackPolicy(Policy):
    """Heuristic policy for NetHack combat that randomly equips a weapon (and armor, if specified), then approaches the monster and attacks it at close range. (If ranged weapon equipped, it will attack from a distance instead of approaching.)"""
    def __init__(self):
        """Initialize the policy."""
        self.tried_armor_indices = []
        self.weapon_choice = 50
    
    def set_config(self, equip_armor=False):
        """Set policy parameters.
        
        Args:
            equip_armor: whether to randomly choose a piece of armor and equip it, to a maximum of five pieces of armor, before starting to approach and attack the monster."""
        self.equip_armor = equip_armor
        self.name = 'appatkWA' if equip_armor else 'appatkW'

    def select_action(self, q_values, valid_action_indices):
        """Return the action corresponding to the heuristic policy.

        Args: 
            q_values: list of q-values, one per action
            valid_action_indices: indices of legal actions (corresponding to the abilities list)
        """
        if self.agent.episode_step == 0:
            self.tried_armor_indices = []
            valid_weapons = [i for i in valid_action_indices if i not in range(9) and self.agent.env.abilities[i].type in ['melee', 'ranged']] + [4]
            self.weapon_choice = random.choice(valid_weapons)
            return self.weapon_choice
        if self.equip_armor and len(self.tried_armor_indices) < 5:
            valid_armor = [i for i in valid_action_indices if i not in range(9) and self.agent.env.abilities[i].type in ARMOR_TYPES and i not in self.tried_armor_indices]
            chosen_armor = random.choice(valid_armor)
            self.tried_armor_indices.append(chosen_armor)
            return chosen_armor # try to equip all armors in succession before beginning approach/attack.
        
        if isinstance(self.agent.env.abilities[self.weapon_choice], tuple) and self.agent.env.abilities[self.weapon_choice].type == 'ranged':
            # ranged weapon equipped, so try to shoot projectile.
            projectile_actions = [i for i in valid_action_indices if i not in range(9) and self.agent.env.abilities[i].type == 'projectile']
            if len(projectile_actions) > 0:
                return random.choice(projectile_actions)
        
        # otherwise, do a melee attack if possible (i.e., if in range)
        if 1 in valid_action_indices:
            return 1 # attack if possible
        elif 2 in valid_action_indices:
            return 2 # otherwise line up (closer)
        elif 0 in valid_action_indices:
            return 0 # regular approach
        else:
            return 5 # wait if monster invis.

class ApproachAttackItemPolicy(Policy):
    """Heuristic policy for NetHack combat that randomly equips a weapon (and armor, if specified), then uses a random item with probability 0.25, and approaches the monster and attacks it at close range with probability 0.75. (If ranged weapon equipped, it will attack from a distance instead of approaching.)"""
    def __init__(self):
        """Initialize the policy."""
        self.tried_armor_indices = []
        self.weapon_choice = 50
    
    def set_config(self, equip_armor=False):
        """Set policy parameters.
        
        Args:
            equip_armor: whether to randomly choose a piece of armor and equip it, to a maximum of five pieces of armor, before starting to approach and attack the monster."""
        self.equip_armor = equip_armor
        self.name = 'appatkitemWASPWa' if equip_armor else 'appatkWPSW'

    def select_action(self, q_values, valid_action_indices):
        """Return the action corresponding to the heuristic policy.

        Args: 
            q_values: list of q-values, one per action
            valid_action_indices: indices of legal actions (corresponding to the abilities list)
        """
        assert len(valid_action_indices) > 0
        if self.agent.episode_step == 0:
            self.tried_armor_indices = []
            valid_weapons = [i for i in valid_action_indices if i not in range(9) and self.agent.env.abilities[i].type in ['melee', 'ranged']] + [4]
            self.weapon_choice = random.choice(valid_weapons)
            return self.weapon_choice
        if self.equip_armor:
            valid_armor = [i for i in valid_action_indices if i not in range(9) and self.agent.env.abilities[i].type in ARMOR_TYPES and i not in self.tried_armor_indices]
            if len(valid_armor) > 0:
                chosen_armor = random.choice(valid_armor)
                self.tried_armor_indices.append(chosen_armor)
                return chosen_armor # try to equip some armors in succession before beginning approach/attack.
                
        if random.random() < 0.75:
            # try to attack first.
            if isinstance(self.agent.env.abilities[self.weapon_choice], tuple) and self.agent.env.abilities[self.weapon_choice].type == 'ranged':
                # ranged weapon equipped, so try to shoot projectile.
                projectile_actions = [i for i in valid_action_indices if i not in range(9) and self.agent.env.abilities[i].type == 'projectile']
                if len(projectile_actions) > 0:
                    return random.choice(projectile_actions)
        
            # otherwise, do a melee attack if possible (i.e., if in range)
            if 1 in valid_action_indices:
                return 1 # attack if possible
            elif 2 in valid_action_indices:
                return 2 # otherwise line up (closer)
            elif 0 in valid_action_indices:
                return 0 # regular approach
            else:
                usable_item_indices = [i for i in valid_action_indices if i not in range(9) and self.agent.env.abilities[i].type in ['potion', 'scroll', 'wand', 'ring']]
                if max(valid_action_indices) > 9 and len(usable_item_indices) > 0:
                    return random.choice(usable_item_indices)
                else:
                    return 6 # no item available, so do random move.
        
        else:
            # try to use random item first.
            usable_item_indices = [i for i in valid_action_indices if i not in range(9) and self.agent.env.abilities[i].type in ['potion', 'scroll', 'wand', 'ring']]
            if max(valid_action_indices) > 9 and len(usable_item_indices) > 0:
                return random.choice(usable_item_indices)
            else:
                if isinstance(self.agent.env.abilities[self.weapon_choice], tuple) and self.agent.env.abilities[self.weapon_choice].type == 'ranged':
                    # ranged weapon equipped, so try to shoot projectile.
                    projectile_actions = [i for i in valid_action_indices if i not in range(9) and self.agent.env.abilities[i].type == 'projectile']
                    if len(projectile_actions) > 0:
                        return random.choice(projectile_actions)
        
                # otherwise, do a melee attack if possible (i.e., if in range)
                if 1 in valid_action_indices:
                    return 1 # attack if possible
                elif 2 in valid_action_indices:
                    return 2 # otherwise line up (closer)
                elif 0 in valid_action_indices:
                    return 0 # regular approach
                else:
                    return 6 # random move

class FireAntPolicy(Policy):
    """Heuristic policy for fire ant, as described in my thesis."""
    def __init__(self):
        """Initialize the policy."""
        super().__init__(name='fAntP')
        self.new_episode()
    
    def new_episode(self):
        """Start a new episode, resetting policy state."""
        self.used_wand_of_cancellation = False
        self.equipped_tsurugi = False
    
    def select_action(self, q_values, valid_action_indices):
        """Return the action corresponding to the heuristic policy.

        Args: 
            q_values: list of q-values, one per action
            valid_action_indices: indices of legal actions (corresponding to the abilities list)
        """
        if self.agent.episode_step == 0:
            self.new_episode()
        
        if not self.equipped_tsurugi:
            for i in valid_action_indices:
                if i > 8 and self.agent.env.abilities[i].full_name == 'uncursed +0 tsurugi':
                    self.equipped_tsurugi = True
                    return i
            assert False
        
        if not self.used_wand_of_cancellation:
            # check if we are lined up.
            for i in valid_action_indices:
                if i > 8 and self.agent.env.abilities[i].full_name == 'uncursed wand of cancellation':
                    self.used_wand_of_cancellation = True
                    return i
            # probably not lined up.
            return 3 # line up (farther) action
        
        if 1 not in valid_action_indices:
            return 0 # approach monster
        return 1 # attack when in range.
