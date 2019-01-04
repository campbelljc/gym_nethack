import random
from copy import deepcopy
from collections import namedtuple

import dill
import numpy as np

from gym_nethack.nhutil import *
from gym_nethack.nhdata import *
from gym_nethack.fileio import append
from gym_nethack.misc import verboseprint
from gym_nethack.envs.base import Terminals, Goals, NetHackRLEnv

# records will saved as a list of the following type
Combat = namedtuple('Combat', 'monster base_map map player_pos monster_positions start_state start_attributes start_stats start_items start_stateffs action_list goal_reached end_attributes end_stats end_items')

class NetHackCombatEnv(NetHackRLEnv):
    """Arena-style player-on-monster NetHack combat environment, with specifiable monsters, items, atts/stats."""
    def __init__(self, nhinfo=None):
        """Initialize basic one-on-one monster combat NetHack environment.
        
        Args:
            nhinfo: NetHackInfo object to be used (in cases of multiple environments like Level). If None (default), it is created in set_config().
        """
        super().__init__(nhinfo)
        self.records['combat'] = []
    
    def load_and_sample_combats(self):
        """Optionally load in a list of objects of type Combat from self.savedir+/combat_records*.dll.
        This list of combats will then be the ones trained against in this environment.
        This method is called if load_combats=True is passed to set_config."""
        
        combat_records = []
        i = 1
        while os.path.exists(self.savedir+"combat_records"+str(i)+".dll"):
            with open(self.savedir+"combat_records"+str(i)+".dll", 'rb') as finput:
                recs = dill.load(finput)
                combat_records.extend(recs)
            i += 1
    
        # filter out invalid monster combats
        combat_records = [rec for rec in combat_records if 'remembered_unseen_creature' not in rec.monster and len(rec.monster) > 0 and rec.monster[0] in MONSTER_NAMES]
                    
        # split up combats based on monster, and eliminate duplicates based on certain criteria
        unique_combats = set()
        combats_per_monster = [[] for i in range(len(MONSTER_NAMES))]
        for combat in combat_records:
            # criteria for duplicate checking: monster name, starting items, clvl, str, dex, AC, role (e.g., werewolf/normal), stat effs, dlvl
            tup = (combat.monster[0], tuple(combat.start_items), combat.start_stats['exp'], combat.start_attributes['st'], combat.start_attributes['dx'], combat.start_stats['ac'], combat.start_attributes['role_title'], tuple(combat.start_stateffs), combat.start_stats['dlvl'])
            if tup not in unique_combats:
                mon_ind = MONSTER_NAMES.index(combat.monster[0])
                combats_per_monster[mon_ind].append(combat)
                unique_combats.add((tup))
    
        # sample final records
        MAX_NUM_COMBATS_PER_MONSTER = 300
        NORMAL_COMBAT_RATIO = 0.85
        WERE_COMBAT_RATIO = 0.00
        STATEFF_COMBAT_RATIO = 0.15
        assert NORMAL_COMBAT_RATIO + WERE_COMBAT_RATIO + STATEFF_COMBAT_RATIO == 1
    
        for m_i, combats in enumerate(combats_per_monster):
            if len(combats) == 0: continue
            
            combats_per_stateff = [[] for i in range(len(STATUS_EFFECTS))]
            were_combats = []
            normal_combats = []
            
            # split combats into normal / lycanthropy / status effects
            # these 3 categories differ greatly in stategy, so we want to sample from each category appropriately.
            # sampling ratios are a few lines above this comment.
            for combat in combats:
                s_eff = False
                for s_i in range(len(STATUS_EFFECTS)):
                    if combat.start_stateffs[s_i] == 1:
                        combats_per_stateff[s_i].append(combat)
                        s_eff = True
                if not s_eff:
                    if 'Were' in combat.start_attributes['role_title']:
                        were_combats.append(combat)
                    else:
                        normal_combats.append(combat)
        
            random.seed(1337)
            sampled_combats = []
            
            stateff_combats = [x for sublist in combats_per_stateff for x in sublist]
            effective_stateff_ratio = STATEFF_COMBAT_RATIO
            if len(stateff_combats) > 0 and effective_stateff_ratio > 0:
                num_recs_to_sample = min(int(MAX_NUM_COMBATS_PER_MONSTER*effective_stateff_ratio), len(stateff_combats))
                if len(stateff_combats) <= num_recs_to_sample:
                    recs = stateff_combats
                else:
                    dist = stats.planck(0.01).pmf(range(len(stateff_combats)))
                    chosen_indices = np.random.choice(list(reversed(range(len(stateff_combats)))), size=num_recs_to_sample, replace=False, p=dist)
                    recs = [stateff_combats[i] for i in chosen_indices]
                sampled_combats.extend(recs)
                effective_stateff_ratio = len(recs) / len(combats)
            
            effective_were_ratio = WERE_COMBAT_RATIO
            if len(were_combats) > 0 and effective_were_ratio > 0:
                num_recs_to_sample = min(int(MAX_NUM_COMBATS_PER_MONSTER*effective_were_ratio), len(were_combats))
                if len(were_combats) <= num_recs_to_sample:
                    recs = were_combats
                else:
                    dist = stats.planck(0.01).pmf(range(len(were_combats)))
                    chosen_indices = np.random.choice(list(reversed(range(len(were_combats)))), size=num_recs_to_sample, replace=False, p=dist)
                    recs = [were_combats[i] for i in chosen_indices]
                sampled_combats.extend(recs)
                effective_were_ratio = len(recs) / len(combats)
            
            effective_norm_ratio = 1 - effective_stateff_ratio - effective_were_ratio
            if len(normal_combats) > 0 and effective_norm_ratio > 0:
                num_recs_to_sample = min(int(MAX_NUM_COMBATS_PER_MONSTER*effective_norm_ratio), len(normal_combats))
                if len(normal_combats) <= num_recs_to_sample:
                    recs = normal_combats
                else:
                    dist = stats.planck(0.1).pmf(range(len(normal_combats)))
                    chosen_indices = np.random.choice(list(reversed(range(len(normal_combats)))), size=num_recs_to_sample, replace=False, p=dist)
                    recs = [normal_combats[i] for i in chosen_indices]
                sampled_combats.extend(recs)
            
            combats_per_monster[m_i] = sampled_combats
        
        combats_per_monster = [(diff, combats, m_i) for diff, (m_i, combats) in zip(MONSTER_DIFFICULTIES, enumerate(combats_per_monster))]
        combats_per_monster.sort(key=lambda x: x[0])
        
        # print out num. combats per monster
        verboseprint("#comb diff mname")
        for diff, combats, m_i in combats_per_monster:
            verboseprint(len(combats), diff, MONSTER_NAMES[m_i])
        
        final_records = []
        for _, combats, _ in combats_per_monster:
            final_records.extend(combats)
        verboseprint("Total num:", len(combat_records), "and num selected:", len(final_records))
        
        return final_records
    
    def get_savedir_info_list(self):
        """Get the strings that should form the save directory name."""
        return [
            *super().get_savedir_info_list(),
            self.monsters_id,
            self.items_id,
            'df' + str(self.clvl_to_mlvl_diff)
        ]
    
    def set_config(self, proc_id, num_actions=-1, num_episodes=-1, clvl_to_mlvl_diff=-3, monsters='none', initial_equipment=[], items=None, item_sampling='uniform', num_start_items=5, action_list='all', fixed_ac=999, dlvl=None, tabular=False, test_policy=None, units_d1=0, units_d2=0, skip_training=False, load_combats=False, **args):
        """Set config.
        
        Args:
            proc_id: process ID of this environment, to be matched with the argument passed to the daemon launching script.
            num_actions: number of total actions to train for.
            num_episodes: number of total episodes to train for (only used if load_combats is False).
            clvl_to_mlvl_diff: the number of levels higher than the monster level that the player level will be set
            monsters: tuple of (idname, ['mon1', 'mon2', ...]) of monsters to be faced
            initial_equipment: list of items that the player will always start each episode with
            items: tuple of (idname, ['item1', 'item2', ...]) of items to be used in sampling
            item_sampling: how to determine which of the above 'items' will be given to the player at each episode start. could be 'all' (all items); 'uniform' (uniform sampling of size equal to the parameter below); or 'type' (see get_initial_inventory() for details)
            num_start_items: number of items to randomly sample from the 'items' parameter if item_sampling == 'uniform'
            action_list: determines what actions the player can use. can be: 'weapons_only' (only weapons allowed); otherwise any action is allowed
            fixed_ac: the player's starting armor class (AC); if < 999, will be set to this value; otherwise default NH initial value will be used
            dlvl: dungeon level for the episode. affects monster attributes (thus difficulty).
            tabular: whether we are using a tabular representation for the Q-values (deprecated)
            test_policy: used for record folder name (also used in ngym.py)
            units_d1: used for record folder name (also used in ngym.py)
            units_d2: used for record folder name (also used in ngym.py)
            skip_training: if True, will not add above info to folder name
            load_combats: whether to load combats from file to use for training; if True, many of the above parameters do not need to be specified.
        """
        
        self.tabular = tabular
        
        self.abilities = ['approach', 'attack monster', 'line up (closer)', 'line up (farther)', 'equip bare hands', 'wait', 'random move', 'move to ammo', 'pick up ammo']
        self.NUM_NON_ITEM_ABILITIES = len(self.abilities)
        
        self.action_list = action_list
        if action_list == 'weapons_only':
            self.item_set = [item for item in ALL_ITEMS if item.type in ['melee', 'ranged', 'projectile']]
            self.abilities.extend(self.item_set)
            self.include_armor = False
        else:
            self.item_set = ALL_ITEMS
            self.abilities.extend(self.get_all_item_abilities())
            self.include_armor = True
                
        self.item_sampling = item_sampling
        
        self.flat_abilities = []
        for i, ability in enumerate(self.abilities):
            if isinstance(ability, tuple):
                if ability.type == 'potion':
                    ability = ability.use_type + " " + ability.full_name
                else:
                    ability = ability.full_name
            self.flat_abilities.append(ability)
        self.flat_abilities.append(None)
        
        self.from_file = load_combats
        if self.from_file:
            self.savedir = 'combat_encounters/'
            self.combat_records = self.load_and_sample_combats()

            self.cur_enc_counter = 0
            if os.path.exists(self.savedir+"enccount.dll"):
                with open(self.savedir+"enccount.dll", 'rb') as finput:
                    self.cur_enc_counter = dill.load(finput)

            self.num_combat_encounters = len(self.combat_records)
            NUM_RUNS_PER_COMBAT_ENCOUNTER = 3 #2 #4 #3
            AVG_ACTIONS_PER_ENCOUNTERS = 25
            num_episodes = self.num_combat_encounters * NUM_RUNS_PER_COMBAT_ENCOUNTER
            num_actions = self.num_episodes * AVG_ACTIONS_PER_ENCOUNTERS
            self.memory_size = self.num_actions #(self.num_actions * 2) // 3 #
            self.num_actions_to_anneal_eps = (self.num_actions * 4) // 5
            verboseprint("Num episodes:", self.num_episodes, "and approx acts:", self.num_actions, "- press [enter]...")
            input("")
        
        if self.tabular:
            self.input_size = len(MONSTERS)+1+len(MONS_CHARS)+1+1+3+len(ROLES)+len(ALIGNMENTS)+1+1+(11*5)+1+len(STATUS_EFFECTS)+len(self.item_set)+len(WEAPONS)+len(MAX_DIST)+4+3 #+1
        else:
            self.input_size = len(MONSTERS)+1+len(MONS_CHARS)+1+1+3+len(ROLES)+len(ALIGNMENTS)+1+1+11+1+len(STATUS_EFFECTS)+len(self.item_set)+len(WEAPONS)+18+4+3 #+1
        
        if self.include_armor:
            self.input_size += len(ARMOR) + len(RINGS)
        
        self.starting_roles = ['Bar']
        
        if not self.from_file:
            self.clvl_to_mlvl_diff = clvl_to_mlvl_diff
            self.dlvl = 1 if dlvl is None else dlvl
            self.monsters = monsters[1]
            self.monsters_id = monsters[0]
            for mon in self.monsters:
                if mon not in MONSTER_NAMES:
                    raise Exception(str(mon) + " is not in monster list!")
                
            self.items = items[1]
            self.items_id = items[0]
            for item in self.items:
                if item not in self.item_set:
                    raise Exception("Can't use " + str(item))
            
            self.initial_equipment = initial_equipment
            self.fixed_ac = fixed_ac
    
            self.num_start_items = num_start_items
        
        #if not skip_training:
        #    self.fname_infos += [
        #        'lr' + str(lr),
        #        'u' + str(units_d1) + '-' + str(units_d2),
        #        str(num_actions)
        #    ]
        
        super().set_config(proc_id, action_size=len(self.abilities), state_size=self.input_size, parse_items=True, max_num_actions=num_actions, max_num_episodes=num_episodes, max_num_actions_per_episode=200, **args)
    
    def get_game_params(self):
        """Parameters to pass to NetHack on the creation of a new game. (Will be saved in the NetHack options file.)"""
        
        self.switch_encounter()
        role = self.get_initial_role()
        self.starting_items, self.starting_item_names = self.get_initial_inventory()
        self.cur_monster = random.choice(self.get_initial_monsters()).lower()
        self.initial_monster = self.cur_monster
        
        if not self.from_file:
            self.clvl = MONSTERS[MONSTER_NAMES.index(self.cur_monster)][1] + self.clvl_to_mlvl_diff
            num_armor_classes = 11 - -40 # 51
            num_levels = 31 - 1 # 30
            ac_lvl_ratio = num_armor_classes / num_levels # 1.7
            if self.fixed_ac < 999:
                self.ac = self.fixed_ac
            else:
                self.ac = 11 - ((self.clvl) * ac_lvl_ratio)
        else:
            if self.cur_monster not in NH_MONS:
                append("Couldn't combat with " + self.cur_monster, self.savedir+"/errors")
                self.reset()
                
        return {
            'proc_id': self.proc_id,
            'character': role,
            'clvl': self.clvl,
            'inven': self.starting_item_names,
            'mtype': NH_MONS.index(self.cur_monster),
            'ac': self.ac,
            'dlvl': self.dlvl,
            'adj_mlvl': True if self.from_file or self.dlvl > 1 else False,
            'st': self.st if self.from_file else 0,
            'dx': self.dx if self.from_file else 0,
            'lyc': self.lyc_type if self.from_file else None,
            'stateffs': self.stateff_flags if self.from_file else 1
        }
    
    def switch_encounter(self):
        """Called at the start of a new episode. If training on combats from file, switches to the next combat in the list."""
        if not self.from_file:
            return
        
        cur_enc = self.combat_records[self.cur_enc_counter%self.num_combat_encounters]
        self.cur_enc_counter += 1
        
        self.enc_start_items = cur_enc.start_items
        self.monsters = [cur_enc.monster]
        self.clvl = cur_enc.start_stats['exp']
        self.st = cur_enc.start_attributes['st']
        self.dx = cur_enc.start_attributes['dx']
        self.ac = cur_enc.start_stats['ac']
        self.dlvl = cur_enc.start_stats['dlvl']
        self.start_state = cur_enc.start_state
        self.start_stateffs = cur_enc.start_stateffs
    
        self.stateff_flags = 1
        for s_eff, prime in zip(self.start_stateffs, [1, 1, 1, 1, 2, 3, 5, 1, 1, 1, 1, 1, 7]):
            if s_eff == 1:
                self.stateff_flags *= prime
    
        self.nh.player_has_lycanthropy = 'Were' in cur_enc.start_attributes['role_title']
        self.lyc_type = None
        if self.nh.player_has_lycanthropy:
            if 'rat' in cur_enc.start_attributes['role_title']:
                self.lyc_type = 0
            elif 'jackal' in cur_enc.start_attributes['role_title']:
                self.lyc_type = 1
            elif 'wolf' in cur_enc.start_attributes['role_title']:
                self.lyc_type = 2
            else:
                raise Exception("Unknown lycanthropic type..!")
    
        if self.cur_enc_counter % 1000:
            with open(self.savedir+"enccount.dll", 'wb') as output:
                dill.dump(self.cur_enc_counter, output)            
    
    def reset(self):
        """Prepare the environment for a new episode."""
        self.prev_prev_monster_pos = None
        self.prev_monster_pos = None
        self.cur_monster_pos = None
        self.cur_monster_glyph = None
        self.lost_health_this_game = False
        #self.starting_moves = True
        self.cur_monster = None
        self.next_square_to_monster = None
        
        return super().reset()
    
    def process_msg(self, msg):
        """Processes the map screen outputted by NetHack."""
        super().process_msg(msg)
        
        self.cur_monsters = self.nh.top_line.split('&')[-2].replace("\x17", "").replace("\x15", "").replace(" ", "_").replace("holding_you", "").split(",_") #.replace("peaceful_", "")
        self.cur_monsters = [mon for mon in self.cur_monsters if len(mon) > 0 and 'peaceful' not in mon and 'tame' not in mon and 'priestess_' not in mon and 'priest_' not in mon and all(m not in SHOPKEEPER_NAMES for m in mon.split())]
        
        temp_cur_monster = self.cur_monsters[0] if len(self.cur_monsters) > 0 else ""
        unseen_monster = 'unseen_creature' in temp_cur_monster
        hidden_monster = 'invisible' in temp_cur_monster
        self.cur_mon_invisible = hidden_monster or unseen_monster or len(self.nh.monster_positions) == 0
        temp_cur_monster = temp_cur_monster.replace("invisible_", "").replace("remembered_unseen_creature", "")
        hallucinating = self.nh.stats['hallu'] == 'Hallu' or temp_cur_monster.lower() in IGNORE_MONS or temp_cur_monster.lower() in MONSTER_NAMES_HALLU
        if len(temp_cur_monster) > 2 and not hallucinating:
            self.cur_monster = temp_cur_monster # else, keep the old value in place.
        self.close_monster_positions = self.nh.monster_positions
        if len(self.nh.monster_positions) > 0:
            # find closest monster position.
            
            dists = []
            for mon_pos in self.nh.monster_positions:
                dists.append((abs(self.nh.cur_pos[0]-mon_pos[0])+abs(self.nh.cur_pos[1]-mon_pos[1]), mon_pos))
                verboseprint("Mon pos:", mon_pos, "and dist to player: ", dists[-1][0])
            self.close_monster_positions = [m[1] for m in dists if m[0] <= 6]
            self.cur_monster_pos = min(dists)[1] if len(dists) > 0 else None
            
            if not hallucinating and not unseen_monster:
                # we can see a monster and are not hallucinating, so save the monster's glyph.
                self.cur_monster_glyph = self.nh.map[self.cur_monster_pos[0]][self.cur_monster_pos[1]]
        else: # even if hallucinating, we know the monster's position.
            self.cur_monster_pos = None
            # keep the previous glyph value in place
        verboseprint(self.cur_monster_pos, self.cur_monster, self.cur_mon_invisible, self.cur_monster_glyph)
        
        self.prev_prev_monster_pos = self.prev_monster_pos
        self.prev_monster_pos = self.cur_monster_pos
        
        if self.single and self.total_actions_this_episode == 0:
            self.nh.mark_all_explored()
        
        # determine some things about the message so we don't keep recomputing them...
        self.monster_in_line_of_fire = self.is_monster_in_line_of_fire() if len(self.cur_monsters) > 0 else False
        
        if len(self.nh.rooms) == 0:
            self.nh.rooms.append(Room(self.nh))
        
        if 'hp' in self.nh.prev_stats and 'hp' in self.nh.stats and int(self.nh.stats['hp']) < int(self.nh.prev_stats['hp']):
            self.lost_health_this_game = True
    
    def get_status(self, msg):
        """Check if we died or the monster died."""
        status = Terminals.OK
        goal_reached = None
        if "***died***" in msg: # if in arena mode and we died, NH will send us this signal
            verboseprint("***Died!***")
            status = Terminals.PLAYER_DIED
            goal_reached = Goals.LOSS
        elif "***mondead***" in msg: # if in arena mode and monster has died, NH will send us this signal
            verboseprint("***Monster has died!***")
            status = Terminals.MONSTER_DIED
            goal_reached = Goals.SUCCESS
        return status, goal_reached
    
    def start_episode(self):
        """Start a new episode by preparing the episode record and checking if setup completed successfully."""
        assert self.nh is not None
        self.rooms = [Room(self.nh)]
        
        self.records['combat'].append(Combat(self.cur_monsters, self.nh.base_map, self.nh.map, self.nh.cur_pos, deepcopy(self.nh.monster_positions), self.get_state(), deepcopy(self.nh.attributes), deepcopy(self.nh.stats), deepcopy(self.nh.inventory), self.get_status_effects(), [], 0, {}, {}, []))
        
        if self.single and not assert_setup(self.starting_items, self.nh.inventory, self.initial_monster, self.nh.top_line, self.nh.monster_positions, self.nh.stats, self.ac):
            verboseprint("Invalid episode")
            return False
        
        return super().start_episode()
    
    def end_episode(self):
        """End episode by recording episode data."""
        assert len(self.records['combat']) > 0
        self.records['combat'][-1] = self.records['combat'][-1]._replace(end_items = deepcopy(self.nh.inventory))
        self.records['combat'][-1] = self.records['combat'][-1]._replace(end_attributes = deepcopy(self.nh.attributes))
        self.records['combat'][-1] = self.records['combat'][-1]._replace(end_stats = deepcopy(self.nh.stats))
        self.records['combat'][-1] = self.records['combat'][-1]._replace(goal_reached = self.goal_reached)
        
        assert self.cur_monster is not None
        #assert self.goal_reached is not None
        assert self.nh.inventory is not None
        append("Goal: " + str(self.goal_reached) + "; actions: " + str(self.records['combat'][-1].action_list) + "; monster: " + self.cur_monster + "; starting inventory: " + str(self.nh.inventory), self.savedir+"trajectories")
        
        super().end_episode()
    
    def get_reward(self, status):
        """Return the reward for the given status."""
                
        if status == Terminals.MONSTER_DIED:
            verboseprint("Reward: 20 (monster not found)")
            return 20
        elif status == Terminals.PLAYER_DIED:
            verboseprint("Reward: -2 (died)")
            return -2
        #elif self.prev_stats['hp'] - self.stats['hp'] > 2:
        #    return -0.5
        #elif len(self.nh.inventory) < self.len_prev_inventory:
        #    return -0.25
        elif status == Terminals.IMPOSSIBLE_ACTION:
            assert False
        else:
            verboseprint("Reward: -0.02 (time)")
            return -1/50
    
    def get_state(self):
        """Create and return the current state vector."""
        state = [
            *self.get_monster_vector(),                     # len(MONSTERS)+1+len(MONS_CHARS)+1
            *self.get_num_monsters(),                       # 3
            *self.get_character_vector(),                   # len(ROLES)
            *self.get_alignment_vector(),                   # len(ALIGNMENTS)
            self.lost_health_this_game,                     # 1
            self.nh.player_has_lycanthropy,                 # 1
            *self.get_norm_stats(discrete=self.tabular),    # 10 // 10*5
            self.nh.is_player_invisible(),                  # 1
            *self.get_status_effects(),                     # len(STATUS_EFFECTS)
            *self.get_inventory_vector(),                   # len(self.item_set)
            *self.get_current_equipment(),                  # len(WEAPONS) (+len(ARMOR)+len(RINGS))
            *self.get_distance_info(True),                  # 5 // len(MAX_DIST)+4
            *self.get_ranged_info(),                        # 3
            #self.starting_moves                             #
        ]
        
        assert len(state) == self.input_size
        return np.array(state)
    
    def get_valid_action_indices(self):
        """Return the list of valid action indices (according to the self.abilities list)."""
        assert len(self.nh.rooms) > 0
                
        valid_actions = []
        self.closest_ammo_pos = None
        
        # common info that might be needed
        cur_weap, cur_weap_cursed = self.nh.get_cur_weapon()
        #assert isinstance(cur_weap, Weapon) or cur_weap is None
        
        mpos = self.cur_monster_pos
        if mpos is not None:
            verboseprint("Monster has position; checking movement actions")
            self.next_square_to_monster = self.nh.pathfind_to(mpos, explored_set=self.nh.explored, override_target_traversability=True, full_path=False)
            
            dx, dy = self.nh.cur_pos[0] - mpos[0], self.nh.cur_pos[1] - mpos[1]
            dx_a, dy_a = abs(dx), abs(dy)
            dx_i, dy_i = -1*(bool(dx > 0) - bool(dx < 0)), -1*(bool(dy > 0) - bool(dy < 0))
            
            lined_positions = self.nh.rooms[0].get_lined_positions(mpos)
            
            if (dx_a > 1 or dy_a > 1) and self.next_square_to_monster is not None:
                valid_actions.append(self.abilities.index("approach"))
            if dx_a <= 1 and dy_a <= 1:
                valid_actions.append(self.abilities.index("attack monster"))
            if not self.monster_in_line_of_fire and (dx_a > 1 or dy_a > 1) and len(lined_positions) > 0:
                valid_actions.append(self.abilities.index("line up (closer)"))
            if not self.monster_in_line_of_fire and len(lined_positions) > 0:
                valid_actions.append(self.abilities.index("line up (farther)"))
                
        if cur_weap is not None and not cur_weap_cursed:
            valid_actions.append(self.abilities.index("equip bare hands"))
        
        if len(self.nh.ammo_positions) > 0 and not all(ammo_pos == self.nh.cur_pos for ammo_pos in self.nh.ammo_positions):
            verboseprint("Number of ammo positions:", len(self.nh.ammo_positions))
            close_ammo = []
            for ammo_pos in self.nh.ammo_positions:
                if self.nh.cur_pos == ammo_pos:
                    continue
                trajectory = deque(self.nh.pathfind_to(ammo_pos, explored_set=self.nh.explored, override_target_traversability=True))
                if len(trajectory) > 6: continue
                close_ammo.append((len(trajectory), trajectory))
            if len(close_ammo) > 0:
                self.closest_ammo_pos = min(close_ammo)
                valid_actions.append(self.abilities.index("move to ammo"))
        if self.nh.char_under_player() == ')':
            valid_actions.append(self.abilities.index("pick up ammo"))
        valid_actions.append(self.abilities.index("wait"))
        valid_actions.append(self.abilities.index("random move"))
        
        # item related ones.
        checked_indices = set() # guard against duplicate items (e.g., one weap equipped and same weap but different copy not equipped)
        for inven_item, _, stripped_name, matched_item, qty in self.nh.inventory:
            verboseprint("Checking inventory for actions:", inven_item)
            if stripped_name in IGNORED_ITEMS or matched_item is None:
                continue
            
            if matched_item.type == 'potion': #isinstance(matched_item, Potion):
                throw_pot = Potion(matched_item.name, matched_item.type, matched_item.buc, 'throw', matched_item.full_name)
                use_pot = Potion(matched_item.name, matched_item.type, matched_item.buc, 'use', matched_item.full_name)
                indices = [self.abilities.index(throw_pot), self.abilities.index(use_pot)]
            else:
                indices = [self.abilities.index(matched_item)]
            
            for i in indices:
                if i in checked_indices:
                    continue
                ability = self.abilities[i]
                
                valid = False
                
                if ability.type in ['melee', 'ranged']:
                    if not wielding(inven_item) and not cur_weap_cursed: valid = True
                elif ability.type in ARMOR_TYPES:
                    if not wearing(inven_item) and ability.type not in self.nh.equipped_armor_types:
                        if ability.type != 'shirt' or ('cloak' not in self.nh.equipped_armor_types and 'body armor' not in self.nh.equipped_armor_types):
                            valid = True
                elif ability.type == 'projectile':
                    if cur_weap is not None and cur_weap.name in RANGED_WEAP_NAMES and self.monster_in_line_of_fire: valid = True
                elif ability.type == 'potion':
                    if ability.use_type == 'throw':
                        if mpos is not None and self.monster_in_line_of_fire: valid=True
                    elif ability.use_type == 'use': valid = True
                elif ability.type == 'scroll':
                    valid = True if self.nh.stats['conf'] != 'Conf' else False
                elif ability.type == 'wand':
                    if mpos is not None and self.monster_in_line_of_fire and qty > 0:
                        valid = True
                elif ability.type == 'ring':
                    if 'hand' not in inven_item and self.nh.num_equipped_rings < 2 and (not cur_weap_cursed or self.nh.num_equipped_rings < 1):
                        valid_actions.append(i)
                
                if valid:
                    valid_actions.append(i)
            
            checked_indices.update(indices)
            if len(indices) == 0:
                # we couldn't map the inventory item to an ability
                append("Cannot map " + inven_item + " to ability (stripped: " + stripped_name + ")", self.savedir+"errors")
                raise Exception("Cannot map " + inven_item + " to ability (stripped: " + stripped_name + ")!")
        
        assert len(valid_actions) > 0
        return np.array(valid_actions)
    
    def get_all_item_abilities(self):
        """Split the potion items into ones of type 'throw' and 'use' - other items are unchanged."""
        items = []
        for item in ALL_ITEMS:
            if item.type == 'potion':
                items.append(Potion(item.name, item.type, item.buc, "throw", item.full_name))
                items.append(Potion(item.name, item.type, item.buc, "use", item.full_name))
            else:
                items.append(item)
        return items
        
    def get_initial_role(self):
        """Get player's initial role for each episode."""
        return random.choice(self.starting_roles)
    
    def get_initial_inventory(self):
        """Get player's starting inventory for each episode."""
        if self.from_file:
            item_names = []
            for j, (inven_item, x, stripped_name, matched_item, qty) in enumerate(self.enc_start_items):
                matchname = matched_item.full_name
                new_matched_item = matched_item
                if 'ring mail' in matchname and '+' not in matchname and '-' not in matchname:
                    matchname = matchname.replace("cursed ", "cursed +0 ").replace("blessed ", "blessed +0 ")
                    new_matched_item = matched_item._replace(full_name = matchname)
                    self.enc_start_items[j] = (inven_item, x, stripped_name, new_matched_item, qty)
                if stripped_name in IGNORED_ITEMS:
                    continue
                if equipped(inven_item):
                    item_names.append("*" + str(qty) + " " + matchname)
                else:
                    item_names.append(str(qty) + " " + matchname)
                items.append(new_matched_item)
            return items, item_names
        elif self.item_sampling == 'all':
            items = self.items
        elif self.item_sampling == 'uniform':
            if len(self.items) == 0:
                return [], []
            items = random.sample(self.items, self.num_start_items)
        elif self.item_sampling == 'type':
            items = []
            
            # one melee weapon per selected materials (4 total)
            #melee_weapons = [weap for weap in self.items if weap.type == 'melee' and weap.buc == 'uncursed' and weap.enchantment == '+0' and weap.condition == '']
            for material in ['iron', 'silver', 'wood']:
                items.append(random.choice([weap for weap in self.items if weap.type == 'melee' and weap.material == material and weap.buc == 'uncursed' and weap.enchantment == '+0' and weap.condition == '']))
            #items.extend(random.sample(melee_weapons, 3))
            
            # add ranged weap
            items.append(random.choice([weap for weap in self.items if weap.type == 'ranged']))
            
            # add 10 of two random ammo types per ranged weap.
            num_ranged = len([weap for weap in items if weap.type == 'ranged' and weap.buc == 'uncursed' and weap.enchantment == '+0' and weap.condition == ''])
            for i in range(num_ranged):
                ammo_types = random.sample([proj for proj in PROJECTILES if proj.buc == 'uncursed' and proj.enchantment == '+0' and proj.condition == ''], 2)
                for ammo in ammo_types:
                    #for i in range(10):
                    items.append(ammo)
            
            # ensure at least one weap for small/large damage bias (skip 'equal' bias type)
            #if not any(weap.dsize == 'small' for weap in items):
            #    items.append(random.choice([weap for weap in self.items if weap.type in ['melee', 'ranged'] and weap.dsize == 'small']))
            #if not any(weap.dsize == 'large' for weap in items):
            #    items.append(random.choice([weap for weap in self.items if weap.type in ['melee', 'ranged'] and weap.dsize == 'large']))
            
            if self.action_list != 'weapons_only':
                # 2 random potions
                potions = [pot for pot in self.items if pot.type == 'potion']
                if len(potions) > 0:
                    items.extend(random.sample(potions, 3))
            
                # 2 random scrolls
                scrolls = [pot for pot in self.items if pot.type == 'scroll']
                if len(scrolls) > 0:
                    items.extend(random.sample(scrolls, 3))
            
                # 2 random wands
                wands = [item for item in self.items if item.type == 'wand']
                if len(wands) > 0:
                    items.extend(random.sample(wands, 3))
                
                rings = [item for item in self.items if item.type == 'ring']
                if len(rings) > 0:
                    items.extend(random.sample(rings, 5))
            
                # five diff. armors
                #selected_armor_type = random.choice(ARMOR_TYPES)
                armors = [item for item in self.items if item.type in ARMOR_TYPES and 'dragon scale mail' in item.full_name and 'gray' not in item.full_name]
                items.extend(armors)
                #if len(armors) > 0:
                #    items.extend(random.sample(armors, 4))
                
                # 5 weaps + 2*10 ammo + 3 items + 5 armors = 15 items in inven.
        else:
            raise Exception("Unknown item sampling type ('uniform'/'type' allowed).")
        
        item_names = []
        for item in items:
            if item.type == 'projectile': # add 10 of selected ammo
                item_names.append("10 " + item.full_name)
            elif item.type == 'ring':
                item_names.append("uncursed +0 " + item.full_name.replace("uncursed ", ""))
            else:
                item_names.append(item.full_name)
        
        for item in self.initial_equipment:
            items.append(item)
            item_names.append("* " + item.full_name)
        
        return items, item_names
    
    def get_initial_monsters(self):
        """Get the possible monsters for each episode."""
        return self.monsters if not self.from_file else self.monsters[0]
    
    def set_test(self):
        """Change environment from training to test mode."""
        #assert self.training
        self.training = False
        self.total_num_games = 0
        self.cur_enc_counter = 0
    
    def save_encounter_info(self):
        """Called during full level combat to save combat encounters for training later on."""
        verboseprint("**   Creating new combat record   **")
        self.records['combat'].append(Combat(self.cur_monsters, self.nh.base_map, self.nh.map, self.nh.cur_pos, deepcopy(self.nh.monster_positions), self.get_state(), deepcopy(self.nh.attributes), deepcopy(self.nh.stats), deepcopy(self.nh.inventory), self.get_status_effects(), [], "", {}, {}, []))
    
    def get_command_for_action(self, action):
        """Translate the given action (of type integer -- an index into the self.abilities list) into a command that can be passed to NetHack, of type CMD."""
        actstr = self.abilities[action]
        self.records['combat'][-1].action_list.append(actstr)
        
        if self.abilities[action] == 'wait':
            return CMD.WAIT
    
        elif self.abilities[action] == 'equip bare hands':
            # de-equip current weapon.
            return [CMD.WIELD, "-"]
        
        elif self.abilities[action] == 'move to ammo':
            assert self.closest_ammo_pos is not None
            next_square = self.closest_ammo_pos[1][0] # first step in trajectory
            dx, dy = self.nh.cur_pos[0] - next_square[0], self.nh.cur_pos[1] - next_square[1]
            assert not (dx == 0 and dy == 0)
            verboseprint("Moving towards ammo")
            dx_i, dy_i = -1*(bool(dx > 0) - bool(dx < 0)), -1*(bool(dy > 0) - bool(dy < 0))
            return get_cmd_from_delta(dx_i, dy_i)
    
        elif self.abilities[action] == 'pick up ammo':
            # move towards ammo ... then pick up.
            assert self.nh.char_under_player() == ')'
            verboseprint("Standing on top of ammo")
            return CMD.PICKUP
        
        elif self.abilities[action] == 'random move':
            return get_cmd_from_delta(*random.choice(DIRS_DIAG))

        elif action >= self.NUM_NON_ITEM_ABILITIES: #isinstance(self.abilities[action], tuple):
            item = self.abilities[action]
            inven_char = self.nh.get_inven_char_for_item(item)
            
            if inven_char is None:
                raise Exception("Tried to use " + item + " but there was no associated inven char!")
                                    
            if item.type in ['melee', 'ranged']:
                return [CMD.WIELD, inven_char]
            elif item.type in ARMOR_TYPES:
                return [CMD.WEAR, inven_char]
            elif item.type == 'potion' and item.use_type == 'use':
                return [CMD.QUAFF, inven_char]
            elif item.type == 'scroll':
                #if 'teleportation' in item.full_name or 'magic mapping' in item.full_name:
                #    self.reset_expl() #TODO: Fix for level context
                return [CMD.READ, inven_char]
            elif item.type == 'ring':
                return [CMD.PUTON, inven_char]
        
        # common info that might be needed
        mpos = self.cur_monster_pos
        #dx, dy = self.nh.cur_pos[0] - mpos[0], self.nh.cur_pos[1] - mpos[1]
        dx, dy = self.nh.cur_pos[0] - self.next_square_to_monster[0], self.nh.cur_pos[1] - self.next_square_to_monster[1]
        dx_a, dy_a = abs(dx), abs(dy)
        dx_i, dy_i = -1*(bool(dx > 0) - bool(dx < 0)), -1*(bool(dy > 0) - bool(dy < 0))

        assert self.nh.cur_pos != self.cur_monster_pos 
        assert dx_i != 0 or dy_i != 0
    
        if self.abilities[action] == 'approach':
            verboseprint("Approaching monster: delta: ", dx_i, dy_i)
            verboseprint("Monster at pos", mpos, "and player is at", self.nh.cur_pos)
            return get_cmd_from_delta(dx_i, dy_i)
    
        elif self.abilities[action] == 'attack monster':
            # get cur weap index in self.abilities...
            cur_weap, _ = self.nh.get_cur_weapon()
            verboseprint("Attacking monster: delta: ", dx_i, dy_i, ", weapon:", cur_weap)
            return get_cmd_from_delta(dx_i, dy_i)
        
        elif 'line up' in self.abilities[action]:
            # move in direction parallel to monster(?)
            lined_positions = self.nh.rooms[0].get_lined_positions(mpos)

            # get positions in monster L.O.F. (line of fire)...
            dist_pos = []
            for lx, ly in list(lined_positions):
                dist_to_player = max(abs(self.nh.cur_pos[0] - lx), abs(self.nh.cur_pos[1] - ly))
                dist_pos.append((dist_to_player, (lx, ly)))
            
            min_dist = min(dist_pos)[0]
            min_positions = [(d, (lx, ly)) for (d, (lx, ly)) in dist_pos if d == min_dist]
            
            for i, (_, (lx, ly)) in enumerate(min_positions):
                dist_to_mon = max(abs(lx - mpos[0]), abs(ly - mpos[1]))
                min_positions[i] = (dist_to_mon, (lx, ly))
            
            closest_pos = min(min_positions)[1] if 'closer' in self.abilities[action] else max(min_positions)[1]
            
            # get direction towards that point
            dlx, dly = self.nh.cur_pos[0] - closest_pos[0], self.nh.cur_pos[1] - closest_pos[1]
            dlx_i, dly_i = -1*(bool(dlx > 0) - bool(dlx < 0)), -1*(bool(dly > 0) - bool(dly < 0))
            return get_cmd_from_delta(dx_i, dy_i)
        
        elif action >= self.NUM_NON_ITEM_ABILITIES:
            item = self.abilities[action]
            inven_char = self.nh.get_inven_char_for_item(item)

            assert inven_char is not None
                    
            direction = get_cmd_from_delta(dx_i, dy_i)
            
            if item.type == 'potion' and item.use_type == 'throw':
                return [CMD.THROW, inven_char, direction]
            elif item.type == 'projectile':
                # throw this projectile using currently equipped ranged weapon.
                direction = get_cmd_from_delta(dx_i, dy_i)
                        
                #if cur_weap.name in [weap.name for weap in ALL_WEAPONS_RANGED]:
                return [CMD.THROW, inven_char, direction]
                #elif cur_weap.name in NH_THROWING_WEAPONS:
                #    return [CMD.THROW, inven_char, direction]
                #else:
                #    raise Exception("Ranged weapon is neither firing or throwing type.")
            
            elif item.type == 'wand':
                return [CMD.ZAP, inven_char, direction]
            else:
                raise Exception("Unhandled item type: " + item_type)
    
        else:
            raise Exception("Unknown action: " + action)
    
    def action_took_effect(self):
        """Check if the last action taken has actually taken effect.
        Helps to diagnose errors that can propagate in state."""
        action = self.last_action
        assert action is not None
        
        if self.abilities[action] in ['wait']: #, 'attack monster']:
            return self.nh.cur_pos == self.nh.prev_pos or 'expel' in self.nh.top_line # or self.stats['conf'] == 'Conf' or 'swap places' in self.top_line or 'picks up' in self.top_line or self.stats['blind'] == 'Blind'
        
        elif self.abilities[action] == 'equip bare hands':
            for inven_item, _, _, obj, _ in self.nh.inventory:
                if wielding(inven_item):
                    return False
        
        elif self.abilities[action] == ['move to ammo', 'approach', 'line up (closer)', 'line up (farther)']:
            return self.nh.cur_pos != self.nh.prev_pos  or self.nh.stats['conf'] == 'Conf'
        
        #elif self.abilities[action] == 'pick up ammo':
        #    for inven_item, _, stripped_inven_item, weap_obj, qty in self.nh.inventory:
        #        if weap_obj.type == 'projectile':
        #            return self.len_prev_inventory < len(self.nh.inventory)
        #    return False
        
        elif action >= self.NUM_NON_ITEM_ABILITIES:
            item = self.abilities[action]
            if item.type in ['melee', 'ranged']:
                for inven_item, _, stripped_inven_item, obj, _ in self.nh.inventory:
                    if wielding(inven_item) and item_match(item.full_name, stripped_inven_item):
                        return True
                return False
            elif item.type in ARMOR_TYPES:
                for inven_item, _, stripped_inven_item, obj, _ in self.nh.inventory:
                    if wearing(inven_item) and item_match(item.full_name, stripped_inven_item):
                        return True
                return False
            elif item.type in ['potion', 'scroll']:
                return self.nh.len_prev_inventory > len(self.nh.inventory)
            #elif item.type == 'projectile':
            #    if self.len_prev_inventory > len(self.nh.inventory):
            #        return True
            #    for inven_item, _, stripped_inven_item, obj, qty in self.nh.inventory:
            #        if item_match(item.full_name, stripped_inven_item):
            #            
            elif item.type == 'ring':
                for inven_item, _, stripped_inven_item, obj, _ in self.nh.inventory:
                    if 'hand)' in inven_item and item_match(item.full_name, stripped_inven_item):
                        return True
                return False
        
        return True
    
    def get_monster_pos(self):
        """Get the position of the current monsteer: assuming only one monster."""
        if len(self.nh.monster_positions) > 0:
            # assume only one monster...!
            return self.nh.monster_positions[0]
        return None
    
    def get_distance_info(self, discrete):
        """Get the distance info for the state."""
        mpos = self.cur_monster_pos if len(self.cur_monsters) > 0 else None
        MAP_WIDTH = 18
        MAP_HEIGHT = 6
        MAX_DIST = 18 #max(MAP_WIDTH, MAP_HEIGHT) # diagonal movement permitted so take max
        
        you_changed_pos = 0 if (self.nh.prev_pos is None or self.nh.cur_pos == self.nh.prev_pos) else 1
        #print("You moved last turn?", True if you_changed_pos == 1 else False)
        #print("Monster cur pos:", mpos, "and prev pos:", self.prev_monster_pos, "and prev prev pos:", self.prev_prev_monster_pos)
        
        verboseprint("Monster:", mpos, ", player:", self.nh.cur_pos)
        
        if discrete:
            dist_vector = [0] * MAX_DIST # one-hot encoding for generalizability (discussed with prof. verbrugge)
                
            if mpos is None:
                dist_vector.extend([False, False, you_changed_pos, False])
                return dist_vector # no monster pos so can't calculate anything
        
            dx, dy = self.nh.cur_pos[0] - mpos[0], self.nh.cur_pos[1] - mpos[1]        
            #print("Player pos:", self.cur_pos, " and dist to monster: ", (dx, dy), " and prev pos:", self.nh.prev_pos, " and prev prev pos:", self.prev_prev_pos)
        
            dist_to_monster = max(abs(dx), abs(dy)) # get distance to monster (diagonal movement permitted)
            assert dist_to_monster >= 0 and dist_to_monster <= MAX_DIST
            dist_vector[dist_to_monster-1] = 1 # dist_vector[0] = 1 --> monster adjacent to player
        
        else:
            if mpos is None:
                return [0, False, False, you_changed_pos, False]
            
            dist_vector = []
            
            dx, dy = self.nh.cur_pos[0] - mpos[0], self.nh.cur_pos[1] - mpos[1]        
            #print("Player pos:", self.cur_pos, " and dist to monster: ", (dx, dy), " and prev pos:", self.nh.prev_pos, " and prev prev pos:", self.prev_prev_pos)
        
            dist_to_monster = max(abs(dx), abs(dy)) # get distance to monster (diagonal movement permitted)
            assert (dist_to_monster >= 0 and dist_to_monster <= MAX_DIST) or self.nh.stats['hallu'] == 'Hallu'
            norm_dist = dist_to_monster / MAX_DIST
            
            dist_vector.append(norm_dist)
        
        monster_changed_pos = 0 if (self.prev_monster_pos is None or mpos == self.prev_monster_pos) else 1
        #print("Monster moved last turn?", True if monster_changed_pos == 1 else False)
        
        if self.prev_prev_monster_pos is not None and self.prev_monster_pos is not None:
            prev_prev_dist = max(abs(self.prev_prev_monster_pos[0] - self.nh.prev_prev_pos[0]), abs(self.prev_prev_monster_pos[1] - self.nh.prev_prev_pos[1]))
            prev_dist = max(abs(self.prev_monster_pos[0] - self.nh.prev_pos[0]), abs(self.prev_monster_pos[1] - self.nh.prev_pos[1]))
            cur_dist = max(abs(mpos[0] - self.nh.cur_pos[0]), abs(mpos[1] - self.nh.cur_pos[1]))
            
            monster_approached = 1 if (prev_prev_dist - cur_dist) > 1 else 0
            #print("Cur dist:", cur_dist, ", prev_dist:", prev_dist, ", prev prev dist:", prev_prev_dist)
            #monster_approached = 1 if cur_dist < prev_dist and self.prev_monster_pos != mpos else 0
        elif self.prev_monster_pos is not None:
            prev_dist = max(abs(self.prev_monster_pos[0] - self.nh.prev_pos[0]), abs(self.prev_monster_pos[1] - self.nh.prev_pos[1]))
            cur_dist = max(abs(mpos[0] - self.nh.cur_pos[0]), abs(mpos[1] - self.nh.cur_pos[1]))
            monster_approached = 1 if cur_dist < prev_dist and self.prev_monster_pos != mpos else 0
        else:
            monster_approached = 0

        verboseprint("Monster approached for two last turns?", True if monster_approached == 1 else False)
        
        you_approached = 0
        if self.nh.prev_pos is not None and self.prev_monster_pos is not None:
            dist_prevplayer_prevmonster = max(abs(self.prev_monster_pos[0] - self.nh.prev_pos[0]), abs(self.prev_monster_pos[1] - self.nh.prev_pos[1]))
            dist_curplayer_prevmonster = max(abs(self.prev_monster_pos[0] - self.nh.cur_pos[0]), abs(self.prev_monster_pos[1] - self.nh.cur_pos[1]))
            if dist_curplayer_prevmonster < dist_prevplayer_prevmonster:
                # got closer to the monster's last position
                you_approached = 1
        
        verboseprint("You approached monster last turn?", True if you_approached == 1 else False)
             
        dist_vector.extend([monster_changed_pos, monster_approached, you_changed_pos, you_approached])
        assert all(d >= 0 and d <= 1 for d in dist_vector)
        return dist_vector
    
    def get_monster_vector(self):
        """Get the monster vector for the state."""
        verboseprint("Monsters detected:" + str(self.cur_monsters))        
        mons_vec = [0] * (len(MONSTERS)+1)
        mons_char_vec = [0] * (len(MONS_CHARS)+1)
        
        if self.cur_monster is None:
            return mons_vec + mons_char_vec + [0]
        elif self.cur_monster_pos is None:
            verboseprint("No monsters detected (monster invisible/hallucinating)")
            mons_vec[MONSTER_NAMES.index(self.cur_monster.lower())] = 1
            mons_char_vec[MONS_CHARS.index(self.cur_monster_glyph)] = 1
            invis = True
        else:
            verboseprint("Monster detected:" + self.cur_monster.lower())
            mons_vec[MONSTER_NAMES.index(self.cur_monster.lower())] = 1            
            mons_char_vec[MONS_CHARS.index(self.cur_monster_glyph)] = 1
            invis = self.cur_mon_invisible
        
        #assert sum(mons_vec) == 1
        return mons_vec + mons_char_vec + [invis]
        
        '''
        if len(self.cur_monsters) == 0:
            verboseprint("No monsters")
        elif self.stats['hallu'] == 'Hallu':
            verboseprint("Hallucinating - not sure which monster present")
            mons_vec[-1] = 1
            mons_char_vec[-1] = 1
        else:
            print(len(self.cur_monsters), len(self.close_monster_positions))
            #assert len(self.cur_monsters) == len(self.close_monster_positions)
            for mname in self.cur_monsters:
                if mname in MONSTER_NAMES_HALLU or mname in IGNORE_MONS:
                    mons_vec[-1] = 1
                else:
                    if mname.lower() not in MONSTER_NAMES:
                        raise Exception(mname.lower())
                    mons_vec[MONSTER_NAMES.index(mname.lower())] = 1
            
            #for mpos in self.close_monster_positions:
            mons_char_vec[MONS_CHARS.index(self.map[self.cur_monster_pos[0]][self.cur_monster_pos[1]])] = 1
            
            verboseprint("There are " + str(sum(mons_vec)) + " monsters...")        
        
        return mons_vec + mons_char_vec
        '''
    
    def get_character_vector(self):
        """Get the player role info for the state."""
        char_vec = [0] * len(ROLES)
        char_vec[ROLES.index(get_role_for_title(self.nh.attributes['role_title']))] = 1
        assert sum(char_vec) == 1
        return char_vec
    
    def get_alignment_vector(self):
        """Get the player alignment info for the state."""
        align_vec = [0] * len(ALIGNMENTS)
        align_vec[ALIGNMENTS.index(self.nh.attributes['align'][:2])] = 1
        assert sum(align_vec) == 1
        return align_vec
        
    def get_norm_stats(self, discrete):
        """Get the player attributes/dungeon level vector for the state."""
        NormStat = namedtuple('normstat', 'name val min_val max_val')
        stats = [
            NormStat('dlvl', int(self.nh.stats['dlvl']), 1, 30),
            NormStat('hp', int(self.nh.stats['hp']), 0, int(self.nh.stats['hp_max'])),
            NormStat('pw', int(self.nh.stats['pw']), 0, int(self.nh.stats['pw_max'])),
            NormStat('clvl', int(self.nh.stats['exp']), 1, 30),
            NormStat('str', int(self.nh.attributes['st']), 3, 25),
            NormStat('dex', int(self.nh.attributes['dx']), 3, 25),
            NormStat('const', int(self.nh.attributes['co']), 3, 25),
            NormStat('int', int(self.nh.attributes['in']), 3, 25),
            NormStat('wis', int(self.nh.attributes['wi']), 3, 25),
            NormStat('cha', int(self.nh.attributes['ch']), 3, 25),
            NormStat('ac', int(self.nh.stats['ac']), -40, 15)
        ]
        
        norm_stats = []
        for stat in stats:
            #verboseprint(stat.name + ": " + str(stat.val))
            assert stat.val >= stat.min_val and stat.val <= stat.max_val
            norm = (stat.val - stat.min_val) / (stat.max_val+1 - stat.min_val)
            assert norm >= 0 and norm <= 1
            
            if discrete:
                norm_stat = [0] * 5 # num bins
                for i in range(1, 6):
                    if norm >= (i-1)*0.2 and norm <= i*0.2:
                        norm_stat[i-1] = 1
                        break
                assert sum(norm_stat) == 1
            
                norm_stats.extend(norm_stat)
            else:
                norm_stats.append(norm)
        return norm_stats
    
    def get_current_equipment(self):
        """Get the current equipment vector (weapons/armor/rings) for the state."""
        weapons = [0] * len(WEAPONS)
        
        for inven_item, _, _, obj, _ in self.nh.inventory:
            if wielding(inven_item):
                weapons[WEAPONS.index(obj)] = 1
                verboseprint("Current equipped weapon:", inven_item)
                break
        assert sum(weapons) <= 1
        
        if not self.include_armor:
            return weapons
        
        armors = [0] * len(ARMOR)
        rings = [0] * len(RINGS)
        for inven_item, _, _, obj, _ in self.nh.inventory:
            if 'being worn' in inven_item:
                armors[ARMOR.index(obj)] = 1
                verboseprint("Armor equipped:", inven_item)
            elif 'on left hand' in inven_item or 'on right hand' in inven_item:
                rings[RINGS.index(obj)] = 1
                verboseprint("Ring equipped:", inven_item)
        
        return weapons + armors + rings
        
    def get_status_effects(self):
        """Get the player status effects info for the state."""
        stat_effs = [0] * len(STATUS_EFFECTS)
        
        for i, (skey, sval) in enumerate(STATUS_EFFECTS):
            if self.nh.stats[skey] == sval:
                verboseprint("Status effect:", sval)
                stat_effs[i] = 1
        
        return stat_effs

    def get_num_monsters(self):
        """Get the number of monsters for the state."""
        num_monsters = len(self.close_monster_positions)
        
        if num_monsters > 2:
            num_monsters = 2
        
        num_mons_vec = [0] * 3
        num_mons_vec[num_monsters] = 1 # 0->0; 1->1; 2+->2
        
        return num_mons_vec
    
    def get_inventory_vector(self):
        """Get the player inventory vector for the state."""
        items = [0] * len(self.item_set)
        
        ignored = 0
        for inven_item, _, stripped_inven_item, weap_obj, qty in self.nh.inventory:
            #if weap_obj not in self.item_set:
            #    continue 
            if weap_obj is None:
                verboseprint("\n", stripped_inven_item, weap_obj)
            elif weap_obj is not None:
                if weap_obj.type == 'projectile':
                    items[self.item_set.index(weap_obj)] = min(1, qty/10)
                elif weap_obj.type == 'wand':
                    items[self.item_set.index(weap_obj)] = min(1, qty/5)
                else:
                    items[self.item_set.index(weap_obj)] = 1
                continue
            if stripped_inven_item in IGNORED_ITEMS:
                ignored += 1
                continue
        
        #if sum(items) + ignored != len(self.nh.inventory):
        #    #print("\n", sum(items), ignored, len(self.nh.inventory), "\n", self.nh.inventory)
        #    #input("")
        
        return items
    
    def is_monster_in_line_of_fire(self):
        """Check if the monster is present in the directions we can fire towards."""
        
        mpos = self.cur_monster_pos #get_monster_pos()
        if mpos is None:
            verboseprint("No monster (so no LOF)")
            return False
        dx, dy = self.nh.cur_pos[0] - mpos[0], self.nh.cur_pos[1] - mpos[1]
        dx_i, dy_i = -1*(bool(dx > 0) - bool(dx < 0)), -1*(bool(dy > 0) - bool(dy < 0))
        
        strength = self.nh.attributes['st']
        if isinstance(strength, str) and '/' in strength: strength = strength.split("/")[0]
        arrow_range = (int(strength)//2) + 1
        for mult in range(arrow_range):
            if self.nh.basemap_char(self.nh.cur_pos[0] + dx_i*mult, self.nh.cur_pos[1] + dy_i*mult) == ' ':
                return False
            if mpos[0] == self.nh.cur_pos[0] + dx_i*mult and mpos[1] == self.nh.cur_pos[1] + dy_i*mult:
                verboseprint("Monster in line of fire")
                return True
        verboseprint("Monster not in line of fire")
        return False
    
    def get_ranged_info(self):
        """Get the projectile/ranged weapon info for the state."""
        line_of_fire = 1 if self.monster_in_line_of_fire else 0
        ammo_on_ground = 1 if len(self.nh.ammo_positions) > 0 else 0
        standing_on_ammo = 1 if self.nh.char_under_player() == ')' else 0                
        return [line_of_fire, ammo_on_ground, standing_on_ammo]

