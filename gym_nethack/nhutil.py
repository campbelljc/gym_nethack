import re, sys
from copy import deepcopy

from gym_nethack.nhdata import *
from gym_nethack.fileio import DIR_CHAR
from gym_nethack.conn import send_msg, rcv_msg, nethack_dir
from gym_nethack.misc import to_matrix, VERBOSE, verboseprint

def unpack_msg(msg, base_map, ignore_monsters=False, parse_ammo=True, update_base=True, parse_monsters=True):
    attrstat = msg[(21*COLNO):]
    
    if 'Dlvl' not in attrstat:
        raise Exception("Unexpected NH message: " + msg)
    
    align_pos = attrstat.find("Dlvl")
    assert align_pos >= 0
    top_delim_pos = attrstat[align_pos:].find("--")
    assert top_delim_pos >= 0
    top_delim_pos += align_pos
    ppos_pos = attrstat[top_delim_pos+2:].find("**")
    assert ppos_pos >= 0
    ppos_pos += top_delim_pos+2
    
    attmsg = attrstat[0:align_pos].replace("18/**", "21")
    sttmsg = attrstat[align_pos:top_delim_pos]
    topmsg = attrstat[top_delim_pos+2:ppos_pos]
    posmsg = attrstat[ppos_pos+2:-8].replace('\x00', "")
    uy, ux = posmsg.split('-')
    cur_pos = (int(ux), int(uy))
    back_glyph = int(attrstat[-6:-2])
    
    if 'laden with moisture' in topmsg or 'engulfs' in topmsg: # map obscured
        update_base = False
    
    full_map = to_matrix(list(msg[:(21*COLNO)]), COLNO) # 20x120
    
    ammo_positions = []
    monster_positions = []
    item_positions = set()
    food_positions = set()
    misc_positions = []
    critical_positions = []
    concrete_positions = set()
    num_explored_squares = 0
    for i, row in enumerate(full_map):
        for j, col in enumerate(row):
            if col != ' ':
                num_explored_squares += 1
            
            if update_base and base_map is not None and col in TOPOGRAPHICAL_CHARS:
                base_map[i][j] = full_map[i][j]
            
            if col in '+.': #|-
                critical_positions.append((i, j))
            elif col in '|-':
                concrete_positions.add((i, j))
            
            #elif col in '`^<' and slim_charset:
            #    base_map[i][j] = '.'
            #    full_map[i][j] = '.'
            
            elif col == ')' and parse_ammo:
                ammo_positions.append((i, j))
            
            elif col in MONS_CHARS and (i, j) != cur_pos:
                monster_positions.append((i, j))
            
            if col in '$[(%?/=!"*):':
                item_positions.add((i, j))
                if not parse_monsters: # or slim_charset:
                    if base_map is not None:
                        base_map[i][j] = '.'
                    full_map[i][j] = '.'
                else:
                    if update_base and base_map is not None and base_map[i][j] == ' ':
                        base_map[i][j] = '^'
                    full_map[i][j] = 'i'
                
                if col == '%':
                    food_positions.add((i, j))
            
            elif col in '_{\}': # room features
                misc_positions.append((i, j))
                if base_map is not None:
                    base_map[i][j] = '.'
                full_map[i][j] = '.'
            
            elif col == '`' or (col in MONS_CHARS and (i, j) != cur_pos):
                if col not in MONS_CHARS:
                    misc_positions.append((i, j))
                if not parse_monsters: # or slim_charset:
                    if base_map is not None:
                        base_map[i][j] = '.'
                    full_map[i][j] = '.'
                else:
                    if update_base and base_map is not None and base_map[i][j] == ' ':
                        base_map[i][j] = '^'
    
    if update_base and base_map is None:
        base_map = deepcopy(full_map)
        base_map[cur_pos[0]][cur_pos[1]] = '.' # player always starts in room.
        for ipos in list(item_positions) + misc_positions:
            base_map[ipos[0]][ipos[1]] = '.'
        for ipos in monster_positions:
            base_map[ipos[0]][ipos[1]] = '^'
        
    concrete_positions = set() #TODO
    return base_map, full_map, attmsg, sttmsg, topmsg, cur_pos, monster_positions, ammo_positions, item_positions, food_positions, back_glyph, critical_positions, concrete_positions, num_explored_squares

def get_inventory(socket):
    try:
        send_msg(socket, CMD.INVENTORY)
        raw = rcv_msg(socket)
    except zmq.error.Again:
        raise Exception("Error occurred communicating with NetHack to get inventory.")
    
    items = raw.split("--")
    inventory = []
    matched_names = []
    for item_str in items[1:-1]:
        if len(item_str.split(",")) != 2:
            print("Couldn't interpret:", item_str)
        item_name, item_char = item_str.split(",")
        item_name = item_name.replace("-2", "-1").replace("+2", "+1").replace("-3", "-1").replace("+3", "+1").replace("+4", "+1").replace("-4", "-1").replace("thoroughly ", "").replace("very ", "")
        
        qty, stripped_name = get_stripped_itemname(item_name)
        if stripped_name in IGNORED_ITEMS:
            continue
        
        matched_item = None
        for i, weap in enumerate(ALL_ITEMS):
            if item_match(weap.full_name, stripped_name):
                #if weap.full_name in matched_names:
                #    print(weap.full_name, "already present in inven! :", inventory)
                
                matched_names.append(weap.full_name)
                matched_item = weap
                break
    
        if matched_item is None:
            verboseprint("\nMatched item was none! Item_str:", item_str, "and stripped:", stripped_name)
            continue
        assert item_char is not None
        
        if ':' in item_name:
            ind = item_name.index(':')
            qty = item_name[ind+1:ind+2]
                
        inventory.append((item_name, item_char, stripped_name, matched_item, 1 if len(qty) == 0 else int(qty)))
    return inventory

def get_stripped_itemname(inventory_name):
    # get rid of opening digits (quantities)
    qty = ""    
    while inventory_name[0].isdigit():
        qty += inventory_name[0]
        inventory_name = inventory_name[1:]

    while inventory_name[0] == ' ':
        inventory_name = inventory_name[1:]
    
    strip_strs = ['a ', 'an ', 'the ']
    for sstr in strip_strs:
        if inventory_name.startswith(sstr):
            inventory_name = inventory_name.replace(sstr, "", 1)
        
    # get rid of brackets
    if '(' in inventory_name:
        inventory_name = inventory_name[:inventory_name.index('(')-1]
        
    # get rid of plurals
    for x, y in PLURALS:
        inventory_name = inventory_name.replace(x, y)
    
    # named items (artifacts)
    if ' named ' in inventory_name:
        assert False #TODO-incorporate buc status
        inventory_name = inventory_name.split(' named ')[1]
    
    if 'ring of' in inventory_name:
        inventory_name = inventory_name.replace("+0 ", "")
    
    # only needed when artifacts are included
    #SPECIAL_ARTIFACT_NAMES = ['Staff of Aesculapius', 'Tsurugi of Muramasa', 'Sceptre of Might']
    #if any(art_name in inventory_name for art_name in SPECIAL_ARTIFACT_NAMES):
    #    inventory_name = inventory_name.replace('The ', '')
    
    return qty, inventory_name

def item_match(item_name, inventory_name):
    if 'cursed' not in inventory_name and 'blessed' not in inventory_name and 'holy water' not in inventory_name:
        inventory_name = 'uncursed ' + inventory_name # default to uncursed    
    #if VERBOSE:
    #    print("matching '", item_name, "' to '", inventory_name, "'")
    return item_name == inventory_name

def update_attrs(attr_line, attributes):
    m = re.search(r'the (?P<role_title>'+NH_ROLE_TITLES_FLAT+')\s*'
                    r'St:(?P<st>[/\d\*]+)\s*'
                    r'Dx:(?P<dx>\d+)\s*'
                    r'Co:(?P<co>\d+)\s*'
                    r'In:(?P<in>\d+)\s*'
                    r'Wi:(?P<wi>\d+)\s*'
                    r'Ch:(?P<ch>\d+)\s*'
                    r'S:(?P<sc>\d+)\s*'
                    r'(I:(?P<inv>\d+)\s*)'
                    r'(?P<align>\S+)', attr_line)
    if m:
        prev_attributes = deepcopy(attributes)
        attributes = m.groupdict()
        for k, v in attributes.items():
            if v and v.isdigit():
                attributes[k] = int(v)
        
        if '/' in str(attributes['st']):
            strength = str(attributes['st']).split("/")
            strength = [int(s) for s in strength]
            assert strength[0] == 18
            if 0 <= strength[1] <= 31:
                strength = 19
            elif 32 <= strength[1] <= 81:
                strength = 20
            elif 82 <= strength[1]:
                strength = 21
            attributes['st'] = strength
        
        return prev_attributes, attributes
    else:
        raise Exception("No attributes! Attr line was:\n" + attr_line)

def update_stats(stat_line, stats):
    m = re.search(r'Dlvl:(?P<dlvl>\S+)\s*'
                   r'\\\w+:(?P<money>\d+)\s*'
                   r'HP:(?P<hp>\d+)\((?P<hp_max>\d+)\)\s*'
                   r'Pw:(?P<pw>\d+)\((?P<pw_max>\d+)\)\s*'
                   r'AC:(?P<ac>[+-]?\d+)\s*'
                   r'R:(?P<rooms>\d+)\s*'
                   r'SD:(?P<sdoor>\d+)\s*'
                   r'Exp:(?P<exp>\d+)\s*'
                   r'(?P<hunger>Satiated|Hungry|Weak|Fainting)?\s*'
                   r'(?P<stun>Stun)?\s*'
                   r'(?P<conf>Conf)?\s*'
                   r'(?P<blind>Blind)?\s*'
                   r'(?P<burden>Burdened|Stressed|Strained|Overtaxed|Overloaded)?\s*'
                   r'(?P<hallu>Hallu)?\s*', stat_line)
    if m:
        prev_stats = deepcopy(stats)
        stats = m.groupdict()
        for k, v in stats.items():
            if v and v.isdigit():
                stats[k] = int(v)
        return prev_stats, stats
    else:
        raise Exception("No statistics! Stat line was: ", stat_line)

def assert_setup(starting_items, inventory, cur_monster, top_line, monster_positions, stats, start_ac):
    verboseprint("Asserting setup...")
    
    # check that we have all 5 items in our inventory...
    for item in starting_items:
        found = False
        if item[0].isdigit():
            item = item[item.index(" ")+1:-1] # get rid of ammo quantity
        for _, _, stripped_inven_item, matched_item, _ in inventory:
            #print("Trying to match the starting item,", item, "to the thing in our inventory, stripped name:", stripped_inven_item)
            itemname = matched_item.full_name.replace("poisoned ", "")
            if itemname == item.full_name: # item_match(item.full_name, stripped_inven_item):
                found = True
                break
        if not found:
            print("\nCould not find", item, "in inventory:")
            for inven_name, _, _, it, _ in inventory:
                print(inven_name, it.full_name.replace("poisoned ", ""))
            #append("Could not find " + item.full_name + " in inventory: " + str(inventory) + " (starting items: " + str(starting_items) + ")", self.savedir+"errors")
            #raise Exception
    
    # check that there is at least one monster...
    if len(monster_positions) < 1 and stats['hallu'] != 'Hallu':
        #append("Could not find monster " + cur_monster, self.savedir+"errors")
        raise Exception("No monster (was looking for " + cur_monster + ")!")
    
    # check that the monster is the correct char
    mon_names = [cur_monster.replace("_", "-").lower(), cur_monster.replace("_", " ").lower()]
    if all(mname not in top_line.lower() for mname in mon_names) and stats['hallu'] != 'Hallu':
        print("Monster (" + cur_monster + ") not in top line (" + top_line.lower() + ")!")
        #append("Monster (" + cur_monster + ") not in top line (" + top_line.lower() + ")", self.savedir+"errors")
        #return False
    
    verboseprint("Setup looks good.")
    return True

def save_nh_conf(proc_id, secret_rooms=False, character="Bar", race="Human", clvl=1, st=0, dx=0, mtype=None, create_mons=False, ac=999, inven=[], dlvl=1, lyc=None, stateffs=1, adj_mlvl=True, create_items=True, seed=-1):
    if sys.platform == "win32":
        sysconf_fname = nethack_dir + DIR_CHAR + "defaults.nh"
    else:
        sysconf_fname = nethack_dir + DIR_CHAR + "sysconf" + str(proc_id)
    verboseprint("Writing to sysconf file:", sysconf_fname)
    with open(sysconf_fname, 'w') as sysconf:
        sysconf.write("OPTIONS=!autopickup, !bones, pushweapon, pettype:none, time, disclose:-i -a -v -g -c -o, ")
        if secret_rooms:
            sysconf.write("secret_rooms, ")
        if not adj_mlvl:
            sysconf.write("!")
        sysconf.write("adjust_mlvl, ")
        if not create_items:
            sysconf.write("!")
        sysconf.write("create_items, ")
        if mtype is not None:
            sysconf.write("combat_setup, create_mons, ")
        elif create_mons:
            sysconf.write("!combat_setup, create_mons, ")
        else:
            sysconf.write("!combat_setup, !create_mons, ")
        sysconf.write("character:"+character+", race:"+race+", gender:male, name:Merlin, align:chaotic, ")
        sysconf.write("reqlevel:"+str(clvl)+", reqstr:"+str(st)+", reqdex:"+str(dx)+", reqdlvl:"+str(dlvl))
        if ac < 999:
            sysconf.write(", reqac:"+str(ac))
        if lyc is not None:
            sysconf.write(", reqlyc:"+str(int(lyc)))
        if stateffs > 1:
            sysconf.write(", stateffs:"+str(stateffs))
        if mtype is not None:
            sysconf.write(", mtypeid:"+str(mtype))
        if seed > -1:
            sysconf.write(", seed:"+str(seed))
        sysconf.write("\nWIZKIT=wizkit" + str(proc_id) + ".txt\n")
    wizkit_fname = nethack_dir + DIR_CHAR + "wizkit" + str(proc_id) + ".txt"
    with open(wizkit_fname, 'w') as wizkit:
        for item in inven:
            #enc = 3 if 'wand' in item else 0 # charges for wand
            #buc = 'uncursed' if 'holy water' not in item else 'blessed'
            #
            #if item[0].isdigit(): # if quantity specified
            #    quantity = item.split(" ")[0]
            #    item = item[item.index(" "):]
            #else:
            #    quantity = 1
            #
            #wizkit.write(str(quantity) + " " + buc + " +" + str(enc) + " " + item + "\n")
            wizkit.write(item + "\n")

def get_cmd_from_delta(dx, dy):
    for delta, cmd in DIR_MAPPING:
        if dx == delta[0] and dy == delta[1]:
            return cmd
    raise Exception("Couldn't find a CMD mapping from " + str(dx) + "," + str(dy))

class Room(object):
    def __init__(self, nh):
        self.nh = nh
        self.wall_positions = set()
        self.wall_openings = set()
        self.corners = set()
        self.positions = set()
        self.top_left_corner = None
        self.__get_wall_infos()
        
        self.centroid = (sum([p[0] for p in self.positions]) // len(self.positions), sum([p[1] for p in self.positions]) // len(self.positions))
    
    def __get_dists_to_walls(self):
        dists = []        
        for dx, dy in DIRS:
            cur_x, cur_y = self.nh.cur_pos
            d = 0
            while self.nh.basemap_char(cur_x+dx, cur_y+dy) not in WALL_CHARS + DOOR_CHARS and self.nh.basemap_char(cur_x+dx, cur_y+dy) not in DOOR_CHARS and (cur_x+dx, cur_y+dy) not in self.nh.room_openings:
                # probably still in a room

                adjacent_chars = self.nh.get_chars_adjacent_to(cur_x+dx, cur_y+dy)
                if (adjacent_chars.count('.') + adjacent_chars.count('<') + adjacent_chars.count('>') + adjacent_chars.count('^')) < 2:
                    break

                cur_x += dx
                cur_y += dy
                d += 1
            dists.append(d)
        return dists
    
    def __get_wall_positions(self, fixed, c1, c2, x_axis=True):
        positions, openings = [], []
        for c in range(c1, c2):
            if x_axis:
                wall = self.nh.base_map[c][fixed] == '|'
                coord = (c, fixed)
            else:
                wall = self.nh.base_map[fixed][c] == '-'
                coord = (fixed, c)
            if wall:
                positions.append(coord)
            else:
                openings.append(coord)
        self.wall_positions.update(positions)
        self.wall_openings.update(openings)
        
    def __get_wall_infos(self):
        x, y = self.nh.cur_pos
        wall_dists = self.__get_dists_to_walls()

        topx = x - wall_dists[0] # (-1, 0)
        topy = y - wall_dists[2] # (0, -1)
        
        bottomx = x + wall_dists[1] # (1, 0)
        bottomy = y + wall_dists[3] # (0, 1)
        
        self.corners = [(topx-1, topy-1), (bottomx+1, bottomy+1), (topx-1, bottomy+1), (bottomx+1, topy-1)]
        
        self.__get_wall_positions(topy-1, topx, bottomx+1)
        self.__get_wall_positions(bottomy+1, topx, bottomx+1)
        self.__get_wall_positions(topx-1, topy, bottomy+1, x_axis=False)
        self.__get_wall_positions(bottomx+1, topy, bottomy+1, x_axis=False)
        
        for px in range(topx, bottomx+1):
            for py in range(topy, bottomy+1):
                self.positions.add((px, py))
        
        self.top_left_corner = (topx, topy)
    
    def count_char(self, char):
        count = 0
        for x, y in self.positions:
            if self.nh.map[x][y] == char:
                count += 1
        return count
    
    def find_char(self, char):
        for x, y in self.positions:
            if self.nh.map[x][y] == char:
                return (True, x, y)
        return (False, -1, -1)
    
    def get_lined_positions(self, mpos):
        lined_positions = set()
            
        # get possible positions
        for dir_x, dir_y in DIRS_DIAG:
            for mrange in range(13): # TODO
                lx = mpos[0] + dir_x*mrange
                ly = mpos[1] + dir_y*mrange

                if (lx, ly) not in self.positions:
                    break
            
                if (mpos[0] != lx and mpos[1] != ly):
                    lined_positions.add((lx, ly))
        
        return lined_positions

class Passage():
    """Helper class to maintain info about map corridors and what rooms they lead to."""
    next_id = 0
    def __init__(self, first_room, first_position):
        self.positions = set()
        self.connected_rooms = set()
        self.connected_room_openings = set()
        self.connect_room(first_room)
        self.add_position(first_position)
        self.id = Passage.next_id
        Passage.next_id += 1
        #verboseprint("Creating a new passage with first pos", self.positions, "and first room", self.connected_rooms)
    def __repr__(self):
        return str(self.id) + str(self.positions) + str(self.connected_rooms)
    def connect_room(self, room_pos):
        self.connected_rooms.add((room_pos))
    def add_position(self, passage_pos):
        self.positions.add((passage_pos))
    @staticmethod
    def merge(p1, p2):
        for room in p2.connected_rooms:
            p1.connect_room(room)
        for pos in p2.positions:
            p1.add_position(pos)
        for pos in p2.connected_room_openings:
            p1.connected_room_openings.add((pos))
        return p1
