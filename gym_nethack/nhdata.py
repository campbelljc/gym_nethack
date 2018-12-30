from string import ascii_letters
from collections import namedtuple

from gym_nethack.conn import nethack_dir
from gym_nethack.fileio import read_line_list

############
# Map Info #
############

COLNO = 80
ROWNO = 21

ITEM_CHAR = ['*']
WALL_CHARS = ['|', '-', ' ']
DOOR_CHARS = ['+', '#']
PASSABLE_CHARS = ['+', '#', '.', '^', '>', '<', '_', ITEM_CHAR] #, '@']
ROOM_CHARS = ['.', '>', '<', '^', ITEM_CHAR]
CORRIDOR_CHARS = ['#', '`', '^']
MONS_CHARS = list(ascii_letters) + [':', '&', ']', ';', '\'', '@']
TOPOGRAPHICAL_CHARS = ['|', '-', '+', '#', '.', '>', '<']

ROOM_OPENING_GLYPHS = [2377, 2378, 2379]
CORRIDOR_GLYPHS = [2386, 2387]

############
# Monsters #
############

NH_MONS = ["giant_ant", "killer_bee", "soldier_ant", "fire_ant", "giant_beetle", "queen_bee", "acid_blob", "quivering_blob", "gelatinous_cube", "chickatrice", "cockatrice", "pyrolisk", "jackal", "fox", "coyote", "werejackal", "little_dog", "dingo", "dog", "large_dog", "wolf", "werewolf", "winter_wolf_cub", "warg", "winter_wolf", "hell_hound_pup", "hell_hound", "gas_spore", "floating_eye", "freezing_sphere", "flaming_sphere", "shocking_sphere", "kitten", "housecat", "jaguar", "lynx", "panther", "large_cat", "tiger", "gremlin", "gargoyle", "winged_gargoyle", "hobbit", "dwarf", "bugbear", "dwarf_lord", "dwarf_king", "mind_flayer", "master_mind_flayer", "manes", "homunculus", "imp", "lemure", "quasit", "tengu", "blue_jelly", "spotted_jelly", "ochre_jelly", "kobold", "large_kobold", "kobold_lord", "kobold_shaman", "leprechaun", "small_mimic", "large_mimic", "giant_mimic", "wood_nymph", "water_nymph", "mountain_nymph", "goblin", "hobgoblin", "orc", "hill_orc", "mordor_orc", "uruk-hai", "orc_shaman", "orc-captain", "rock_piercer", "iron_piercer", "glass_piercer", "rothe", "mumak", "leocrotta", "wumpus", "titanothere", "baluchitherium", "mastodon", "sewer_rat", "giant_rat", "rabid_rat", "wererat", "rock_mole", "woodchuck", "cave_spider", "centipede", "giant_spider", "scorpion", "lurker_above", "trapper", "pony", "white_unicorn", "gray_unicorn", "black_unicorn", "horse", "warhorse", "fog_cloud", "dust_vortex", "ice_vortex", "energy_vortex", "steam_vortex", "fire_vortex", "baby_long_worm", "baby_purple_worm", "long_worm", "purple_worm", "grid_bug", "xan", "yellow_light", "black_light", "zruty", "couatl", "aleax", "angel", "ki-rin", "archon", "bat", "giant_bat", "raven", "vampire_bat", "plains_centaur", "forest_centaur", "mountain_centaur", "baby_gray_dragon", "baby_silver_dragon", "baby_red_dragon", "baby_white_dragon", "baby_orange_dragon", "baby_black_dragon", "baby_blue_dragon", "baby_green_dragon", "baby_yellow_dragon", "gray_dragon", "silver_dragon", "red_dragon", "white_dragon", "orange_dragon", "black_dragon", "blue_dragon", "green_dragon", "yellow_dragon", "stalker", "air_elemental", "fire_elemental", "earth_elemental", "water_elemental", "lichen", "brown_mold", "yellow_mold", "green_mold", "red_mold", "shrieker", "violet_fungus", "gnome", "gnome_lord", "gnomish_wizard", "gnome_king", "giant", "stone_giant", "hill_giant", "fire_giant", "frost_giant", "ettin", "storm_giant", "titan", "minotaur", "jabberwock", "keystone_kop", "kop_sergeant", "kop_lieutenant", "kop_kaptain", "lich", "demilich", "master_lich", "arch-lich", "kobold_mummy", "gnome_mummy", "orc_mummy", "dwarf_mummy", "elf_mummy", "human_mummy", "ettin_mummy", "giant_mummy", "red_naga_hatchling", "black_naga_hatchling", "golden_naga_hatchling", "guardian_naga_hatchling", "red_naga", "black_naga", "golden_naga", "guardian_naga", "ogre", "ogre_lord", "ogre_king", "gray_ooze", "brown_pudding", "green_slime", "black_pudding", "quantum_mechanic", "rust_monster", "disenchanter", "garter_snake", "snake", "water_moccasin", "python", "pit_viper", "cobra", "troll", "ice_troll", "rock_troll", "water_troll", "olog-hai", "umber_hulk", "vampire", "vampire_lord", "vlad_the_impaler", "barrow_wight", "wraith", "nazgul", "xorn", "monkey", "ape", "owlbear", "yeti", "carnivorous_ape", "sasquatch", "kobold_zombie", "gnome_zombie", "orc_zombie", "dwarf_zombie", "elf_zombie", "human_zombie", "ettin_zombie", "ghoul", "giant_zombie", "skeleton", "straw_golem", "paper_golem", "rope_golem", "gold_golem", "leather_golem", "wood_golem", "flesh_golem", "clay_golem", "stone_golem", "glass_golem", "iron_golem", "human", "human_wererat", "human_werejackal", "human_werewolf", "elf", "woodland-elf", "green-elf", "grey-elf", "elf-lord", "elvenking", "doppelganger", "shopkeeper", "guard", "prisoner", "oracle", "aligned_priest", "high_priest", "soldier", "sergeant", "nurse", "lieutenant", "captain", "watchman", "watch_captain", "medusa", "wizard_of_yendor", "croesus", "ghost", "shade", "water_demon", "succubus", "horned_devil", "incubus", "erinys", "barbed_devil", "marilith", "vrock", "hezrou", "bone_devil", "ice_devil", "nalfeshnee", "pit_fiend", "sandestin", "balrog", "juiblex", "yeenoghu", "orcus", "geryon", "dispater", "baalzebub", "asmodeus", "demogorgon", "death", "pestilence", "famine", "mail_daemon", "djinni", "jellyfish", "piranha", "shark", "giant_eel", "electric_eel", "kraken", "newt", "gecko", "iguana", "baby_crocodile", "lizard", "chameleon", "crocodile", "salamander", "tail_of_a_long_worm", "archeologist", "barbarian", "caveman", "cavewoman", "healer", "knight", "monk", "priest", "priestess", "ranger", "rogue", "samurai", "tourist", "valkyrie", "wizard", "lord_carnarvon", "pelias", "shaman_karnov", "hippocrates", "king_arthur", "grand_master", "arch_priest", "orion", "master_of_thieves", "lord_sato", "twoflower", "norn", "neferet_the_green", "minion_of_huhetotl", "thoth_amon", "chromatic_dragon", "cyclops", "ixoth", "master_kaen", "nalzok", "scorpius", "master_assassin", "ashikaga_takauji", "lord_surtur", "dark_one", "student", "chieftain", "neanderthal", "attendant", "page", "abbot", "acolyte", "hunter", "thug", "ninja", "roshi", "guide", "warrior", "apprentice"] #"remembered_unseen_creature"]
#"
IGNORE_MONS = ["ixoth", "lord_surtur", "sandestin", "priestess", "giant_mimic", "large_mimic", "short_mimic", "ashikaga_takauji", "stone_golem", "iron_golem", "earth_elemental", "master_assassin", "scorpius", "thoth_amon", "chromatic_dragon", "master_kaen", "mionion_of_huhetotl", "cyclops", "nalzok", "master_of_thieves", "lord_surtur", "dark_one", "human_wererat", "tail_of_a_long_worm", "human", "orc", "keystone_kop", "human_werejackal", "woodchuck", "red_naga_hatchling", "guardian_naga_hatchling", "golden_naga_hatchling", "black_naga_hatchling", "baby_crocodile", "kop_sergeant", "jellyfish", "kop_lieutenant", "human_werewolf", "piranha", "kop_kaptain", "giant_eel", "warrior", "thug", "roshi", "page", "ninja", "student", "neanderthal", "hunter", "chieftain", "attendant", "water_moccasin", "giant", "djinni", "watchman", "guide", "apprentice", "acolyte", "abbot", "black_light", "chameleon", "baby_long_worm", "stalker", "baby_purple_worm", "monk", "water_demon", "doppelganger", "ghost", "watch_captain", "valkyrie", "tourist", "samurai", "rogue", "knight", "barbarian", "archeologist", "wizard", "ranger", "priest", "priestess", "healer", "elf", "caveman", "cavewoman", "salamander", "water_troll", "skeleton", "shade", "guard", "prisoner", "sandestin", "aligned priest", "shopkeeper", "minotaur", "scorpius", "purple_worm", "vlad_the_impaler", "ashikaga_takauji", "lord_surtur", "master_assassin", "dark_one", "vampire_lord", "aligned_priest", "guardian_naga"]

# quest monsters -> ignored due to dialogue
# long worm tail -> ignored (monster part)
# human -> frequency=0
# black light,stalker: invisible at start...
# chameleon,doppelganger,sandestin: changes to diff monster...
# guardian naga -> can cause paralysis which causes bug

# src: monstr.c
NH_MONS_DIFF = [4,  5,  6,  6,  6, 12,  2,  6,  8,  7,  8,  8,  1,  1,  2,  4,
 3,  5,  5,  7,  6,  7,  7,  8,  9,  9, 14,  2,  3,  8,  8,  8,
 3,  5,  6,  7,  7,  7,  8,  8,  8, 11,  2,  4,  5,  6,  8, 13,
19,  3,  3,  4,  5,  7,  7,  5,  6,  8,  1,  2,  3,  4,  4,  8,
 9, 11,  5,  5,  5,  1,  3,  3,  4,  5,  5,  5,  7,  4,  6,  9,
 4,  7,  8,  9, 13, 15, 22,  1,  2,  4,  4,  4,  4,  3,  4,  7,
 8, 12, 14,  4,  6,  6,  6,  7,  9,  4,  6,  7, 10,  9, 10,  6,
 9, 10, 17,  1,  9,  5,  7, 11, 11, 12, 19, 21, 26,  2,  3,  6,
 7,  6,  8,  9, 13, 13, 13, 13, 13, 13, 13, 13, 13, 20, 20, 20,
20, 20, 20, 20, 20, 20,  9, 10, 10, 10, 10,  1,  2,  2,  2,  2,
 2,  5,  3,  4,  5,  6,  8,  8, 10, 11, 13, 13, 19, 20, 17, 18,
 3,  4,  5,  6, 14, 18, 21, 29,  4,  5,  6,  6,  7,  7,  8, 10,
 4,  4,  4,  4,  8, 10, 13, 16,  7,  9, 11,  4,  6,  8, 12,  9,
 8, 14,  3,  6,  7,  8,  9, 10,  9, 12, 12, 13, 16, 12, 12, 14,
18,  7,  8, 17, 11,  4,  6,  7,  7,  8,  9,  1,  2,  3,  3,  4,
 5,  7,  5,  9, 14,  4,  4,  6,  6,  7,  8, 10, 12, 15, 18, 22,
 2,  3,  3,  6, 12,  6,  7,  8, 11, 11, 11, 15, 14, 14, 13, 15,
30,  8, 10, 13, 12, 14,  8, 12, 25, 34, 22, 12, 14, 11,  8,  9,
 8, 10, 10, 11, 11, 12, 13, 14, 15, 16, 15, 20, 26, 31, 36, 36,
40, 45, 53, 57, 34, 34, 34, 26,  8,  5,  6,  9,  7, 10, 22,  1,
 2,  3,  4,  6,  7,  7, 12,  1, 12, 12, 12, 12, 12, 12, 11, 12,
12, 12, 12, 12, 12, 12, 12, 22, 22, 22, 22, 23, 30, 30, 22, 24,
23, 22, 23, 23, 23, 22, 23, 23, 22, 31, 23, 17, 20, 19, 19, 20,
 7,  7,  7,  7,  7,  8,  8,  7,  7,  7,  7,  8,  7,  8] #, -1]
 
assert len(NH_MONS) == len(NH_MONS_DIFF)

MONSTERS = [(mon, diff) for mon, diff in zip(NH_MONS, NH_MONS_DIFF) if mon not in IGNORE_MONS]
MONSTERS.sort(key=lambda tup: tup[1])
MONSTER_NAMES = [m[0] for m in MONSTERS]
MONSTER_DIFFICULTIES = [m[1] for m in MONSTERS]
MONSTER_NAMES_HALLU = [name.replace("\n", "").replace(" ", "_").replace("+", "").replace("=", "").replace("|", "").lower() for name in read_line_list(nethack_dir + '/bogusmon', load_float=False)]

SHOPKEEPER_NAMES = ["Njezjin", "Tsjernigof", "Ossipewsk", "Gorlowka", "Gomel", "Konosja", "Weliki Oestjoeg", "Syktywkar", "Sablja", "Narodnaja", "Kyzyl", "Walbrzych", "Swidnica", "Klodzko", "Raciborz", "Gliwice", "Brzeg", "Krnov", "Hradec Kralove", "Leuk", "Brig", "Brienz", "Thun", "Sarnen", "Burglen", "Elm", "Flims", "Vals", "Schuls", "Zum Loch", "Skibbereen", "Kanturk", "Rath Luirc", "Ennistymon", "Lahinch", "Kinnegad", "Lugnaquillia", "Enniscorthy", "Gweebarra", "Kittamagh", "Nenagh", "Sneem", "Ballingeary", "Kilgarvan", "Cahersiveen", "Glenbeigh", "Kilmihil", "Kiltamagh", "Droichead Atha", "Inniscrone", "Clonegal", "Lisnaskea", "Culdaff", "Dunfanaghy", "Inishbofin", "Kesh", "Demirci", "Kalecik", "Boyabai", "Yildizeli", "Gaziantep", "Siirt", "Akhalataki", "Tirebolu", "Aksaray", "Ermenak", "Iskenderun", "Kadirli", "Siverek", "Pervari", "Malasgirt", "Bayburt", "Ayancik", "Zonguldak", "Balya", "Tefenni", "Artvin", "Kars", "Makharadze", "Malazgirt", "Midyat", "Birecik", "Kirikkale", "Alaca", "Polatli", "Nallihan", "Yr Wyddgrug", "Trallwng", "Mallwyd", "Pontarfynach", "Rhaeader", "Llandrindod", "Llanfair-ym-muallt", "Y-Fenni", "Maesteg", "Rhydaman", "Beddgelert", "Curig", "Llanrwst", "Llanerchymedd", "Caergybi", "Nairn", "Turriff", "Inverurie", "Braemar", "Lochnagar", "Kerloch", "Beinn a Ghlo", "Drumnadrochit", "Morven", "Uist", "Storr", "Sgurr na Ciche", "Cannich", "Gairloch", "Kyleakin", "Dunvegan", "Feyfer", "Flugi", "Gheel", "Havic", "Haynin", "Hoboken", "Imbyze", "Juyn", "Kinsky", "Massis", "Matray", "Moy", "Olycan", "Sadelin", "Svaving", "Tapper", "Terwen", "Wirix", "Ypey", "Rastegaisa", "Varjag Njarga", "Kautekeino", "Abisko", "Enontekis", "Rovaniemi", "Avasaksa", "Haparanda", "Lulea", "Gellivare", "Oeloe", "Kajaani", "Fauske", "Djasinga", "Tjibarusa", "Tjiwidej", "Pengalengan", "Bandjar", "Parbalingga", "Bojolali", "Sarangan", "Ngebel", "Djombang", "Ardjawinangun", "Berbek", "Papar", "Baliga", "Tjisolok", "Siboga", "Banjoewangi", "Trenggalek", "Karangkobar", "Njalindoeng", "Pasawahan", "Pameunpeuk", "Patjitan", "Kediri", "Pemboeang", "Tringanoe", "Makin", "Tipor", "Semai", "Berhala", "Tegal", "Samoe", "Voulgezac", "Rouffiac", "Lerignac", "Touverac", "Guizengeard", "Melac", "Neuvicq", "Vanzac", "Picq", "Urignac", "Corignac", "Fleac", "Lonzac", "Vergt", "Queyssac", "Liorac", "Echourgnac", "Cazelon", "Eypau", "Carignan", "Monbazillac", "Jonzac", "Pons", "Jumilhac", "Fenouilledes", "Laguiolet", "Saujon", "Eymoutiers", "Eygurande", "Eauze", "Labouheyre", "Ymla", "Eed-morra", "Cubask", "Nieb", "Bnowr Falr", "Telloc Cyaj", "Sperc", "Noskcirdneh", "Yawolloh", "Hyeghu", "Niskal", "Trahnil", "Htargcm", "Enrobwem", "Kachzi Rellim", "Regien", "Donmyar", "Yelpur", "Nosnehpets", "Stewe", "Renrut", "Zlaw", "Nosalnef", "Rewuorb", "Rellenk", "Yad", "Cire Htims", "Y-crad", "Nenilukah", "Corsh", "Aned", "Niknar", "Lapu", "Lechaim", "Rebrol-nek", "AlliWar Wickson", "Oguhmk", "Erreip", "Nehpets", "Mron", "Snivek", "Kahztiy", "Lexa", "Niod", "Nhoj-lee", "Evad\'kh", "Ettaw-noj", "Tsew-mot", "Ydna-s", "Yao-hang", "Tonbar", "Kivenhoug", "Llardom", "Falo", "Nosid-da\'r", "Ekim-p", "Noslo", "Yl-rednow", "Mured-oog", "Ivrajimsal", "Nivram", "Nedraawi-nav", "Lez-tneg", "Ytnu-haled", "Zarnesti", "Slanic", "Nehoiasu", "Ludus", "Sighisoara", "Nisipitu", "Razboieni", "Bicaz", "Dorohoi", "Vaslui", "Fetesti", "Tirgu Neamt", "Babadag", "Zimnicea", "Zlatna", "Jiu", "Eforie", "Mamaia", "Silistra", "Tulovo", "Panagyuritshte", "Smolyan", "Kirklareli", "Pernik", "Lom", "Haskovo", "Dobrinishte", "Varvara", "Oryahovo", "Troyan", "Lovech", "Sliven", "Hebiwerie", "Possogroenoe", "Asidonhopo", "Manlobbi", "Adjama", "Pakka Pakka", "Kabalebo", "Wonotobo", "Akalapi", "Sipaliwini", "Annootok", "Upernavik", "Angmagssalik", "Aklavik", "Inuvik", "Tuktoyaktuk", "Chicoutimi", "Ouiatchouane", "Chibougamau", "Matagami", "Kipawa", "Kinojevis", "Abitibi", "Maganasipi", "Akureyri", "Kopasker", "Budereyri", "Akranes", "Bordeyri", "Holmavik", "Ga'er", "Zhangmu", "Rikaze", "Jiangji", "Changdu", "Linzhi", "Shigatse", "Gyantse", "Ganden", "Tsurphu", "Lhasa", "Tsedong", "Drepung", "=Azura", "=Blaze", "=Breanna", "=Breezy", "=Dharma", "=Feather", "=Jasmine", "=Luna", "=Melody", "=Moonjava", "=Petal", "=Rhiannon", "=Starla", "=Tranquilla", "=Windsong", "=Zennia", "=Zoe", "=Zora"]
#"
SHOPKEEPER_NAMES = [mon.replace(" ", "_").replace("=", "") for mon in SHOPKEEPER_NAMES]

#########
# Items #
#########

BUC_WORDS = ['uncursed', 'cursed', 'blessed']
ENCHANTMENT_LEVELS = ['-1', '+0', '+1']
MATERIALS = ['bronze', 'cloth', 'dragon hide', 'glass', 'iron', 'leather', 'mithril', 'silver', 'wood', 'copper', 'plastic', 'mineral', 'metal']
MATERIAL_EROSION = [('bronze', ['corroded']), ('cloth', ['burnt', 'rotted']), ('dragon hide', []), ('glass', []), ('iron', ['rusty', 'corroded']), ('leather', ['burnt', 'rotted']), ('mithril', []), ('silver', []), ('wood', ['burnt', 'rotted']), ('copper', ['corroded']), ('plastic', ['burnt']), ('mineral', []), ('metal', [])]
assert len(MATERIALS) == len(MATERIAL_EROSION)

# src: http://stackoverflow.com/questions/11351032/named-tuple-and-optional-keyword-arguments
Weapon = namedtuple('Weapon', 'name type material dsize buc condition enchantment full_name')
Weapon.__new__.__defaults__ = (None,) * len(Weapon._fields)

#WEAPONS_MELEE_ARTIFACT = ['Cleaver', 'Demonbane', 'Excalibur', 'Fire Brand', 'Frost Brand', 'Giantslayer', 'Grayswandir',  'Grimtooth', 'Magicbane', 'Mjollnir', 'Ogresmasher', 'Orcrist', 'Sceptre of Might', 'Sunsword', 'Tsurugi of Muramasa', 'Vorpal Blade', 'Werebane', 'Snickersnee', 'Trollsbane', 'Dragonbane', 'Staff of Aesculapius', 'Sting'] # Stormbringer

WEAPONS_MELEE = [('orcish dagger', 'iron', 'equal'), ('dagger', 'iron', 'small'), ('silver dagger', 'silver', 'small'), ('athame', 'iron', 'small'), ('elven dagger', 'wood', 'small'), ('knife', 'iron', 'small'), ('stiletto', 'iron', 'small'), ('scalpel', 'metal', 'equal'), ('axe', 'iron', 'small'), ('battle-axe', 'iron', 'large'), ('pick-axe', 'iron', 'small'), ('dwarvish mattock', 'iron', 'large'), ('orcish short sword', 'iron', 'large'), ('dwarvish short sword', 'iron', 'large'), ('short sword', 'iron', 'large'), ('elven short sword', 'wood', 'equal'), ('broadsword', 'iron', 'equal'), ('elven broadsword', 'wood', 'small'), ('long sword', 'iron', 'large'), ('katana', 'iron', 'large'), ('two-handed sword', 'iron', 'large'), ('tsurugi', 'metal', 'small'), ('scimitar', 'iron', 'equal'), ('silver saber', 'silver', 'equal'), ('club', 'wood', 'small'), ('aklys', 'iron', 'small'), ('mace', 'iron', 'small'), ('morning star', 'iron', 'equal'), ('flail', 'iron', 'equal'), ('grappling hook', 'iron', 'large'), ('war hammer', 'iron', 'small'), ('quarterstaff', 'wood', 'equal'), ('orcish spear', 'iron', 'large'), ('silver spear', 'silver', 'large'), ('dwarvish spear', 'iron', 'equal'), ('elven spear', 'wood', 'large'), ('spear', 'iron', 'large'), ('javelin', 'iron', 'equal'), ('trident', 'iron', 'large'), ('lance', 'iron', 'large')]

#'runesword'   - not randomly generated
#'rubber hose' - as above
#'worm tooth'  - as above
#'crysknife'
#'bullwhip'
#'unicorn horn'

#NH_POLEARMS = ['partisan', 'fauchard', 'glaive', 'bec-de-corbin', 'spetum', 'lucern hammer', 'guisarme', 'ranseur', 'voulge', 'bill-guisarme', 'bardiche', 'halberd']

FIRING_WEAPONS = [('bow', 'wood', 'equal'), ('elven bow', 'wood', 'equal'), ('orcish bow', 'wood', 'equal'), ('yumi', 'wood', 'equal'), ('crossbow', 'wood', 'large'), ('sling', 'leather', 'equal')]

#NH_THROWING_WEAPONS = []
#FIRING_WEAPONS_ARTIFACT = [('Longbow of Diana', 'arrow')]

PROJECTILE_NAMES = [('elven arrow', 'wood'), ('orcish arrow', 'iron'), ('silver arrow', 'silver'), ('arrow', 'iron'), ('ya', 'metal'), ('crossbow bolt', 'iron'), ('rock', 'mineral'), ('gem', 'mineral'), ('flint stone', 'mineral')]

def get_full_name(name, enchantment, condition, buc):
    if len(enchantment) > 0:
        if 'ring of' not in name or '+0' not in enchantment:
            name = enchantment + " " + name
    if len(condition) > 0:
        name = condition + " " + name
    if 'holy water' in name:
        return name
    name = buc + " " + name
    return name

PROJECTILES = []
for weap_name, weap_material in PROJECTILE_NAMES:
    for buc in BUC_WORDS:
        if weap_name in ['rock', 'gem', 'flint stone']:
            PROJECTILES.append(Weapon(weap_name, 'projectile', weap_material, None, buc, '', '', get_full_name(weap_name, '', '', buc)))
            continue
        for cond in ['poisoned', '']:
            for enchantment in ENCHANTMENT_LEVELS:
                PROJECTILES.append(Weapon(weap_name, 'projectile', weap_material, None, buc, cond, enchantment, get_full_name(weap_name, enchantment, cond, buc)))

ALL_WEAPONS_MELEE = [] #WEAPONS_MELEE # + WEAPONS_MELEE_ARTIFACT
for weap, mat, dsize in WEAPONS_MELEE: #range(len(ALL_WEAPONS_MELEE)):
    ALL_WEAPONS_MELEE.append(Weapon(weap, "melee", mat, dsize))

ALL_WEAPONS_RANGED = [] #FIRING_WEAPONS # + FIRING_WEAPONS_ARTIFACT + NH_THROWING_WEAPONS
for weapname, mat, dsize in FIRING_WEAPONS:
    ALL_WEAPONS_RANGED.append(Weapon(weapname, "ranged", mat, dsize))

WEAPONS = []
for weap in ALL_WEAPONS_MELEE + ALL_WEAPONS_RANGED: #+ NH_POLEARMS
    for buc in BUC_WORDS:
        for enchantment in ENCHANTMENT_LEVELS:
            WEAPONS.append(Weapon(weap.name, weap.type, weap.material, weap.dsize, buc, '', enchantment, get_full_name(weap.name, enchantment, '', buc)))
            
            for erode_word in MATERIAL_EROSION[MATERIALS.index(weap.material)][1]:
                #WEAPONS.append(Weapon(weap.name, weap.type, weap.material, weap.dsize, buc, 'very '+erode_word, enchantment, get_full_name(weap.name, enchantment, 'very '+erode_word, buc)))
                WEAPONS.append(Weapon(weap.name, weap.type, weap.material, weap.dsize, buc, erode_word, enchantment, get_full_name(weap.name, enchantment, erode_word, buc)))

RANGED_WEAP_NAMES = [weap.name for weap in ALL_WEAPONS_RANGED]
WEAPON_NAMES = [weap.name for weap in WEAPONS]

# other items

#NH_ATTACK_SPELLBOOKS = ['spellbook of force bolt', 'spellbook of drain life', 'spellbook of magic missile', 'spellbook of cone of cold', 'spellbook of fireball', 'spellbook of finger of death']

Potion = namedtuple('Potion', 'name type buc use_type full_name')
Potion.__new__.__defaults__ = (None,) * len(Potion._fields)

POTION_NAMES = ['booze', 'sickness', 'confusion', 'extra healing', 'hallucination', 'healing', 'restore ability', 'sleeping', 'blindness', 'gain energy', 'monster detection', 'full healing', 'acid', 'gain ability', 'gain level', 'invisibility']
# 'levitation', 'see invisible', 'speed', 'enlightenment', 'polymorph'
# 'oil', 'fruit juice', 'unholy water', 'holy water', 'water', 'object detection', 'paralysis',

POTIONS = []
for name in POTION_NAMES:
    for buc in BUC_WORDS:
        if 'holy water' in name:
            if buc == 'uncursed':
                POTIONS.append(Potion('potion of ' + name, 'potion', buc, None, 'potion of '+name))
        else:
            POTIONS.append(Potion('potion of ' + name, 'potion', buc, None, get_full_name('potion of '+name, '', '', buc)))

Item = namedtuple('Item', 'name type buc full_name')
Item.__new__.__defaults__ = (None,) * len(Item._fields)

SCROLL_NAMES = ['light', 'confuse monster', 'destroy armor', 'fire', 'food detection', 'gold detection', 'scare monster', 'punishment', 'remove curse']
# 'identify', 'charging', 'genocide', 'stinking cloud', 'earth', 'amnesia', 'enchant weapon', 'enchant armor'
# 'mail', 'magic mapping', 'create monster', 'teleportation', 'blank paper', 'taming',

SCROLLS = []
for name in SCROLL_NAMES:
    for buc in BUC_WORDS:
        SCROLLS.append(Item('scroll of ' + name, 'scroll', buc, get_full_name('scroll of '+name, '', '', buc)))

WAND_NAMES = ['magic missile', 'make invisible', 'opening', 'slow monster', 'speed monster', 'striking', 'undead turning', 'cold', 'fire', 'lightning', 'sleep', 'cancellation', 'polymorph', 'death', 'locking']
# 'nothing', 'light', enlightenment', 'secret door detection', 'create monster', 'wishing'
# 'digging',  'teleportation', 'probing',

WANDS = []
for name in WAND_NAMES:
    for buc in BUC_WORDS:
        WANDS.append(Item('wand of ' + name, 'wand', buc, get_full_name('wand of '+name, '', '', buc)))

RING_NAMES = ['ring of protection', 'ring of protection from shape changers', 'ring of increase accuracy', 'ring of increase damage', 'ring of invisibility', 'ring of see invisible', 'ring of free action']

Ring = namedtuple('Ring', 'name type buc condition enchantment full_name')
RINGS = []
for name in RING_NAMES:
    for buc in BUC_WORDS:
        for enchantment in ENCHANTMENT_LEVELS:
            RINGS.append(Ring(name, 'ring', buc, '', enchantment, get_full_name(name, enchantment, '', buc)))

# armor

ARMOR_TYPES = ['helm', 'cloak', 'body armor', 'shirt', 'shield', 'gloves', 'boots']

HELM_NAMES = ('helm', [('fedora', 'cloth'), ('dunce cap', 'cloth'), ('cornuthaum', 'cloth'), ('elven leather helm', 'leather'), ('orcish helm', 'iron'), ('dented pot', 'iron'), ('dwarvish iron helm', 'iron'), ('helmet', 'iron'), ('helm of brilliance', 'iron'), ('helm of opposite alignment', 'iron'), ('helm of telepathy', 'iron')])

CLOAK_NAMES = ('cloak', [('mummy wrapping', 'cloth'), ('orcish cloak', 'cloth'), ('dwarvish cloak', 'cloth'), ('leather cloak', 'leather'), ('cloak of displacement', 'cloth'), ('oilskin cloak', 'cloth'), ('alchemy smock', 'cloth'), ('cloak of invisibility', 'cloth'), ('cloak of magic resistance', 'cloth'), ('elven cloak', 'cloth'), ('robe', 'cloth'), ('cloak of protection', 'cloth')])

BODY_ARMOR_NAMES = ('body armor', [('leather jacket', 'leather'), ('leather armor', 'leather'), ('orcish ring mail', 'iron'), ('studded leather armor', 'leather'), ('ring mail', 'iron'), ('scale mail', 'iron'), ('orcish chain mail', 'iron'), ('chain mail', 'iron'), ('elven mithril-coat', 'mithril'), ('splint mail', 'iron'), ('banded mail', 'iron'), ('dwarvish mithril-coat', 'mithril'), ('bronze plate mail', 'bronze'), ('plate mail', 'iron'), ('crystal plate mail', 'glass'), ('blue dragon scale mail', 'dragon hide'), ('black dragon scale mail', 'dragon hide'), ('gray dragon scale mail', 'dragon hide'), ('green dragon scale mail', 'dragon hide'), ('orange dragon scale mail', 'dragon hide'), ('red dragon scale mail', 'dragon hide'), ('silver dragon scale mail', 'dragon hide'), ('white dragon scale mail', 'dragon hide'), ('yellow dragon scale mail', 'dragon hide')])
#('set of blue dragon scales', 'dragon hide'), ('set of black dragon scales', 'dragon hide'), ('set of gray dragon scales', 'dragon hide'), ('set of green dragon scales', 'dragon hide'), ('set of orange dragon scales', 'dragon hide'), ('set of red dragon scales', 'dragon hide'), ('set of silver dragon scales', 'dragon hide'), ('set of white dragon scales', 'dragon hide'), ('set of yellow dragon scales', 'dragon hide'),

SHIRT_NAMES = ('shirt', [('T-shirt', 'cloth'), ('Hawaiian shirt', 'cloth')])

SHIELD_NAMES = ('shield', [('orcish shield', 'iron'), ('large shield', 'iron'), ('dwarvish roundshield', 'iron'), ('elven shield', 'wood'), ('small shield', 'wood'), ('shield of reflection', 'silver')])
# ('Uruk-hai shield', 'iron') : bug with wishing for it ?

GLOVE_NAMES = ('gloves', [('gauntlets of dexterity', 'leather'), ('gauntlets of fumbling', 'leather'), ('gauntlets of power', 'iron'), ('leather gloves', 'leather')])

BOOT_NAMES = ('boots', [('low boots', 'leather'), ('high boots', 'leather'), ('iron shoes', 'iron'), ('elven boots', 'leather'), ('kicking boots', 'iron'), ('fumble boots', 'leather'), ('levitation boots', 'leather'), ('jumping boots', 'leather'), ('speed boots', 'leather'), ('water walking boots', 'leather')])

ARMOR_NAMES = [HELM_NAMES, CLOAK_NAMES, BODY_ARMOR_NAMES, SHIRT_NAMES, SHIELD_NAMES, GLOVE_NAMES, BOOT_NAMES]

Armor = namedtuple('Armor', 'name type material buc condition enchantment full_name')
Armor.__new__.__defaults__ = (None,) * len(Armor._fields)

ARMOR = []
for armor_type, armors in ARMOR_NAMES:
    for armor_name, armor_material in armors:
        if armor_type in ['gloves', 'boots']:
            armor_name = 'pair of ' + armor_name
        for buc in BUC_WORDS:
            for enchantment in ENCHANTMENT_LEVELS:
                ARMOR.append(Armor(armor_name, armor_type, armor_material, buc, '', enchantment, get_full_name(armor_name, enchantment, '', buc)))
                for erode_word in MATERIAL_EROSION[MATERIALS.index(armor_material)][1]:
                    #ARMOR.append(Armor(armor_name, armor_type, armor_material, buc, 'very '+erode_word, enchantment, get_full_name(armor_name, enchantment, 'very '+erode_word, buc)))
                    ARMOR.append(Armor(armor_name, armor_type, armor_material, buc, erode_word, enchantment, get_full_name(armor_name, enchantment, erode_word, buc)))

MISC = [
    Item('heavy iron ball', 'misc', 'uncursed', 'uncursed heavy iron ball'),
    Item('heavy iron ball', 'misc', 'cursed', 'cursed heavy iron ball'),
    Item('heavy iron ball', 'misc', 'blessed', 'blessed heavy iron ball'),
    Item('food ration', 'food', 'uncursed', 'uncursed food ration'),
    Item('oil lamp', 'food', 'uncursed', 'uncursed oil lamp')
]

ITEMS_BY_PRIORITY = [[] for i in range(3)]
for item in WEAPONS + PROJECTILES + ARMOR + POTIONS + SCROLLS + WANDS + RINGS + MISC:
    if (not isinstance(item, Weapon) and not isinstance(item, Armor) and not isinstance(item, Ring)) or (item.condition == "" and item.enchantment == "+0"):
        if item.buc == 'uncursed':
            priority = 0
        elif item.buc == 'cursed':
            priority = 1
        else:
            priority = 2
    else:
        if 'very' in item.condition:
            priority = 2
        else:
            if item.enchantment == "-1" or item.enchantment == "+0":
                priority = 1
            else:
                priority = 2
    ITEMS_BY_PRIORITY[priority].append(item)

ALL_ITEMS = ITEMS_BY_PRIORITY[0] + ITEMS_BY_PRIORITY[1] + ITEMS_BY_PRIORITY[2]
assert len(ALL_ITEMS) == len(WEAPONS + PROJECTILES + ARMOR + POTIONS + SCROLLS + WANDS + RINGS + MISC)

PLURALS = [("potions", "potion"), ("wands", "wand"), ("rings", "ring"), ("scrolls", "scroll"), ("spears", "spear"), ("daggers", "dagger"), ("scalpels", "scalpel"), ("javelins", "javelin"), ("athames", "athame"), ("teeth", "tooth"), ("stilettos", "stiletto"), ("knives", "knife"), ("arrows", "arrow"), ("bolts", "bolt"), ("rocks", "rock"), ("darts", "dart"), ("rations", "ration")]

IGNORED_ITEMS = ['uncursed small glob of brown pudding', '+0 shuriken', '+0 bill-guisarme', '+0 halberd', '+0 partisan', '+0 fauchard', '+0 glaive', '+0 bec-de-corbin', '+0 spetum', '+0 lucern hammer', '+0 guisarme', '+0 ranseur', '+0 voulge', '+0 bardiche', '+0 dart', '-1 dart', '+1 dart', 'cursed -1 dart', 'blessed -1 dart', 'blessed +0 dart', 'cursed +0 dart', 'cursed +1 dart', 'blessed +1 dart', 'gold pieces']

##############
# Characters #
##############

ROLES = ['Arc', 'Bar', 'Cav', 'Hea', 'Kni', 'Mon', 'Pri', 'Ran', 'Rog', 'Sam', 'Tou', 'Val', 'Wiz', 'Lyc']

NH_ROLE_TITLES = [("Arc", ["Digger", "Field Worker", "Investigator", "Exhumer", "Excavator", "Spelunker", "Speleologist", "Collector", "Curator"]), ("Bar", ["Plunderer", "Plunderess", "Pillager", "Bandit", "Brigand", "Raider", "Reaver", "Slayer", "Chieftain", "Chieftainess", "Conqueror", "Conqueress"]), ("Cav", ["Cavewoman", "Troglodyte", "Aborigine", "Wanderer", "Vagrant", "Wayfarer", "Roamer", "Nomad", "Rover", "Pioneer"]), ("Hea", ["Rhizotomist", "Empiric", "Embalmer", "Dresser", "Medicus ossium", "Medica ossium", "Herbalist", "Magister", "Magistra", "Physician", "Chirurgeon"]), ("Kni", ["Gallant", "Esquire", "Bachelor", "Sergeant", "Knight", "Banneret", "Chevalier", "Chevaliere", "Seignieur", "Dame", "Paladin"]), ("Mon", ["Candidate", "Novice", "Initiate", "Student of Stones", "Student of Waters", "Student of Metals", "Student of Winds", "Student of Fire", "Master"]), ("Pri", ["Priestess",  "Aspirant", "Acolyte", "Adept", "Priest", "Priestess", "Curate", "Canon", "Canoness", "Lama", "Patriarch", "Matriarch", "High Priest", "High Priestess"]), ("Rog", ["Footpad", "Cutpurse", "Rogue", "Pilferer", "Robber", "Burglar", "Filcher", "Magsman", "Magswoman", "Thief"]), ("Ran", ["Tenderfoot", "Lookout", "Trailblazer", "Reconnoiterer", "Reconnoiteress", "Scout", "Arbalester", "Archer", "Sharpshooter", "Marksman", "Markswoman"]), ("Sam", ["Hatamoto", "Ronin", "Ninja", "Kunoichi", "Joshu", "Ryoshu", "Kokushu", "Daimyo", "Kuge", "Shogun"]), ("Tou", ["Rambler", "Sightseer", "Excursionist", "Peregrinator", "Peregrinatrix", "Traveler", "Journeyer", "Voyager", "Explorer", "Adventurer"]), ("Val", ["Stripling", "Skirmisher", "Fighter", "Woman-at-arms", "Warrior", "Swashbuckler", "Hero", "Heroine", "Champion", "Lord", "Lady"]), ("Wiz", ["Evoker", "Conjurer", "Thaumaturge", "Magician", "Enchanter", "Enchantress", "Sorcerer", "Sorceress", "Necromancer", "Wizard", "Mage"]), ("Lyc", ["Werewolf", "Werejackal", "Wererat"])]

def get_role_for_title(title):
    for role, titles in NH_ROLE_TITLES:
        if title in titles:
            return role
    raise Exception("Role title '" + title + "' unrecognized!")

NH_ROLE_TITLES_FLAT = r""
for abbrv, lst in NH_ROLE_TITLES:
    for title in lst:
        NH_ROLE_TITLES_FLAT += title + "|"
NH_ROLE_TITLES_FLAT = NH_ROLE_TITLES_FLAT[:-1]

ALIGNMENTS = ['Ne', 'Ch', 'La']

STATUS_EFFECTS = [('hunger', 'Satiated'), ('hunger', 'Hungry'), ('hunger', 'Weak'), ('hunger', 'Fainting'), ('stun', 'Stun'), ('conf', 'Conf'), ('blind', 'Blind'), ('burden', 'Burdened'), ('burden', 'Stressed'), ('burden', 'Strained'), ('burden', 'Overtaxed'), ('burden', 'Overloaded'), ('hallu', 'Hallu')]

############
# Commands #
############

class CMD:
    # based on https://github.com/lmjohns3/shrieker
    class DIR:
        NW = 'y'
        N = 'k'
        NE = 'u'
        E = 'l'
        SE = 'n'
        S = 'j'
        SW = 'b'
        W = 'h'

        UP = '<'
        DOWN = '>'

    PICKUP = ','
    WAIT = '.'

    APPLY = 'a'
    CLOSE = 'c'
    DROP = 'd'
    EAT = 'e'
    ENGRAVE = 'E'
    FIRE = 'f'
    INVENTORY = '~'
    OPEN = 'o'
    PAY = 'p'
    PUTON = 'P'
    QUAFF = 'q'
    QUIVER = 'Q'
    READ = 'r'
    REMOVE = 'R'
    SEARCH = 's'
    THROW = 't'
    TAKEOFF = 'T'
    WIELD = 'w'
    WEAR = 'W'
    EXCHANGE = 'x'
    ZAP = 'z'
    CAST = 'Z'

    MORE = ' ' #'\x0d'      # ENTER
    KICK = '\x04'      # ^D
    TELEPORT = '\x14'  # ^T

    class SPECIAL:
        CHAT = '#chat'
        DIP = '#dip'
        FORCE = '#force'
        INVOKE = '#invoke'
        JUMP = '#jump'
        LOOT = '#loot'
        MONSTER = '#monster'
        OFFER = '#offer'
        PRAY = '#pray'
        RIDE = '#ride'
        RUB = '#rub'
        SIT = '#sit'
        TURN = '#turn'
        WIPE = '#wipe'

DIR_MAPPING = [((-1, 0), CMD.DIR.N), ((1, 0), CMD.DIR.S), ((0, -1), CMD.DIR.W), ((0, 1), CMD.DIR.E), ((-1, -1), CMD.DIR.NW), ((1, 1), CMD.DIR.SE), ((-1, 1), CMD.DIR.NE), ((1, -1), CMD.DIR.SW)]

DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS_DIAG = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
DIRS_CMDS = [CMD.DIR.N, CMD.DIR.S, CMD.DIR.W, CMD.DIR.E, CMD.DIR.NW, CMD.DIR.SE, CMD.DIR.NE, CMD.DIR.SW]

def wielding(inven_item):
    return 'in hand' in inven_item or '(wielded)' in inven_item

def wearing(inven_item):
    return 'being worn' in inven_item

def equipped(inven_item):
    return wielding(inven_item) or wearing(inven_item)
