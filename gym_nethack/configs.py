from keras.optimizers import Adam
from libs.rl.agents.dqn import DQNAgent
from libs.rl.memory import SequentialMemory
from libs.rl.policy import LinearAnnealedPolicy

from gym_nethack.nhdata import *
from gym_nethack.policies import *

###########
# Configs #
###########
# Note: Last line of file controls which config list is currently active.
# Note: For explanation of config parameters, check the comments in the set_config() method of the particular env.
#       E.g., gym_nethack/envs/combat.py::set_config() describes all combat parameters.

# Note: Each config has two parts: one for the environment (e.g., items/monsters), and one for the agent/model (e.g., learning rate/policies).
#       Parameters for the second part are described throughout ngym.py, if their name is not evocative of their purpose.

###########
######## Combat
###########

# Monster listings for combat env.
# Syntax: ('group-name', ['list', 'of', 'monsters'])

test_monsters_one_to_six = ('suite1-6', ['gas_spore', 'brown_mold', 'acid_blob', 'giant_bat', 'gray_ooze', 'rabid_rat', 'blue_jelly', 'uruk_hai', 'spotted_jelly', 'brown_pudding', 'dwarf_lord', 'raven', 'soldier_ant'])
test_monsters_svn_to_elv = ('suite7-11', ['chickatrice', 'owlbear', 'barrow_wight', 'mumak', 'vampire_bat', 'cockatrice', 'ochre_jelly', 'leocrotta', 'steam_vortex', 'winged_gargoyle', 'elvenking', 'ogre_king', 'marilith'])
test_monsters_twl_to_svt = ('suite12-17', ['black_pudding', 'lurker_above', 'umber_hulk', 'oracle', 'mind_flayer', 'nurse', 'golden_naga', 'disenchanter', 'lich', 'nalfeshnee', 'olog_hai', 'guardian_naga', 'nazgul'])
monster_slice = ('all14-17', [mon for mon, diff in MONSTERS if diff in range(14, 18)])
dragons = ('dragons', [mon for mon, diff in MONSTERS if 'dragon' in mon and 'baby' not in mon])

combat_thesis_configs = [
    ({ # 0
        # env. parameters
        'env_name':          'NetHackCombat-v0',
        'name':              'thesis_dqn',
        'num_actions':       40000,
        'monsters':          ('fireant', ['fire_ant']),
        'items':             ('W', [Weapon('dagger', 'melee', 'iron', 'small', 'uncursed', '', '+0', 'uncursed +0 dagger'), \
                                    Weapon('tsurugi', 'melee', 'metal', 'small', 'uncursed', '', '+0', 'uncursed +0 tsurugi'), \
                                    Item('wand of cancellation', 'wand', 'uncursed', 'uncursed wand of cancellation'), \
                                    Item('wand of locking', 'wand', 'uncursed', 'uncursed wand of locking'), \
                                    Item('wand of make invisible', 'wand', 'uncursed', 'uncursed wand of make invisible') \
                                    ]),
        'item_sampling':     'all',
        'clvl_to_mlvl_diff': -3,
        'dlvl':              10
    }, { # model parameters
        'agent':             DQNAgent,
        'agent_params': {
            'nb_steps_warmup':          4000, # 10%
            'enable_dueling_network':   True,
            'dueling_type':             'max',
            'gamma':                    0.99,
            'delta_clip':               1.,
            'memory':                   SequentialMemory,
            'target_model_update':      400
        },
        'optimizer':         Adam(0.0001),
        'policy':            LinearAnnealedPolicy,
        'test_policy':       EpsGreedyPossibleQPolicy(eps=0),
        'units_d1':          32,
        'units_d2':          16
    }, { # policy parameters
        'inner_policy':      EpsGreedyPossibleQPolicy(),
        'attr':              'eps',
        'value_max':         1,
        'value_min':         0,
        'value_test':        0
    }),
    ({ # 1
        'env_name':          'NetHackCombat-v0',
        'name':              'thesis_baseline',
        'monsters':          ('fireant', ['fire_ant']),
        'items':             ('W', [Weapon('dagger', 'melee', 'iron', 'small', 'uncursed', '', '+0', 'uncursed +0 dagger'), \
                                    Weapon('tsurugi', 'melee', 'metal', 'small', 'uncursed', '', '+0', 'uncursed +0 tsurugi'), \
                                    Item('wand of cancellation', 'wand', 'uncursed', 'uncursed wand of cancellation'), \
                                    Item('wand of locking', 'wand', 'uncursed', 'uncursed wand of locking'), \
                                    Item('wand of make invisible', 'wand', 'uncursed', 'uncursed wand of make invisible') \
                                    ]),
        'item_sampling':     'all',
        'clvl_to_mlvl_diff': -3,
        'dlvl':              10
    }, {
        'test_policy':       FireAntPolicy(),
        'skip_training':     True,
        'learning_agent':    False
    }, {
        
    }),
    ({ # 2
        'env_name':          'NetHackCombat-v0',
        'name':              'thesis_baseline',
        'monsters':          ('lich', ['lich']),
        'items':             ('WASPWa', ITEMS_BY_PRIORITY[0]),
        'item_sampling':     'type',
        'clvl_to_mlvl_diff': -3,
        'fixed_ac':          0,
        'dlvl':              25
    }, {
        'test_policy':       ApproachAttackItemPolicy,
        'skip_training':     True,
        'learning_agent':    False
    }, {
        'equip_armor':       True
    }),
    ({ # 3
        'env_name':          'NetHackCombat-v0',
        'name':              'thesis_dqn',
        'num_actions':       300000,
        'monsters':          ('lich', ['lich']),
        'items':             ('WASPWa', ITEMS_BY_PRIORITY[0]),
        'item_sampling':     'type',
        'clvl_to_mlvl_diff': -3,
        'fixed_ac':          0,
        'dlvl':              25
    }, {
        'agent':             DQNAgent,
        'agent_params': {
            'nb_steps_warmup':          30000, # 10%
            'enable_dueling_network':   True,
            'dueling_type':             'max',
            'gamma':                    0.99,
            'delta_clip':               1.,
            'memory':                   SequentialMemory,
            'target_model_update':      3000,
        },
        'optimizer':         Adam(0.000001),
        'policy':            LinearAnnealedPolicy,
        'test_policy':       EpsGreedyPossibleQPolicy(eps=0),
        'units_d1':          64,
        'units_d2':          32
    }, { # policy parameters
        'inner_policy':      EpsGreedyPossibleQPolicy(),
        'attr':              'eps',
        'value_max':         1,
        'value_min':         0,
        'value_test':        0
    }),
    ({ # 4
        'env_name':          'NetHackCombat-v0',
        'name':              'thesis_baseline',
        'monsters':          monster_slice,
        'items':             ('WASPWa', ITEMS_BY_PRIORITY[0]),
        'item_sampling':     'type',
        'clvl_to_mlvl_diff': 3,
        'fixed_ac':          -15
    }, {
        'test_policy':       ApproachAttackItemPolicy,
        'skip_training':     True,
        'learning_agent':    False
    }, {
        'equip_armor':       True
    }),
    ({ # 5
        'env_name':          'NetHackCombat-v0',
        'name':              'thesis_dqn',
        'num_actions':       2000000,
        'monsters':          monster_slice,
        'items':             ('WASPWa', ITEMS_BY_PRIORITY[0]),
        'item_sampling':     'type',
        'clvl_to_mlvl_diff': 3,
        'fixed_ac':          -15
    }, {
        'agent':             DQNAgent,
        'agent_params': {
            'nb_steps_warmup':          200000, # 10%
            'enable_dueling_network':   True,
            'dueling_type':             'max',
            'gamma':                    0.99,
            'delta_clip':               1.,
            'memory':                   SequentialMemory,
            'target_model_update':      20000
        },
        'optimizer':         Adam(0.000001),
        'policy':            LinearAnnealedPolicy,
        'test_policy':       EpsGreedyPossibleQPolicy(eps=0),
        'units_d1':          64,
        'units_d2':          32
    }, { # policy parameters
        'inner_policy':      EpsGreedyPossibleQPolicy(),
        'attr':              'eps',
        'value_max':         1,
        'value_min':         0,
        'value_test':        0
    }),
]

###########
######## Exploration
###########

exploration_configs = [
    ({ # 0
        # env parameters
        'env_name':               'NetHackExplEnv-v0',
        'num_episodes':           5,
        'save_maps':              False,
        'parse_items':            False,
        'dataset':                'fixed'
    }, { # model parameters
        'test_policy':            GreedyExplorationPolicy,
        'skip_training':          True,
        'learning_agent':         False
    }, { # policy parameters
        'compute_optimal_path':   False,
        'get_food':               False,
        'show_graph':             False
    }),
    ({ # 1
        # env parameters
        'env_name':               'NetHackExplEnv-v0',
        'num_episodes':           5,
        'save_maps':              False,
        'parse_items':            False,
        'secret_rooms':           True,
        'dataset':                'random'
    }, { # model parameters
        'test_policy':            SecretGreedyExplorationPolicy,
        'skip_training':          True,
        'learning_agent':         False
    }, { # policy parameters
        'compute_optimal_path':   False,
        'get_food':               False,
        'show_graph':             False,
        'grid_search':            True,
        'num_episodes_per_combo': 1
    }),
    ({ # 2
        # env parameters
        'env_name':               'NetHackExplEnv-v0',
        'num_episodes':           5,
        'save_maps':              False,
        'parse_items':            False,
        'dataset':                'fixed'
    }, { # model parameters
        'test_policy':            OccupancyMapPolicy,
        'skip_training':          True,
        'learning_agent':         False
    }, { # policy parameters
        'compute_optimal_path':   False,
        'get_food':               False,
        'show_graph':             True,
        'grid_search':            False,
        'top_models':             False,
        #'num_episodes_per_combo': 1
    }),
    ({ # 3
        # env parameters
        'env_name':               'NetHackExplEnv-v0',
        'num_episodes':           5,
        'save_maps':              False,
        'parse_items':            False,
        'secret_rooms':           True,
        'dataset':                'fixed'
    }, { # model parameters
        'test_policy':            SecretOccupancyMapPolicy,
        'skip_training':          True,
        'learning_agent':         False
    }, { # policy parameters
        'compute_optimal_path':   False,
        'get_food':               False,
        'show_graph':             True,
        'grid_search':            False,
        'top_models':             False,
        #'num_episodes_per_combo': 1
    })
]


###########
######## Level
###########

level_configs = [
    ({ # 0
        # env parameters
        'env_name':          'NetHackLevel-v0',
        'name':              'level',
        'num_episodes':      200,
        'num_actions':       300000,
        'monsters':          ('arenabaseline', []),
        'items':             ('WASPWa', ITEMS_BY_PRIORITY[0]),
        'item_sampling':     'type'
    }, { # model parameters
        'test_policy':       LevelPolicy,
        #'memory':            SequentialMemory,
        #'lr':                0.001,
        #'units_d1':          32,
        'skip_training':     True,
        'learning_agent':    False
    }, { # policy parameters
        'combat_policy':     ApproachAttackItemPolicy,
        'combat_policy_params': {
            'equip_armor':            True
        },
        'exploration_policy': GreedyExplorationPolicy,
        'exploration_policy_params': {
            'compute_optimal_path':   False,
            'get_food':               False,
            'show_graph':             False
        }
    })
]

configs = combat_thesis_configs
#configs = exploration_configs
#configs = level_configs