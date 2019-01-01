import os, sys

import gym

if sys.platform == "darwin":
    # The following lines enable support for CUDA on OS X. Make sure to edit the paths as necessary.
    print("Darwin detected.\nMake sure to update paths in ngym.py lines 9-12.")
    os.environ['CUDA_HOME'] = '/Developer/NVIDIA/CUDA-9.0'
    os.environ['PATH'] += '/Developer/NVIDIA/CUDA-9.0/bin'
    lib_path = "/usr/local/cuda/lib"
    inc_path = "/usr/local/cuda/include"
    os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cuda0,floatX=float32,dnn.include_path="+inc_path+",dnn.library_path="+lib_path+",gcc.cxxflags=\"-I/usr/local/include -L/usr/local/lib\""
    os.environ["DYLD_LIBRARY_PATH"] = lib_path
    os.environ["LD_LIBRARY_PATH"] = lib_path

from gym_nethack.nhdata import *
from gym_nethack.policies import *
from gym_nethack.configs import *

def get_env(proc_id, config_id):
    """ Creates a Gym environment with name configs[config_id][0]['env_name'] (specified in config file), calls the set_config() method on it, and returns the environment.
    """    
    args = configs[config_id][0].copy()
    args.update(configs[config_id][1])

    ENV_NAME = args['env_name']
    env = gym.make(ENV_NAME)
    env.set_config(proc_id, **args)
    return env

def get_model(env, config):
    """ Creates the neural network whose type is specified by config['nnet_type'] or config['units_d1']/config['units_d2'].
    I know, this method is not very good!
    """
    from keras.models import Sequential
    from keras.layers import Dense, Activation, Flatten, Conv2D, Lambda, Reshape
    
    model = Sequential()
    if 'nnet_type' in config and config['nnet_type'] == 'conv':
        # Create a convolutional net with input of size (ROWNO, COLNO, 3), with two Conv2D and one Dense layer.
        layers = [
            Lambda(lambda a: a / 255.0, input_shape=(1,) + (ROWNO, COLNO, 3), output_shape=(ROWNO, COLNO, 3)),
            Reshape(target_shape=(ROWNO, COLNO, 3)),
            Conv2D(filters=16, kernel_size=(8, 8), strides=4, input_shape=(ROWNO, COLNO, 3), activation='relu'),
            Conv2D(filters=32, kernel_size=(4, 4), strides=2, activation='relu'),
            Flatten(),
            Dense(256, activation='relu')
         ]        
        #for layer in layers:
        #    print(layer)
        #    model.add(layer)
    else:
        # Fully-connected neural network with one or two layers, of sizes given by config['units_d1']/config['units_d2'].
        model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
        if 'units_d1' not in config:
            print("Warning: no hidden layers. Using layer of size 1...")
            config['units_d1'] = 1
        model.add(Dense(config['units_d1'])) # 32
        model.add(Activation('relu'))
        if 'units_d2' in config:
            model.add(Dense(config['units_d2'])) # 32
            model.add(Activation('relu'))
    
    # Always have a dense layer at the end for the output, size specified by the environment's action_space variable.
    print("Adding output layer.")
    model.add(Dense(env.action_space.n, activation='linear'))
    print(model.summary())
    return model

def get_agent(model, env, config, policy_config=None):
    """ Configure and compile the agent.
    
    Args:
        config['learning_agent']: If False, then use a TestAgent with heuristic (not learned) policy given by config['test_policy']. Else, use deep Q-learning agent.
        config['policy']: Keras annealing policy class, e.g., LinearAnnealedPolicy.
        config['test_policy']: The test policy class object to use (should already be instantiated).
        config['optimizer']: instantiated Keras-RL optimizer object
        config['agent']: Keras-RL learning agent class
        config['agent_params']: parameters for constructor of above agent class
    """
    
    policy = None
    test_policy = config['test_policy'] if 'test_policy' in config else None
    
    if 'learning_agent' in config and not config['learning_agent']:
        from gym_nethack.agents import TestAgent
        
        inst = type(test_policy) == type
        if inst:
            # must be instantiated (e.g., map exploration policies)
            test_policy = test_policy()
            test_policy.env = env
        
        agent = TestAgent(test_policy=test_policy)
        
        if inst:
            test_policy.agent = agent
            test_policy.set_config(**policy_config)
    else:
        config['agent_params']['memory'] = config['agent_params']['memory'](limit=env.memory_size if env.from_file else env.max_num_actions, window_length=1)        
        policy = config['policy'](nb_steps=env.max_num_actions_to_anneal_eps if env.from_file else env.max_num_actions, **policy_config)
        agent = config['agent'](model=model, nb_actions=env.action_space.n, policy=policy, test_policy=test_policy, **config['agent_params'])
        agent.compile(config['optimizer'], metrics=['mae'])
    
    return agent

if __name__ == '__main__':
    """
    Ways to call the script:
    
    python3 ngym.py CONFIGNUM
    python3 ngym.py PROCID CONFIGNUM
    python3 ngym.py PROCID CONFIGNUM NUMPROCS
    
    CONFIGNUM specifies the index into the configs.py config list. (e.g., 0, 1, 2, 3...)
    
    PROCID specifies the process number.
    If none is specified, it is set to the same as CONFIGNUM (in that case, you would give the nhdaemon the confignum as procid argument).
    
    NUMPROCS specifies the total number of processes. It is set to 1 by default.
    If set to 1, then after each nethack game ends and the process exits, we can clean up for better stability by killall'ing nethack and removing lock files (see base.py::NetHackEnv::reset() method).
    """
    proc_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    config_id = int(sys.argv[2]) if len(sys.argv) > 2 else proc_id 
    num_procs = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    print("Proc id:", proc_id, ", config id:", config_id)
    
    config = configs[config_id]
    
    config[0]['num_procs'] = num_procs
    if len(config) >= 3 and ('learning_agent' in config[1] and not config[1]['learning_agent']):
        configs[2]['proc_id'] = proc_id
        configs[2]['num_procs'] = num_procs
    
    env = get_env(proc_id, config_id)
    
    if not os.path.exists(env.savedir):
        os.makedirs(env.savedir)
    
    learning = False if 'learning_agent' in config[1] and not config[1]['learning_agent'] else True
    model = get_model(env, config[1]) if learning else None
    dqn = get_agent(model, env, config[1], config[2] if len(config) >= 3 else None)
    
    if 'skip_training' not in config[1] or not config[1]['skip_training']:
        filename = env.savedir + '/duel_dqn_{}_weights.h5f'.format(config[0]['env_name'])
        loaded = False
        
        # Check for existence of weight file. If it exists, ask if we want to train or not. 
        do_fit = True
        if os.path.exists(filename): # Load weights if they exist on disk.
            print("Loading weights...")
            dqn.load_weights(filename)
            loaded = True
            print(dqn.step)
            #input("Loaded existing weights for testing - press [enter]...")
            ans = input("Loaded. Fit? (Y/N) >")
            if 'N' in ans: do_fit = False
            #do_fit = False
        
        if do_fit:
            from libs.rl.callbacks import FileLogger
            dqn.fit(env, nb_steps=env.max_num_actions, nb_episodes=env.max_num_episodes,
                    callbacks=[FileLogger(env.savedir + '/duel_dqn_{}_log.json'.format(config[0]['env_name']),
                    interval=max(env.max_num_actions/10, 100001))], verbose=2)
            dqn.save_weights(filename, overwrite=True) #, memory=False)
        
        env.set_test()
        env.load_records()
    
    # Determine the number of episodes for test duration.
    nb_episodes = env.max_num_episodes
    if 'num_episodes' in configs[config_id][0]:
        nb_episodes = configs[config_id][0]['num_episodes']
    elif 'num_test_episodes' in configs[config_id][0]:
        nb_episodes = configs[config_id][0]['num_test_episodes']
    elif 'NetHackCombat' in configs[config_id][0]['env_name']:
        if 'monsters' in configs[config_id][0]: # test on 400 per each monster.
            nb_episodes = (400*len(env.monsters))-len(list(env.records.keys())[0])
        else: # test on the combat records
            nb_episodes = len(list(env.records.keys())[0])
    
    print("Testing...")
    dqn.test(env, nb_episodes=nb_episodes, visualize=False, verbose=1 if learning else 0)
