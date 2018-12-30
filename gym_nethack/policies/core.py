import os

from gym_nethack.misc import verboseprint
from gym_nethack.fileio import get_dir_for_params

class Policy(object):
    """Standard policy class taken from Keras-RL with a few extensions."""
    name = 'unnamed'
    
    def __init__(self, name='obsolete'):
        self.name = name
    
    def _set_agent(self, agent):
        self.agent = agent
    
    @property
    def metrics_names(self):
        return []

    @property
    def metrics(self):
        return []

    def select_action(self, **kwargs):
        raise NotImplementedError

    def get_config(self):
        return {}
    
    def set_config(self):
        pass

class ParameterizedPolicy(Policy):
    """Extension of policy class that allows for grid-search on specified parameters."""
    def __init__(self):
        """Initialize a policy that has parameters which can be modified (e.g., through grid search)."""
        self.cur_combo = -1
        self.num_games_this_combo = 0
    
    def set_config(self, grid_search=False, top_models=False, num_episodes_per_combo=200, proc_id=0, num_procs=1, param_combos=None, param_abbrvs=None):
        """Set config.
        
        Args:
            grid_search: whether to change parameters every certain number of episodes.
            top_models: whether to load from a text file and use the specified param combos inside. (Must have grid_search=True)
            num_episodes_per_combo: if grid search, number of episodes per each combination of alg. parameters.
            proc_id: if grid search, process ID of this environment, to be matched with the argument passed to the daemon launching script.
            num_procs: if grid search, number of processes that will be running in parallel
            param_combos: list of lists of parameter combinations
            param_abbrvs: abbreviated parameter names (for directory name)
        """
        self.grid_search = grid_search
        assert not top_models or grid_search
        
        if grid_search:
            self.num_episodes_per_combo = num_episodes_per_combo
            self.param_abbrvs = param_abbrvs
            
            if os.path.isfile('combos_to_try.txt'):
                param_combos = read_list('combos_to_try')
        
            elif top_models and os.path.isfile(self.base_dir + 'top_models.txt'):
                param_combos = [get_params_for_dir(dirname) for dirname in read_list(self.env.basedir + 'top_models')]
            
            else:
                num_combos_per_proc = max(len(param_combos) // num_procs, 1)
                param_combos_per_proc = [param_combos[i:i + num_combos_per_proc] for i in range(0, len(param_combos), num_combos_per_proc)]
                self.combos_to_set = param_combos_per_proc[proc_id if num_procs > 1 else 0]
                verboseprint(param_combos_per_proc)
        else:
            self.set_params(self.get_default_params())
    
    def reset(self):
        """Called on starting a new episode."""
        self.switch_encounter()
    
    def switch_encounter(self):
        """Alter alg. parameters if using grid search."""
        if not self.grid_search:
            return
        elif self.combos_to_set is not None: # initial setup
            self.set_combos(self.combos_to_set)
            self.combos_to_set = None
        elif self.cur_combo > -1 and self.num_games_this_combo < self.num_episodes_per_combo:
            verboseprint("Not finished with current param combo yet")
            return
        while True:
            if self.num_games_this_combo > 0:
                self.env.save_records() # save current records to the current directroy
            
            self.cur_combo += 1
            if self.cur_combo >= len(self.param_combos):
                verboseprint("Past max combo, going back to combo 0")
                self.cur_combo = 0
            
            print("Combo", self.cur_combo, "/", len(self.param_combos))
            
            cur_params = self.param_combos[self.cur_combo]
            self.set_params(cur_params)
            self.env.savedir = self.env.basedir + get_dir_for_params(cur_params, self.param_abbrvs)
            self.env.load_records() # load any existing records from the new directory
            self.num_games_this_combo = 0
            self.env.total_num_games += len(self.env.records['expl'])
            break
        
        verboseprint("Cur params:", self.env.savedir)
        
    def end_episode(self):
        """Record new episode ended."""
        self.num_games_this_combo += 1
    
    def set_combos(self, combos):
        """Update list of parameter combinations to try.
        
        Args:
            combos: list of combinations to use"""
        self.param_combos = combos
        self.env.max_num_episodes = self.num_episodes_per_combo * len(combos)
    
    def get_default_params(self):
        """Get the default parameters for the policy."""
        return []
    
    def set_params(self, params):
        """Set the current parameters for the policy.
        
        Args:
            params: policy parameters"""
        pass
    