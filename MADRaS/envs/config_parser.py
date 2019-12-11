import numpy as np
import yaml

def parse_yaml(yaml_file):
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)


class MadrasEnvConfig(object):
    """Configuration class for MADRaS Gym environment."""
    def __init__(self):
        self.torcs_server_port = 6006
        self.visualise = False
        self.vision = False
        self.no_of_visualisations = 1
        self.track_len = 7014.6
        self.max_steps = 20000
        self.agents = {}
        self.traffic = []

    def update(self, cfg_dict):
        """Update the configuration terms from a dictionary.
        
        Args:
            cfg_dict: dictionary whose keys are the names of class attributes whose
                      values must be updated
        """
        if cfg_dict is None:
            return
        direct_attributes = ['torcs_server_port', 'visualise', 'vision', 'no_of_visualizations', 'track_len',
                             'max_steps', 'agents', 'traffic']
        for key in direct_attributes:
            if key in cfg_dict:
                exec("self.{} = {}".format(key, cfg_dict[key]))
        self.validate()

    def validate(self):
        pass


class MadrasAgentConfig(object):
    def __init__(self):
        self.vision = False
        self.throttle = True
        self.gear_change = False
        self.pid_assist = False
        self.pid_settings = {}
        self.client_max_steps = np.inf
        self.target_speed = 15.0
        self.normalize_actions = False
        self.early_stop = True
        self.observations = None
        self.rewards = {}
        self.dones = {}

    def update(self, cfg_dict):
        """Update the configuration terms from a dictionary.
        
        Args:
            cfg_dict: dictionary whose keys are the names of class attributes whose
                      values must be updated
        """
        if cfg_dict is None:
            return
        direct_attributes = ['track_len', 'max_steps', 'vision', 'throttle', 'gear_change', 'pid_assist',
                             'pid_latency', 'target_speed', 'early_stop', 'normalize_actions', 'observations', 'rewards', 'dones',
                             'pid_settings']
        for key in direct_attributes:
            if key in cfg_dict:
                exec("self.{} = {}".format(key, cfg_dict[key]))
        self.client_max_steps = (np.inf if cfg_dict['client_max_steps'] == -1
                                 else cfg_dict['client_max_steps'])
        self.validate()

    def validate(self):
        assert self.vision == False, "Vision input is not yet supported."
        assert self.throttle == True, "Throttle must be True."
        assert self.gear_change == False, "Only automatic transmission is currently supported."
        # TODO(santara): add checks for self.state_dim
