import numpy as np
import math
import logging
logger = logging.getLogger(__name__)

class DoneHandler(object):
    """Composes the done function from a given done configuration."""
    def __init__(self, cfg):
        self.dones = {}
        for key in cfg:
            try:
                exec("self.dones['{}'] = {}()".format(key, key))
            except:
                raise ValueError("Unknown done class {}".format(key))

        if not self.dones:
            logging.warning("No done function specified. Setting TorcsDone "
                            "as done.")
            self.dones['TorcsDone'] = TorcsDone()

    def get_done_signal(self, game_config, game_state):
        done_signals = []
        for done_function in self.dones.values():
            done_signals.append(done_function.check_done(game_config, game_state))
        return np.any(done_signals)

    def reset(self):
        for done_function in self.dones.values():
            done_function.reset()


class MadrasDone(object):
    """Base class of MADRaS done function classes.
    Any new done class must inherit this class and implement
    the following methods:
        - [required] check_done(game_config, game_state)
        - [optional] reset()
    """
    def __init__(self):
        pass

    def check_done(self, game_config, game_state):
        del game_config, game_state
        raise NotImplementedError("Successor class must implement this method.")

    def reset(self):
        pass


class TorcsDone(MadrasDone):
    """Vanilla done function provided by TORCS."""
    def check_done(self, game_config, game_state):
        del game_config
        if not math.isnan(game_state["torcs_done"]):
            return game_state["torcs_done"]
        else:
            return True


class RaceOver(MadrasDone):
    """Terminates episode when the agent has finishes one lap."""
    def __init__(self):
        self.num_steps = 0

    def check_done(self, game_config, game_state):
        self.num_steps += 1
        if game_state["distance_traversed"] >= game_config.track_len:
            logging.info("Done: Race over in {} steps!".format(self.num_steps))
            return True
        else:
            return False
      
    def reset(self):
        self.num_steps = 0


class TimeOut(MadrasDone):
    def __init__(self):
        self.num_steps = 0

    def check_done(self, game_config, game_state):
        del game_state
        self.num_steps += 1
        if not game_config.max_steps:
            max_steps = int(game_config.track_len / game_config.target_speed * 50)
        else:
            max_steps = game_config.max_steps
        if self.num_steps >= max_steps:
            logging.info("Done: Episode terminated due to timeout.")
            self.num_steps = 0
            return True
        else:
            return False

    def reset(self):
        self.num_steps = 0


class Collision(MadrasDone):
    def __init__(self):
        self.damage = 0.0
        self.num_steps = 0

    def check_done(self, game_config, game_state):
        del game_config
        self.num_steps += 1
        if self.num_steps == 1:
            self.damage = game_state["damage"]
            return False

        if self.damage < game_state["damage"]:
            logging.info("Done: Episode terminated because agent collided after {} steps.".format(self.num_steps))
            self.damage = 0.0
            return True
        else:
            return False

    def reset(self):
        self.damage = 0.0
        self.num_steps = 0


class TurnBackward(MadrasDone):
    def __init__(self):
        self.num_steps = 0

    def check_done(self, game_config, game_state):
        del game_config
        self.num_steps += 1
        if np.cos(game_state["angle"]) < 0:
            logging.info("Done: Episode terminated because agent turned backward after {} steps.".format(self.num_steps))
            return True
        else:
            return False

    def reset(self):
        self.num_steps = 0


class OutOfTrack(MadrasDone):
    def __init__(self):
        self.num_steps = 0

    def check_done(self, game_config, game_state):
        self.track_limits = game_config.track_limits
        self.num_steps += 1
        if (game_state["trackPos"] < self.track_limits['low'] or 
            game_state["trackPos"] > self.track_limits['high'] or 
            np.any(np.asarray(game_state["track"]) < 0)):
            logging.info("Done: Episode terminated because agent went out of track after {} steps.".format(self.num_steps))
            self.num_steps = 0
            return True
        else:
            return False

    def reset(self):
        self.num_steps = 0

class Rank1(MadrasDone):
    def __init__(self):
        self.num_steps = 0

    def check_done(self, game_config, game_state):
        self.num_steps += 1
        if game_state["racePos"] == 1:
            logging.info("Done: Episode terminated because agent"
                         " is Rank 1 after {} steps.".format(self.num_steps))
            self.num_steps = 0
            return True
        else:
            return False

    def reset(self):
        self.num_steps = 0