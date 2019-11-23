import numpy as np
import math
import warnings


class DoneManager(object):
    """Composes the done function from a given done configuration."""
    def __init__(self, cfg):
        self.dones = {}
        for key in cfg:
            try:
                exec("self.dones[{}] = {}()".format(key, key))
            except:
                raise ValueError("Unknown done class {}".format(key))

        if not self.dones:
            warnings.warn("No done function specified. Setting TorcsDone "
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
    def check_done(self, game_config, game_state):
        if game_state["distance_traversed"] >= game_config.track_len:
            print("Done: Race over!")
            return True
        else:
            return False


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
            print("Done: Episode terminated due to timeout.")
            return True
        else:
            return False


class Collision(MadrasDone):
    def __init__(self):
        self.damage = 0.0

    def check_done(self, game_config, game_state):
        del game_config
        if self.damage < game_state["damage"]:
            print("Done: Episode terminated because agent collided.")
            return True
        else:
            return False


class TurnBackward(MadrasDone):
    def check_done(self, game_config, game_state):
        del game_config
        if np.cos(game_state["angle"]) < 0:
            print("Done: Episode terminated because agent turned backward.")
            return True
        else:
            return False