import numpy as np
import random
import os
import logging
logger = logging.getLogger(__name__)

path_and_file = os.path.realpath(__file__)
path, _ = os.path.split(path_and_file)

QUICKRACE_TEMPLATE_PATH = os.path.join(path, "data", "quickrace.template")
CAR_CONFIG_TEMPATE_PATH = os.path.join(path, "data", "car_config.template")
SCR_SERVER_CONFIG_TEMPLATE_PATH = os.path.join(path, "data", "scr_server_config.template")

MAX_NUM_CARS = 10  # There are 10 scr-servers in the TORCS GUI
TRACK_NAMES = [
    "aalborg",
    "alpine-1",
    "alpine-2",
    "brondehach",
    "g-track-1",
    "g-track-2",
    "g-track-3",
    "corkscrew",
    "eroad",
    "e-track-1",
    "e-track-2",
    "e-track-3",
    "e-track-4",
    "e-track-6",
    "forza",
    "ole-road-1",
    "ruudskogen",
    "spring",
    "street-1",
    "wheel-1",
    "wheel-2"
]


class TorcsConfig(object):
    def __init__(self, cfg, randomize=False):
        self.max_cars = cfg["max_cars"] if "max_cars" in cfg else MAX_NUM_CARS
        self.track_category = "road"  # for test purposes - will add "dirt" later
        self.track_names = cfg["track_names"] if  "track_names" in cfg else TRACK_NAMES
        self.distance_to_start = cfg["distance_to_start"] if "distance_to_start" in cfg else 0
        self.torcs_server_config_dir = (cfg["torcs_server_config_dir"] if "torcs_server_config_dir" in cfg
                                        else "/home/anirban/.torcs/config/raceman/")
        self.scr_server_config_dir = (cfg["scr_server_config_dir"] if "scr_server_config_dir" in cfg
                                          else "/home/anirban/usr/local/share/games/torcs/drivers/scr_server/")
        with open(QUICKRACE_TEMPLATE_PATH, 'r') as f:
            self.quickrace_template = f.read()
        with open(CAR_CONFIG_TEMPATE_PATH, 'r') as f:
            self.car_config_template = f.read()
        with open(SCR_SERVER_CONFIG_TEMPLATE_PATH, 'r') as f:
            self.scr_server_config_template = f.read()
        self.randomize = randomize)
        self.quickrace_xml_path = os.path.join(self.torcs_server_config_dir, "quickrace.xml")
        self.scr_server_xml_path = os.path.join(self.scr_server_config_dir, "scr_server.xml")
        self.traffic_car_type = cfg['traffic_car']
        self.learning_car_type = cfg['learning_car']

    def get_num_traffic_cars(self):
        if not self.randomize:
            return self.max_cars-1
        else:
            num_traffic_cars = np.random.randint(low=0, high=self.max_cars)
            return num_traffic_cars

    def get_track_name(self):
        if not self.randomize:
            track_name = self.track_names[0]
        else:
            track_name = random.sample(self.track_names, 1)[0]
        logging.info("-------------------------CURRENT TRACK:{}------------------------".format(track_name))
        return track_name

    def generate_torcs_server_config(self):
        self.num_traffic_cars = self.get_num_traffic_cars()
        self.num_learning_cars = 1

        car_config = "\n".join(self.car_config_template.format(
                      **{"section_no": i+1, "car_no": i}) for i in range(
                        self.num_traffic_cars + self.num_learning_cars))
        context = {
            "track_name": self.get_track_name(),
            "track_category": self.track_category,
            "distance_to_start": self.distance_to_start,
            "car_config": car_config
        }
        torcs_server_config = self.quickrace_template.format(**context)
        with open(self.quickrace_xml_path, "w") as f:
            f.write(torcs_server_config)

        car_name_list = ([self.traffic_car_type]*self.num_traffic_cars + [self.learning_car_type] +
                         [self.traffic_car_type]*(MAX_NUM_CARS-self.num_traffic_cars-1))
        scr_server_config = self.scr_server_config_template.format(*car_name_list)
        with open(self.scr_server_xml_path, "w") as f:
            f.write(scr_server_config)