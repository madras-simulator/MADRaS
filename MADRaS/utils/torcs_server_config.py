import numpy as np
import random
import os
import logging
logger = logging.getLogger(__name__)

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

QUICKRACE_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE params SYSTEM "params.dtd">


<params name="Quick Race">
  <section name="Header">
    <attstr name="name" val="Quick Race"/>
    <attstr name="description" val="Quick Race"/>
    <attnum name="priority" val="10"/>
    <attstr name="menu image" val="data/img/splash-qr.png"/>
  </section>

  <section name="Tracks">
    <attnum name="maximum number" val="1"/>
    <section name="1">
      <attstr name="name" val="{track_name}"/>
      <attstr name="category" val="{track_category}"/>
    </section>

  </section>

  <section name="Races">
    <section name="1">
      <attstr name="name" val="Quick Race"/>
    </section>

  </section>

  <section name="Quick Race">
    <attnum name="distance" unit="km" val="0"/>
    <attstr name="type" val="race"/>
    <attstr name="starting order" val="drivers list"/>
    <attstr name="restart" val="yes"/>
    <attnum name="laps" val="1"/>
    <section name="Starting Grid">
      <attnum name="rows" val="2"/>
      <attnum name="distance to start" val="{distance_to_start}"/>
      <attnum name="distance between columns" val="20"/>
      <attnum name="offset within a column" val="10"/>
      <attnum name="initial speed" val="0"/>
      <attnum name="initial height" val="0.2"/>
    </section>

  </section>

  <section name="Drivers">
    <attnum name="maximum number" val="40"/>
    <attnum name="focused idx" val="0"/>
    <attstr name="focused module" val="scr_server"/>
    {car_config}

  </section>

  <section name="Configuration">
    <attnum name="current configuration" val="4"/>
    <section name="1">
      <attstr name="type" val="track select"/>
    </section>

    <section name="2">
      <attstr name="type" val="drivers select"/>
    </section>

    <section name="3">
      <attstr name="type" val="race config"/>
      <attstr name="race" val="Quick Race"/>
      <section name="Options">
        <section name="1">
          <attstr name="type" val="race length"/>
        </section>

      </section>

    </section>

  </section>

</params>
"""

CAR_CONFIG_TEMPATE = """<section name="{section_no}">
      <attnum name="idx" val="{car_no}"/>
      <attstr name="module" val="scr_server"/>
    </section>
"""

class TorcsConfig(object):
    def __init__(self, cfg, randomize=False):
        self.max_cars = cfg["max_cars"] if "max_cars" in cfg else MAX_NUM_CARS
        self.track_category = "road"  # for test purposes - will add "dirt" later
        self.track_names = cfg["track_names"] if  "track_names" in cfg else TRACK_NAMES
        self.distance_to_start = cfg["distance_to_start"] if "distance_to_start" in cfg else 0
        self.torcs_server_config_path = cfg["torcs_server_config_path"] if "torcs_server_config_path" in cfg else "/tmp"
        self.template = QUICKRACE_TEMPLATE
        self.car_config_template = CAR_CONFIG_TEMPATE
        self.randomize = randomize
        os.system("mkdir -p {}".format(self.torcs_server_config_path))
        self.quickrace_path = os.path.join(self.torcs_server_config_path, "quickrace.xml")

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

        car_config = "\n    ".join(self.car_config_template.format(
                               **{"section_no": i+1, "car_no": i}) for i in range(
                                  self.num_traffic_cars + self.num_learning_cars))
        context = {
            "track_name": self.get_track_name(),
            "track_category": self.track_category,
            "distance_to_start": self.distance_to_start,
            "car_config": car_config
        }
        torcs_server_config = self.template.format(**context)
        with open(self.quickrace_path, "w") as f:
            f.write(torcs_server_config)