import configparser
import pathlib
import pickle
import time

import numpy as np

from dust_map import DustMap
from genetic_algorithm import Fitness
from map import Map
from neural_GA import ANN, sigmoid, tanh, get_ANN_GA
from player import Player

working_directory = pathlib.Path(__file__).parent.absolute()
config = configparser.ConfigParser()
config.read(f"{working_directory}/config.ini")


class MiniSim:
    def __init__(self, ann: ANN, ann_bridge: callable, map_to_load: str):
        self.ann = ann
        self.ann_bridge_func = ann_bridge
        self.map = Map()
        self.map.load_map_from_json(f"{working_directory}/maps/{map_to_load}")
        bounds = (float(config['DUST']['min_x']), float(config['DUST']['min_y']), float(config['DUST']['max_x']),
                  float(config['DUST']['max_y']))
        self.player = Player(self.map,
                             DustMap(bounds, float(config['DUST']['regen_rate']), float(config['DUST']['density']),
                                     str(config['DUST']['random_state'])))

    def run(self, iterations):
        for iter in range(1, iterations + 1):
            self.player.step()
            self.player.calculate_sensor_distance()
            distances = np.array([dict_list[1] for dict_list in self.player.sensor_lines.values()],
                                 dtype=np.float64) / float(config['BOT']['vision_range'])
            velocities = np.array(self.player.vel, dtype=np.float64) / float(config['ANN']['max_speed'])
            new_velocities = np.array(self.ann_bridge_func(self.ann, distances, velocities))
            self.player.vel = np.clip(new_velocities, -1, 1) * float(config['ANN']['max_speed'])


def basic_fitness(individual):
    sim = MiniSim(individual['ann'], lambda ann, dist, vel: ann.forward(np.concatenate([dist, vel])), "map_2.json")
    plr = sim.player
    sim.run(100)
    maxvel = 0
    if len(plr.collision_velocities) > 0: maxvel = np.max(plr.collision_velocities)
    print("finished sim with score", plr.points * 0.25, "-", len(plr.collision_velocities), "-", round(maxvel ** 2, 2),
          "=",
          plr.points * 0.25 - len(plr.collision_velocities) - maxvel ** 2)
    return plr.points * 0.25 - len(plr.collision_velocities) - maxvel ** 2


def save_ann(ann: ANN):
    # Here UTC+0 time in ISO 8601 format is appended to make the filenames unique
    filename = f"ann_object_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.pkl"
    directory = f"{working_directory}/anns/{filename}"
    with open(directory, 'wb') as f:
        pickle.dump(ann, f)


def load_ann(ann_file: str):
    with open(ann_file, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    num_inputs, num_outputs = int(config['BOT']['num_sensors']) + 2, 2
    hidden_layers = [4]
    activation_functions = [sigmoid, tanh]
    GA = get_ANN_GA(Fitness(basic_fitness), num_inputs, num_outputs, hidden_layers, activation_functions, popsize=25)
    GA.iterate(2)
    import pygame
    from simulation import Simulation

    sim = Simulation(ann=GA.population['ann'][0])
    sim.run()
    pygame.quit()

    save_ann(GA.population['ann'][0])
