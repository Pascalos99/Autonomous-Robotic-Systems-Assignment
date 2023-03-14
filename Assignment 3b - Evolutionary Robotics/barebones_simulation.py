import configparser
import pathlib
import pickle
import time

import numpy as np
from matplotlib import pyplot as plt

from dust_map import DustMap
from genetic_algorithm import Fitness
from map import Map
from neural_GA import ANN, sigmoid, tanh, get_ANN_GA
from player import Player
from simulation import default_ann_bridge, get_recurrent_ann_bridge, default_fitness_func

working_directory = pathlib.Path(__file__).parent.absolute()
config = configparser.ConfigParser()
config.read(f"{working_directory}/config.ini")
training_maps = [f"train/map_{i}.json" for i in range(1,19)]
testing_maps = [f"map_{i}.json" for i in range(19,26)]

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


def get_average_fitness(fitness_func=None, ann_bridge=None, iterations_per_map=100, maps_to_load=None):
    if fitness_func is None:
        fitness_func = default_fitness_func
    if ann_bridge is None:
        ann_bridge = default_ann_bridge
    if maps_to_load is None:
        maps_to_load = training_maps

    def average_fitness(individual):
        fitness = 0
        for i in range(len(maps_to_load)):
            sim = MiniSim(individual['ann'], ann_bridge, maps_to_load[i])
            plr = sim.player
            sim.run(iterations_per_map)
            fitness += fitness_func(plr)
        print(f'   Evaluated at: {round(fitness / float(len(maps_to_load)),2)}')
        return fitness / float(len(maps_to_load))
    return average_fitness


def save_ann(ann: ANN):
    # Here UTC+0 time in ISO 8601 format is appended to make the filenames unique
    filename = f"ann_object_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.pkl"
    directory = f"{working_directory}/anns/{filename}"
    with open(directory, 'wb') as f:
        pickle.dump(ann, f)
    print(f"Saved ANN successfully to {directory}")


def load_ann(ann_file: str):
    with open(ann_file, 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    latent_size = 4
    num_inputs, num_outputs = int(config['BOT']['num_sensors']) + latent_size, 2
    hidden_layers = [latent_size]
    activation_functions = [sigmoid, tanh]
    ann_bridge = get_recurrent_ann_bridge(latent_size)
    fitness = Fitness(get_average_fitness(
        fitness_func=default_fitness_func,
        ann_bridge=ann_bridge,
        iterations_per_map=100,
        maps_to_load = training_maps
    ))
    GA = get_ANN_GA(fitness, num_inputs, num_outputs, hidden_layers, activation_functions, popsize=50)
    fitness = GA.iterate(20)
    plt.plot(fitness)
    plt.show()

    import pygame
    from simulation import Simulation
    for map_file in training_maps + testing_maps:
        sim = Simulation(ann=GA.population['ann'][0], ann_bridge=ann_bridge, map_file=map_file, iterations=100)
        sim.run()
    pygame.quit()

    save_ann(GA.population['ann'][0])
