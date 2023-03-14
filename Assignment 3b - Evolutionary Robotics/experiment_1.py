from neural_GA import get_ANN_GA, sigmoid, tanh
from genetic_algorithm import Fitness
from barebones_simulation import MiniSim
from simulation import Simulation, get_recurrent_ann_bridge

import os
import pickle
import numpy as np


def ann_bridge(ANN, distances, velocity):
    return get_recurrent_ann_bridge(4, -2)(ANN, distances, velocity)


def fitness_func(individual, simulation_iterations=100):
    ANN = individual["ann"]
    
    fitness = 0
    train_maps = get_train_maps()
    if len(train_maps) > 0:
        for train_map in train_maps:
            # Run simulation for ANN on test map.
            simulation = MiniSim(ANN, ann_bridge, train_map)
            simulation.run(simulation_iterations)  
            
            player = simulation.player
            
            dust_cleaned = player.points
            collision_penalty = (abs(((len(player.collision_velocities) ** 2) * np.max(player.collision_velocities))) if len(player.collision_velocities) > 0 else 0)
            map_fitness = dust_cleaned - collision_penalty
            
            # print(f"{ dust_cleaned } - { collision_penalty } = { map_fitness }")
            
            fitness += map_fitness
        
        # Calculate average over the test maps.
        fitness = fitness / len(train_maps)
        print("Avg. fitness:", fitness)
    
    return fitness


def get_train_maps():
    return get_maps_from_directory("train")


def get_maps_from_directory(name):
    return [f"{ name }/" + file for file in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"maps/{ name }"))]


latent_size = 4
num_inputs, num_outputs = 12 + latent_size, 2
hidden_layers = [latent_size]

fitness = Fitness(fitness_func)
GA = get_ANN_GA(fitness, num_inputs, num_outputs, hidden_layers, [sigmoid, tanh], 20)

GA.iterate(10, display_iterations=True)
ANN_10 = GA.population["ann"][0]
    
GA.iterate(40, display_iterations=True)
ANN = GA.population["ann"][0]

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "./ANNs/experiment_1A.pkl"), "wb+") as file:
    pickle.dump(ANN_10, file)
    print("Successfully saved ANN after 10 iterations.")

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "./ANNs/experiment_1B.pkl"), "wb+") as file:
    pickle.dump(ANN, file)
    print("Successfully saved best ANN.")

print(GA.population)
input("Press any key to continue to simulation.")

print("After 10 generations")
for train_map in [*get_train_maps(), *get_maps_from_directory("test")]:
    simulation = Simulation(ann=ANN_10, ann_bridge=ann_bridge, iterations=500, map_file=train_map)
    simulation.run()
    
print("After 50 generations")
for train_map in [*get_train_maps(), *get_maps_from_directory("test")]:
    simulation = Simulation(ann=ANN, ann_bridge=ann_bridge, iterations=500, map_file=train_map)
    simulation.run()