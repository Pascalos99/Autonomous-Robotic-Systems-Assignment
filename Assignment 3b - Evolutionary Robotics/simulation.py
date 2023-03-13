import configparser
import pathlib
import pickle
import time

import numpy as np
import pygame

from dust_map import DustMap
from genetic_algorithm import Fitness
from map import Map
from neural_GA import ANN, sigmoid, tanh, get_ANN_GA
from ann_visualizer import ANN_Visualizer
from player import Player

pygame.init()
pygame.font.init()

working_directory = pathlib.Path(__file__).parent.absolute()
config = configparser.ConfigParser()
config.read(f"{working_directory}/config.ini")

WIDTH, HEIGHT = float(config['PROGRAM']['window_width']), float(config['PROGRAM']['window_height'])
FPS = float(config['PROGRAM']['fps'])
FONT = pygame.font.SysFont('Consolas', 14)


class Simulation:
    def __init__(self, ann: ANN = None, ann_bridge: callable = None, iterations: int = None,
                 visualize_game: bool = True, map_file: str = None):
        self.ann = ann
        self.ann_bridge_func = ann_bridge
        self.iterations = iterations
        self.iter_counter = 0
        self.visualize_game = visualize_game
        self.map_file = map_file
        self.map = map
        self.manual_mode = 0
        self.win = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.map = Map()

        if self.ann is None:
            self.manual_mode = 1

        if self.map_file is None:
            self.map_file = str(config['ANN']['map_file'])

        self.map.load_map_from_json(f"{working_directory}/maps/{self.map_file}")

        # self.min_x, self.min_y, self.max_x, self.max_y
        bounds = (float(config['DUST']['min_x']), float(config['DUST']['min_y']), float(config['DUST']['max_x']),
                  float(config['DUST']['max_y']))
        self.player = Player(self.map,
                             DustMap(bounds, float(config['DUST']['regen_rate']), float(config['DUST']['density']),
                                     str(config['DUST']['random_state'])))

        self.sensitivity = float(config['ANN']['speed_step'])
        self.do_draw_dust = config.getboolean('DUST', 'draw')

        self.ANN_Viz = None

        # In the README.md is explained how the bot is controlled with a keyboard.
        self.key_config = {pygame.K_q: lambda: self.player.change_vel(0, self.sensitivity),
                           pygame.K_a: lambda: self.player.change_vel(0, -self.sensitivity),
                           pygame.K_w: lambda: self.player.change_vel(self.sensitivity, self.sensitivity),
                           pygame.K_s: lambda: self.player.change_vel(-self.sensitivity, -self.sensitivity),
                           pygame.K_e: lambda: self.player.change_vel(self.sensitivity, 0),
                           pygame.K_d: lambda: self.player.change_vel(-self.sensitivity, 0),
                           pygame.K_x: lambda: self.player.reset_vel(), }

    def run(self):
        is_running = True
        while is_running:
            if self.iterations is not None and self.iter_counter >= self.iterations:
                is_running = False

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_running = False

                if event.type == pygame.KEYDOWN and self.manual_mode:
                    try:
                        self.key_config[event.key]()
                    except KeyError:
                        pass

                if event.type == pygame.VIDEORESIZE:
                    self.win = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

            self.player.step()
            self.player.calculate_sensor_distance()

            if not self.manual_mode:
                self.ann_bridge()

            if self.visualize_game:
                self.draw()

            if self.visualize_game:
                self.clock.tick(FPS)
            self.iter_counter += 1

    def ann_bridge(self):
        distances = np.array([dict_list[1] for dict_list in self.player.sensor_lines.values()],
                             dtype=np.float64) / float(config['BOT']['vision_range'])
        velocities = np.array(self.player.vel, dtype=np.float64) / float(config['ANN']['max_speed'])

        if self.ann_bridge_func is None:
            self.ann_bridge_func = lambda ann, dist, vel: ann.forward(np.concatenate([dist, vel]))

        new_velocities = np.array(self.ann_bridge_func(self.ann, distances, velocities))
        self.player.vel = np.round(np.clip(new_velocities, -1, 1) * float(config['ANN']['max_speed']))

    def draw(self):
        self.clear()
        self.draw_fps()
        self.draw_map()
        self.draw_player()
        if self.do_draw_dust: self.draw_dust()
        self.draw_ann()
        pygame.display.flip()

    def draw_fps(self):
        fps_text = FONT.render(f'FPS: {int(self.clock.get_fps())}', False, '#00dd00')
        self.win.blit(fps_text, dest=[5, 5])

    def draw_map(self):
        for line in self.map.lines:
            pygame.draw.line(self.win, '#aaaaaa', line[0], line[1], width=self.map.line_width)

    def draw_dust(self):
        # Draw Dust
        if self.player.dust is not None:
            self.player.dust.draw_dust(self.win)

    def draw_player(self):
        # Draw Main Circle
        pygame.draw.circle(self.win, '#00aacc', self.player.pos, self.player.radius)

        # Draw Direction Line
        pygame.draw.line(self.win, '#000000', self.player.pos, (
            self.player.pos[0] + self.player.radius * np.sin(self.player.direction + 0.5 * np.pi),
            self.player.pos[1] - self.player.radius * np.cos(self.player.direction + 0.5 * np.pi),), width=2)

        # Draw all sensor lines
        angle = 2 * np.pi / self.player.num_sensors
        for i in range(self.player.num_sensors):
            start_x, start_y = self.player.sensor_lines[i][0][0]
            end_x, end_y = self.player.sensor_lines[i][0][1]
            pygame.draw.line(self.win, '#dd0000', start_pos=[start_x, start_y, ], end_pos=[end_x, end_y, ], width=1)

        # Show motor numbers
        x_text = FONT.render(f'l:{round(self.player.vel[1] / self.sensitivity, 1)}', False, '#dddddd')
        y_text = FONT.render(f'r:{round(self.player.vel[0] / self.sensitivity, 1)}', False, '#dddddd')
        self.win.blit(x_text, dest=[
            self.player.pos[0] - x_text.get_width() // 2 + (self.player.radius // 2) * np.sin(self.player.direction),
            self.player.pos[1] - x_text.get_height() // 2 - (self.player.radius // 2) * np.cos(
                self.player.direction), ])
        self.win.blit(y_text, dest=[(self.player.pos[0] - y_text.get_width() // 2 + (self.player.radius // 2) * np.sin(
            np.pi + self.player.direction)), (
                                            self.player.pos[1] - y_text.get_height() // 2 - (
                                            self.player.radius // 2) * np.cos(
                                        np.pi + self.player.direction)), ])

        # Show Distance Numbers
        for i in range(self.player.num_sensors):
            distance = self.player.sensor_lines[i][1]

            if not config.getboolean('PROGRAM', 'sensor_data_separate'):
                text = FONT.render(str(int(round(distance, 0))), False, '#dddddd')
            else:
                text = FONT.render(str(i), False, '#dddddd')

            self.win.blit(text, dest=[(self.player.pos[0] - text.get_width() // 2 + (self.player.radius + 20) * np.cos(
                i * angle - self.player.direction)), (
                                              self.player.pos[1] - text.get_height() // 2 - (
                                              self.player.radius + 20) * np.sin(
                                          i * angle - self.player.direction)), ])

        if config.getboolean('PROGRAM', 'sensor_data_separate'):
            for i in range(self.player.num_sensors):
                distance = self.player.sensor_lines[i][1]
                text = FONT.render(f"Sensor {i}: {int(round(distance, 0))}", False, '#dddddd')
                self.win.blit(text, dest=[self.win.get_width() - 150, 50 + i * 15])

    def draw_ann(self):
        if not self.ann: return 
        if not self.ann.network: return 
        if not self.ann.activations: return

        if not self.ANN_Viz:
            self.ANN_Viz = ANN_Visualizer(self.ann, self.win)

        img = self.ANN_Viz.animate(self.ann)
        self.win.blit(
            source=img,
            dest=(self.win.get_rect().width - img.get_rect().width, 0)
        )

    def clear(self):
        self.win.fill('#232323')


def basic_fitness(individual):
    viz = False
    sim = Simulation(ann=individual['ann'], iterations=100, visualize_game=viz)
    plr = sim.player
    sim.run()
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
    hidden_layers = [3]
    activation_functions = [sigmoid, tanh]

    GA = get_ANN_GA(Fitness(basic_fitness), num_inputs, num_outputs, hidden_layers, activation_functions, popsize=1)
    GA.iterate(15)
    sim = Simulation(ann=GA.population['ann'][0])

    # sim = Simulation()
    sim.run()
    pygame.quit()

    save_ann(GA.population['ann'][0])
