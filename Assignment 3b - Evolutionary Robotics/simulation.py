import configparser
import datetime
import json
import pathlib

import numpy as np
import pygame

from player import Player
from dust_map import DustMap
from neural_GA import ANN
from maps import Map

pygame.init()
pygame.font.init()

working_directory = pathlib.Path(__file__).parent.absolute()
config = configparser.ConfigParser()
config.read(f"{working_directory}/config.ini")

WIDTH, HEIGHT = float(config['PROGRAM']['window_width']), float(config['PROGRAM']['window_height'])
FPS = float(config['PROGRAM']['fps'])
FONT = pygame.font.SysFont('Consolas', 14)


class Simulation:
    def __init__(self):
        self.win = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.map = Map()
        self.map.load_map_from_json(f"{working_directory}/maps/{str(config['ANN']['map_file'])}")
        self.player = Player(self.map, DustMap())

        self.sensitivity = float(config['ANN']['speed_step'])

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
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_running = False

                if event.type == pygame.KEYDOWN and bool(int(config['PROGRAM']['manual_mode'])):
                    try:
                        self.key_config[event.key]()
                    except KeyError:
                        pass

                if event.type == pygame.VIDEORESIZE:
                    self.win = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

            self.player.step()
            self.player.calculate_sensor_distance()

            if not bool(int(config['PROGRAM']['manual_mode'])):
                ann = ANN(14, 2, [4])
                self.ann_bridge(ann)

            if bool(int(config['PROGRAM']['visualize_game'])):
                self.draw()
            print(self.player.vel)
            self.clock.tick(FPS)

    def ann_bridge(self, ann: ANN):
        distances = np.array([dict_list[1] for dict_list in self.player.sensor_lines.values()], dtype=np.float64)
        velocities = np.array(self.player.vel, dtype=np.float64)
        print(np.concatenate([distances, velocities]))
        new_velocities = np.array(ann.forward(np.concatenate([distances, velocities])))
        self.player.vel = new_velocities * int(config['ANN']['max_speed'])

    def draw(self):
        self.clear()
        self.draw_fps()
        self.draw_map()
        self.draw_player() # and dust
        pygame.display.flip()

    def draw_fps(self):
        fps_text = FONT.render(f'FPS: {int(self.clock.get_fps())}', False, '#00dd00')
        self.win.blit(fps_text, dest=[5, 5])

    def draw_map(self):
        for line in self.map.lines:
            pygame.draw.line(self.win, '#aaaaaa', line[0], line[1], width=self.map.line_width)

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
            self.player.pos[1] - x_text.get_height() // 2 - (self.player.radius // 2) * np.cos(self.player.direction),
        ])
        self.win.blit(y_text, dest=[
            (self.player.pos[0] - y_text.get_width() // 2 + (self.player.radius // 2)
             * np.sin(np.pi + self.player.direction)),
            (self.player.pos[1] - y_text.get_height() // 2 - (self.player.radius // 2)
             * np.cos(np.pi + self.player.direction)),
        ])

        # Show Distance Numbers
        for i in range(self.player.num_sensors):
            distance = self.player.sensor_lines[i][1]

            if not bool(int(config['PROGRAM']['sensor_data_separate'])):
                text = FONT.render(str(int(round(distance, 0))), False, '#dddddd')
            else:
                text = FONT.render(str(i), False, '#dddddd')

            self.win.blit(text, dest=[
                (self.player.pos[0] - text.get_width() // 2 + (self.player.radius + 20)
                 * np.cos(i * angle - self.player.direction)),
                (self.player.pos[1] - text.get_height() // 2 - (self.player.radius + 20)
                 * np.sin(i * angle - self.player.direction)),
            ])

        if bool(int(config['PROGRAM']['sensor_data_separate'])):
            for i in range(self.player.num_sensors):
                distance = self.player.sensor_lines[i][1]
                text = FONT.render(f"Sensor {i}: {int(round(distance, 0))}", False, '#dddddd')
                self.win.blit(text, dest=[self.win.get_width() - 150, 50 + i * 15])

        # Draw Dust
        if self.player.dust is not None:
            self.player.dust.draw_dust(self.win)

    def clear(self):
        self.win.fill('#232323')


if __name__ == '__main__':
    sim = Simulation()
    sim.run()
    pygame.quit()
