import configparser
import datetime
import json
import pathlib

import numpy as np
import pygame

from player import Player

pygame.init()
pygame.font.init()

working_directory = pathlib.Path(__file__).parent.absolute()
config = configparser.ConfigParser()
config.read(f"{working_directory}/config.ini")

WIDTH, HEIGHT = float(config['PROGRAM']['window_width']), float(config['PROGRAM']['window_height'])
FPS = float(config['PROGRAM']['fps'])
FONT = pygame.font.SysFont('Consolas', 14)


class Map:
    def __init__(self):
        self.lines = []

    def add_line(self, pos_start, pos_end):
        self.lines.append([pos_start, pos_end])

    def export_map_to_json(self, filename=None):
        export = []
        for line in self.lines:
            export.append({
                "start_pos": line[0],
                "end_pos": line[1],
            })

        if not filename:
            filename = str(datetime.datetime.now())[:19]
            filename = filename.replace("-", "").replace(":", "").replace(" ", "_")

        with open(filename + '.json', 'w') as outfile:
            outfile.write(json.dumps(export, indent=2))

    def load_map_from_json(self, filename):
        with open(filename, 'r') as infile:
            map_data = json.load(infile)
        for item in map_data:
            self.lines.append([item['start_pos'], item['end_pos']])


class Simulation:
    def __init__(self):
        self.win = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.map = Map()
        self.map.load_map_from_json(f'{working_directory}/maps/rect_map_2.json')
        self.player = Player(self.map)

        self.sensitivity = float(config['PROGRAM']['speed_step'])
        self.key_config = {
            pygame.K_q: lambda: self.player.change_vel(0, self.sensitivity),
            pygame.K_a: lambda: self.player.change_vel(0, -self.sensitivity),
            pygame.K_w: lambda: self.player.change_vel(self.sensitivity, self.sensitivity),
            pygame.K_s: lambda: self.player.change_vel(-self.sensitivity, -self.sensitivity),
            pygame.K_e: lambda: self.player.change_vel(self.sensitivity, 0),
            pygame.K_d: lambda: self.player.change_vel(-self.sensitivity, 0),
            pygame.K_x: lambda: self.player.reset_vel(),
        }

    def run(self):
        is_running = True
        while is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_running = False

                if event.type == pygame.KEYDOWN:
                    try:
                        self.key_config[event.key]()
                    except KeyError:
                        pass
                if event.type == pygame.VIDEORESIZE:
                    self.win = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

            self.player.step()
            self.draw()
            self.clock.tick(FPS)

    def draw(self):
        self.clear()
        self.draw_fps()
        self.draw_map()
        self.draw_player()
        pygame.display.flip()

    def draw_fps(self):
        fps_text = FONT.render(f'FPS: {int(self.clock.get_fps())}', False, '#00dd00')
        self.win.blit(fps_text, dest=[5, 5])

    def draw_map(self):
        for line in self.map.lines:
            pygame.draw.line(self.win, '#aaaaaa', line[0], line[1], width=2)

    def draw_player(self):
        # Draw Main Circle
        pygame.draw.circle(self.win, '#00aacc', self.player.pos, self.player.radius)

        # Draw Direction Line
        pygame.draw.line(self.win, '#000000', self.player.pos, (
            self.player.pos[0] + self.player.radius * np.sin(self.player.direction + 0.5 * np.pi),
            self.player.pos[1] - self.player.radius * np.cos(self.player.direction + 0.5 * np.pi),
        ), width=2)

        # Draw all sensor lines
        angle = 2 * np.pi / self.player.num_sensors
        for i in range(self.player.num_sensors):
            start_x = self.player.pos[0] + self.player.radius * np.cos(i * angle - self.player.direction)
            start_y = self.player.pos[1] - self.player.radius * np.sin(i * angle - self.player.direction)
            end_x = self.player.pos[0] + (self.player.vision_range + self.player.radius) * np.cos(
                i * angle - self.player.direction)
            end_y = self.player.pos[1] - (self.player.vision_range + self.player.radius) * np.sin(
                i * angle - self.player.direction)

            self.player.sensor_lines[i][0] = np.array([[start_x, start_y], [end_x, end_y]], dtype=np.float64)

            pygame.draw.line(self.win, '#dd0000', start_pos=[
                start_x,
                start_y,
            ], end_pos=[
                end_x,
                end_y,
            ], width=1)

        # Calculate if one or more sensor line(s) intersect with an object, if so calculate the distance
        self.player.calculate_sensor()

        # Show motor numbers
        x_text = FONT.render(f'l:{int(self.player.vel[1] / self.sensitivity)}', False, '#dddddd')
        y_text = FONT.render(f'r:{int(self.player.vel[0] / self.sensitivity)}', False, '#dddddd')
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
                self.win.blit(text, dest=[
                    self.win.get_width() - 150, 50 + i * 15
                ])

    def clear(self):
        self.win.fill('#232323')


if __name__ == '__main__':
    sim = Simulation()
    sim.run()
    pygame.quit()
