import pygame
import numpy as np
import datetime
import json
import math
pygame.init()
pygame.font.init()

WIDTH, HEIGHT = 1200, 800
FPS = 60
FONT = pygame.font.SysFont('Consolas', 14)

class Player:
    def __init__(self):
        self.pos = np.array([WIDTH / 2, HEIGHT / 2], dtype=np.float64)
        self.vel = [0, 0]
        self.radius = 30
        self.num_sensors = 12
        self.direction = 0
        self.vision_range = 200

    def change_vel(self, left, right):
        self.vel[0] += left 
        self.vel[1] += right
        
    def reset_vel(self):
        self.vel = [0, 0]
        
    def step(self):
        new_pos = self.get_new_pose()
        print(new_pos)
        self.pos[0] = new_pos[0, 0]
        self.pos[1] = new_pos[1, 0]
        self.direction = new_pos[2, 0]
        
        if self.vel[0] == self.vel[1]:
            self.pos[0] += self.v * np.cos(self.direction)
            self.pos[1] += self.v * np.sin(self.direction)
        
    def get_new_pose(self):
        if self.ICC[0] == float('inf') or self.ICC[1] == float('inf'):
            return np.array([[self.pos[0]], [self.pos[1]], [self.direction]])
        
        m1 = np.array([
                [np.cos(self.w), -np.sin(self.w), 0],
                [np.sin(self.w),  np.cos(self.w), 0],
                [0,               0,              1]
            ])
        m2 = np.array([
                [self.pos[0] - self.ICC[0]],
                [self.pos[1] - self.ICC[1]],
                [self.direction]
            ]) 
        m3 = np.array([
                [self.ICC[0]],
                [self.ICC[1]],
                [self.w]
            ])
        res = np.matmul(m1, m2) + m3
        res = np.nan_to_num(res)
        return res
        
    @property
    def v(self):
        return (self.vel[1] + self.vel[0]) / 2    
        
    @property
    def l(self):
        return 2 * self.radius
        
    @property
    def R(self):
        try:
            return (self.l / 2) * ((self.vel[1] + self.vel[0]) / (self.vel[1] - self.vel[0]))
        except ZeroDivisionError:
            return float('inf')
        
    @property
    def w(self):
        return (self.vel[1] - self.vel[0]) / self.l
    
    @property
    def ICC(self):
        return [
            self.pos[0] - self.R * np.sin(self.direction),
            self.pos[1] + self.R * np.cos(self.direction)
        ]

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
        self.win = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.player = Player()
        self.map = Map()
        self.map.load_map_from_json('./Assignment 2 - Mobile Robot Simulation/rect_map.json')
        
        self.sensitivity = 0.5
        self.key_config = {
            pygame.K_q: lambda: self.player.change_vel(self.sensitivity, 0),
            pygame.K_a: lambda: self.player.change_vel(-self.sensitivity, 0),
            pygame.K_w: lambda: self.player.change_vel(self.sensitivity, self.sensitivity),
            pygame.K_s: lambda: self.player.change_vel(-self.sensitivity, -self.sensitivity),
            pygame.K_e: lambda: self.player.change_vel(0, self.sensitivity),
            pygame.K_d: lambda: self.player.change_vel(0, -self.sensitivity),
            pygame.K_x: lambda: self.player.reset_vel(),
        }
        
    def run(self):
        self.is_running = True
        while self.is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False

                if event.type == pygame.KEYDOWN:
                    try:
                        self.key_config[event.key]()
                    except KeyError:
                        pass
            
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
            pygame.draw.line(self.win, '#dd0000', start_pos=[
                self.player.pos[0] + self.player.radius * np.sin(i * angle+ self.player.direction),
                self.player.pos[1] - self.player.radius * np.cos(i * angle + self.player.direction),
            ], end_pos=[
                self.player.pos[0] + self.player.vision_range * np.sin(i * angle+ self.player.direction),
                self.player.pos[1] - self.player.vision_range * np.cos(i * angle + self.player.direction),
            ], width=1)
        
        # Show motor numbers
        x_text = FONT.render(f'l:{int(self.player.vel[0] / self.sensitivity)}', False, '#dddddd')
        y_text = FONT.render(f'r:{int(self.player.vel[1] / self.sensitivity)}', False, '#dddddd')
        self.win.blit(x_text, dest=[
            self.player.pos[0] - x_text.get_width() // 2 + (self.player.radius // 2) * np.sin(self.player.direction),
            self.player.pos[1] - x_text.get_height() // 2 - (self.player.radius // 2) * np.cos(self.player.direction),
        ])
        self.win.blit(y_text, dest=[
            self.player.pos[0] - y_text.get_width() // 2 + (self.player.radius // 2) * np.sin(np.pi + self.player.direction),
            self.player.pos[1] - y_text.get_height() // 2 - (self.player.radius // 2) * np.cos(np.pi + self.player.direction),
        ])

        # Show Distance Numbers
        for i in range(self.player.num_sensors):
            text = FONT.render(str(self.player.vision_range), False, '#dddddd')
            self.win.blit(text, dest=[
                self.player.pos[0] - text.get_width() // 2 + (self.player.radius + 20) * np.sin(i * angle+ self.player.direction),
                self.player.pos[1] - text.get_height() // 2 - (self.player.radius + 20) * np.cos(i * angle + self.player.direction),
            ])
        
    def clear(self):
        self.win.fill('#232323')
    
if __name__ == '__main__':
    sim = Simulation()
    sim.run()
    pygame.quit()