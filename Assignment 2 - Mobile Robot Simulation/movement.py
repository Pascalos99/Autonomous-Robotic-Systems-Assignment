import pygame
import numpy as np
import datetime
import pathlib
import json
import math
pygame.init()
pygame.font.init()

working_directory = pathlib.Path(__file__).parent.absolute()

WIDTH, HEIGHT = 1200, 800
FPS = 60
FONT = pygame.font.SysFont('Consolas', 14)

class Player:
    def __init__(self, player_map):
        self.map = player_map
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
        
        # Get current and new position.
        current_position = self.pos
        new_position = np.array([new_pos[0, 0], new_pos[1, 0]])
        direction = new_pos[2, 0]
        
        if self.vel[0] == self.vel[1]:
            new_position[0] += self.v * np.cos(direction)
            new_position[1] += self.v * np.sin(direction)
    
        # Check if position has moved.
        if not np.array_equal(self.pos, new_position):
            # Check if new position is inside a wall or has passed through a wall.
            if self.position_intersects_wall(new_position):
                # If so, move the player to the closest point on the wall.
                # Number of steps to check between current and new position to find the closest point on the wall.
                num_steps = 100
                x_positions = np.linspace(current_position[0], new_position[0], num_steps)
                y_positions = np.linspace(current_position[1], new_position[1], num_steps)
                while self.position_intersects_wall(new_position):
                    new_position = np.array([x_positions[-1], y_positions[-1]])
                    
                    x_positions = np.delete(x_positions, -1)
                    y_positions = np.delete(y_positions, -1)
                    
                # Make robot move along the wall.
                if np.array_equal(current_position, new_position):
                    circle_intersections = []
                    for line in self.map.lines:
                        intersection = self.get_intersection_circle_line(line)
                        if intersection is not None:
                            circle_intersections.append(intersection)
                            
                    if circle_intersections:
                        print("Moving along wall")
                     
        # Update position and direction.
        self.pos[0] = new_position[0]
        self.pos[1] = new_position[1]
        self.direction = direction
        
        
    def position_intersects_wall(self, position):
        player_lines = {
            "left": np.array([self.pos - [self.radius, 0], position - [self.radius, 0]]),
            "right": np.array([self.pos + [self.radius, 0], position + [self.radius, 0]]),
            "upper": np.array([self.pos + [0, self.radius], position + [0, self.radius]]),
            "bottom": np.array([self.pos - [0, self.radius], position - [0, self.radius]])
        }
        
        for line in self.map.lines:
            for player_line in player_lines.values():
                intersection = self.get_intersection_lines(line, player_line)
                if intersection is not None:
                    return True
                
        return False
    
    
    def get_intersection_circle_line(self, line):
        pass
        
    def get_intersection_lines(self, line_1, line_2):
        # Calculate intersection point between two lines using https://en.m.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line_segment.
        x1, y1, x2, y2 = line_1[0][0], line_1[0][1], line_1[1][0], line_1[1][1]
        x3, y3, x4, y4 = line_2[0][0], line_2[0][1], line_2[1][0], line_2[1][1]
        
        t = np.divide((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4), (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        u = np.divide((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2), (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
        
        if t >= 0 and t <= 1 and u >= 0 and u <= 1:
            x_intercept, y_intercept = (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
            
            return np.array([x_intercept, y_intercept])
        
        return None
        
    def get_new_pose(self):
        if self.ICC[0] == float('inf') or self.ICC[1] == float('inf') or self.ICC[0] == float('-inf') or self.ICC[1] == float('-inf'):
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
        self.map = Map()
        self.map.load_map_from_json(f'{working_directory}/rect_map.json')
        self.player = Player(self.map)
        
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