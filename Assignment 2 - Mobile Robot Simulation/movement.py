import configparser
import datetime
import json
import pathlib
import math
import numpy as np
import pygame

pygame.init()
pygame.font.init()

working_directory = pathlib.Path(__file__).parent.absolute()
config = configparser.ConfigParser()
config.read(f"{working_directory}/config.ini")

WIDTH, HEIGHT = float(config['PROGRAM']['window_width']), float(config['PROGRAM']['window_height'])
FPS = float(config['PROGRAM']['fps'])
FONT = pygame.font.SysFont('Consolas', 14)


class Player:
    def __init__(self, player_map):
        self.map = player_map
        self.pos = np.array([WIDTH / 2, HEIGHT / 2], dtype=np.float64)
        self.vel = [0, 0]
        self.radius = int(config['BOT']['radius'])
        self.num_sensors = int(config['BOT']['num_sensors'])
        self.vision_range = int(config['BOT']['vision_range'])
        self.sensor_lines = {key: [None, self.vision_range] for key in range(self.num_sensors)}
        self.direction = -float(config['BOT']['direction']) * np.pi / 180

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
                while self.position_intersects_wall(new_position) and len(x_positions) > 0:
                    new_position = np.array([x_positions[-1], y_positions[-1]])

                    x_positions = np.delete(x_positions, -1)
                    y_positions = np.delete(y_positions, -1)
                    
                if len(x_positions) == 0:
                    new_position = current_position
                    
                # Make robot move along the wall.
                if np.array_equal(current_position, new_position):
                    circle_intersections = []
                    for line in self.map.lines:
                        intersection = self.get_intersection_circle_line(line, new_position, (self.radius + 1))
                        
                        if intersection is not None:
                            circle_intersections.append([line, intersection])
                            
                    if circle_intersections:
                        new_position = current_position
                        for intersection in circle_intersections:
                            # Get the wall that collides with the robot and the collision point.
                            wall = intersection[0]
                            collision_point = intersection[1][0]
                            
                            # Get wall direction and unit vector.
                            wall_direction = np.array([wall[1][0] - wall[0][0], wall[1][1] - wall[0][1]])
                            wall_direction_unit = wall_direction / np.linalg.norm(wall_direction)
                            
                            # Calculate the velocity of the robot and where it should move to.
                            velocity = (self.vel[0] + self.vel[1]) / 2
                            velocity_vector = np.array([velocity * np.cos(direction), velocity * np.sin(direction)])

                            # Calculate the parallel velocity.
                            parallel_velocity = np.dot(velocity_vector, wall_direction_unit)
                            velocity_vector = parallel_velocity * wall_direction_unit
                            
                            # Calculate the new position.
                            new_position = new_position + velocity_vector
                        
                        # Check if new position is inside a wall or has passed through a wall.
                        num_steps = 100
                        x_positions = np.linspace(current_position[0], new_position[0], num_steps)
                        y_positions = np.linspace(current_position[1], new_position[1], num_steps)
                        while self.position_intersects_wall(new_position) == True and len(x_positions) > 0:
                            new_position = np.array([x_positions[-1], y_positions[-1]])
                            
                            x_positions = np.delete(x_positions, -1)
                            y_positions = np.delete(y_positions, -1)
                        
                        if len(x_positions) == 0:
                            new_position = current_position
                        
        # Update position and direction.awwd
        self.pos[0] = new_position[0]
        self.pos[1] = new_position[1]
        self.direction = direction

    def position_intersects_wall(self, position):
        # Check if the robot intersects with any of the walls using lines.
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
                
        # Lastly, check if the robot intersects with any of the walls using circle at location.
        for line in self.map.lines:
            intersection = self.get_intersection_circle_line(line, position, self.radius)
            if intersection is not None and len(intersection) > 1:
                return True
                
        return False
    
    
    def get_intersection_circle_line(self, line, position, radius):
        # Calculate the distance between the center of the circle and the line
        x_diff = line[1][0] - line[0][0]
        y_diff = line[1][1] - line[0][1]
        num = abs(y_diff * position[0] - x_diff * position[1] + line[1][0] * line[0][1] - line[1][1] * line[0][0])
        den = np.sqrt(y_diff**2 + x_diff**2)

        if num / den > radius:
            # The circle and line segment do not intersect.
            return None

        # Calculate the closest point on the line to the center of the circle
        u = ((position[0] - line[0][0]) * x_diff + (position[1] - line[0][1]) * y_diff) / (den**2)
        closest_point = np.array([line[0][0] + u * x_diff, line[0][1] + u * y_diff])

        # Calculate the distance between the closest point and the center of the circle
        dist_to_closest_point = np.sqrt((closest_point[0] - position[0])**2 + (closest_point[1] - position[1])**2)

        if dist_to_closest_point > radius:
            # Distance between the closest point and the center of the circle is greater than the radius of the circle.
            return None

        # Calculate the distance between the intersection points and the closest point
        dist_to_intersection = np.sqrt(radius**2 - dist_to_closest_point**2)

        # Calculate the intersection points
        if y_diff == 0: 
            # Horizontal line
            intersection_1 = np.array([closest_point[0] + dist_to_intersection, closest_point[1]])
            intersection_2 = np.array([closest_point[0] - dist_to_intersection, closest_point[1]])
        elif x_diff == 0:
            # Vertical line
            intersection_1 = np.array([closest_point[0], closest_point[1] + dist_to_intersection])
            intersection_2 = np.array([closest_point[0], closest_point[1] - dist_to_intersection])
        else: 
            # Diagonal line
            m = y_diff / x_diff
            b = line[0][1] - m * line[0][0]
            x_1 = closest_point[0] + (dist_to_intersection / np.sqrt(1 + m**2))
            x_2 = closest_point[0] - (dist_to_intersection / np.sqrt(1 + m**2))
            intersection_1 = np.array([x_1, m * x_1 + b])
            intersection_2 = np.array([x_2, m * x_2 + b])

        # Return the intersection points
        if np.array_equal(intersection_1, intersection_2):
            return [intersection_1]
        else:
            return [intersection_1, intersection_2]
        
        
    def get_intersection_lines(self, line_1, line_2):
        # Calculate intersection point between two lines using https://en.m.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line_segment.
        x1, y1, x2, y2 = line_1[0][0], line_1[0][1], line_1[1][0], line_1[1][1]
        x3, y3, x4, y4 = line_2[0][0], line_2[0][1], line_2[1][0], line_2[1][1]

        t = np.divide(
            (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4),
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        )
        u = np.divide(
            (x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2),
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        )

        if 0 <= t <= 1 and 0 <= u <= 1:
            x_intercept, y_intercept = (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
            return np.array([x_intercept, y_intercept])

        return None
        
        
    def point_on_line_segment(self, line, point):
        x1, y1 = line[0][0], line[0][1]
        x2, y2 = line[1][0], line[1][1]
        
        if min(x1, x2) <= point[0] <= max(x1, x2) and min(y1, y2) <= point[1] <= max(y1, y2):
            return True
        
        return False
        
        
    def get_new_pose(self):
        if (self.ICC[0] == float('inf') or
                self.ICC[1] == float('inf') or
                self.ICC[0] == float('-inf') or
                self.ICC[1] == float('-inf')):
            return np.array([[self.pos[0]], [self.pos[1]], [self.direction]])

        m1 = np.array([
            [np.cos(self.w), -np.sin(self.w), 0],
            [np.sin(self.w), np.cos(self.w), 0],
            [0, 0, 1]
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

    def calculate_sensor(self):
        for sensor_number, sensor in self.sensor_lines.items():
            no_intersect_counter = 0
            sensor_line = sensor[0]
            x1, y1, x2, y2 = sensor_line[0][0], sensor_line[0][1], sensor_line[1][0], sensor_line[1][1]

            # Check if line intersects with any wall.
            for wall in self.map.lines:
                x3, y3, x4, y4 = wall[0][0], wall[0][1], wall[1][0], wall[1][1]

                # Calculate intersection point of line and wall using:
                # https://en.m.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line_segment.
                t = np.divide(
                    (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4),
                    (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                )
                u = np.divide(
                    (x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2),
                    (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                )

                if 0 <= t <= 1 and 0 <= u <= 1:
                    # Line intersects with a wall.
                    no_intersect_counter = 0
                    # Calculate x and y coordinates of intersection point.
                    x_intercept, y_intercept = (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
                    # Calculate distance with Pythagorean theorem.
                    sensor_distance = ((x_intercept - x1) ** 2 + (y_intercept - y1) ** 2) ** 0.5
                    self.sensor_lines[sensor_number][1] = sensor_distance
                else:
                    no_intersect_counter += 1
                    if no_intersect_counter == len(self.map.lines):
                        # Sensor line did not intersect with any objects, so reset distance number to the vision range.
                        self.sensor_lines[sensor_number][1] = self.vision_range

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
        self.map.load_map_from_json(f'{working_directory}/rect_map_2.json')
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
            text = FONT.render(str(int(round(distance, 0))), False, '#dddddd')
            self.win.blit(text, dest=[
                (self.player.pos[0] - text.get_width() // 2 + (self.player.radius + 20)
                 * np.cos(i * angle - self.player.direction)),
                (self.player.pos[1] - text.get_height() // 2 - (self.player.radius + 20)
                 * np.sin(i * angle - self.player.direction)),
            ])

    def clear(self):
        self.win.fill('#232323')


if __name__ == '__main__':
    sim = Simulation()
    sim.run()
    pygame.quit()
