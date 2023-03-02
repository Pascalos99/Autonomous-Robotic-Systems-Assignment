import configparser
import pathlib
from typing import Union

import numpy as np

working_directory = pathlib.Path(__file__).parent.absolute()
config = configparser.ConfigParser()
config.read(f"{working_directory}/config.ini")

WIDTH, HEIGHT = float(config['PROGRAM']['window_width']), float(config['PROGRAM']['window_height'])


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

    def change_vel(self, left: float, right: float):
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
                # Save the intended new position.
                intended_new_position = new_position
                
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
                        # Clear intersections which are not in the movement direction of the robot.
                        # Only needs to be done if the robot is colliding with more than one wall.
                        if len(circle_intersections) > 1:
                            correct_intersections = []
                            
                            x1, y1 = current_position[0], current_position[1]
                            x2, y2 = intended_new_position[0], intended_new_position[1]
                            for intersection in circle_intersections:
                                # Check if the wall intersects with the original movement direction of the robot.
                                wall = intersection[0]
                                
                                x3, y3 = wall[0][0], wall[0][1]
                                x4, y4 = wall[1][0], wall[1][1]
                                
                                denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
                                if denominator != 0:
                                    u = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denominator
                                    if u > 0:
                                        # Wall intersects with the original movement direction of the robot so add it to the correct intersections.
                                        correct_intersections.append(intersection)
                            
                            # Replace the circle intersections with the correct ones.
                            circle_intersections = correct_intersections
                        
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

        # Update position and direction.
        self.pos[0] = new_position[0]
        self.pos[1] = new_position[1]
        self.direction = direction

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

    def position_intersects_wall(self, position: np.ndarray):
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

    def calculate_sensor(self):
        for sensor_number, sensor in self.sensor_lines.items():
            no_intersect_counter = 0
            sensor_line = sensor[0]

            # Check if line intersects with any wall.
            for wall in self.map.lines:
                intersect_coordinates = self.get_intersection_lines(sensor_line, wall)
                if intersect_coordinates is not None:
                    # Calculate distance with Pythagorean theorem.
                    sensor_distance = ((intersect_coordinates[0] - sensor_line[0][0]) ** 2 +
                                       (intersect_coordinates[1] - sensor_line[0][1]) ** 2) ** 0.5
                    self.sensor_lines[sensor_number][1] = sensor_distance
                else:
                    no_intersect_counter += 1
                    if no_intersect_counter == len(self.map.lines):
                        # Sensor line did not intersect with any objects, so reset distance number to the vision range.
                        self.sensor_lines[sensor_number][1] = self.vision_range

    @staticmethod
    def get_intersection_lines(line_1: Union[list, np.ndarray], line_2: Union[list, np.ndarray]) -> Union[np.ndarray, None]:
        # Calculate intersection point between two lines using:
        # https://en.m.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line_segment.
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

    def get_intersection_circle_line(self, line: Union[list, np.ndarray], position: Union[list, np.ndarray], radius: float) -> Union[np.ndarray, None]:
        # Calculate the distance between the center of the circle and the line.
        x_diff = line[1][0] - line[0][0]
        y_diff = line[1][1] - line[0][1]
        num = abs(y_diff * position[0] - x_diff * position[1] + line[1][0] * line[0][1] - line[1][1] * line[0][0])
        den = np.sqrt(y_diff ** 2 + x_diff ** 2)

        if num / den > radius:
            # The circle and line segment do not intersect.
            return None

        # Calculate the closest point on the line to the center of the circle
        u = ((position[0] - line[0][0]) * x_diff + (position[1] - line[0][1]) * y_diff) / (den ** 2)
        closest_point = np.array([line[0][0] + u * x_diff, line[0][1] + u * y_diff])

        # Calculate the distance between the closest point and the center of the circle
        dist_to_closest_point = np.sqrt((closest_point[0] - position[0]) ** 2 + (closest_point[1] - position[1]) ** 2)

        if dist_to_closest_point > radius:
            # Distance between the closest point and the center of the circle is greater than the radius of the circle.
            return None

        # Calculate the distance between the intersection points and the closest point.
        dist_to_intersection = np.sqrt(radius ** 2 - dist_to_closest_point ** 2)

        # Calculate the intersection points.
        if y_diff == 0:
            # Horizontal line.
            intersection_1 = np.array([closest_point[0] + dist_to_intersection, closest_point[1]])
            intersection_2 = np.array([closest_point[0] - dist_to_intersection, closest_point[1]])
        elif x_diff == 0:
            # Vertical line.
            intersection_1 = np.array([closest_point[0], closest_point[1] + dist_to_intersection])
            intersection_2 = np.array([closest_point[0], closest_point[1] - dist_to_intersection])
        else:
            # Diagonal line.
            m = y_diff / x_diff
            b = line[0][1] - m * line[0][0]
            x_1 = closest_point[0] + (dist_to_intersection / np.sqrt(1 + m ** 2))
            x_2 = closest_point[0] - (dist_to_intersection / np.sqrt(1 + m ** 2))
            intersection_1 = np.array([x_1, m * x_1 + b])
            intersection_2 = np.array([x_2, m * x_2 + b])
            
        # Check if the intersection points lie on the line segment.
        intersections = []
        if self.point_on_line_segment(line, intersection_1):
            intersections.append(intersection_1)
        if self.point_on_line_segment(line, intersection_2) and not np.array_equal(intersection_1, intersection_2):
            intersections.append(intersection_2)

        # Return the intersection points.
        if len(intersections) == 0:
            return None
        
        return intersections

    @staticmethod
    def point_on_line_segment(line, point):
        x1, y1 = line[0][0], line[0][1]
        x2, y2 = line[1][0], line[1][1]

        if min(x1, x2) <= point[0] <= max(x1, x2) and min(y1, y2) <= point[1] <= max(y1, y2):
            return True

        return False

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
