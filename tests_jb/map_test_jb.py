import pygame
import json
import datetime
from shapely.geometry import LineString
pygame.init()

WIDTH, HEIGHT = 1200, 800
FPS = 60

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

class Collisions:
    def __init__(self):
        pass

    @staticmethod
    def are_lines_colliding(l1, l2):
        line1 = LineString(l1)
        line2 = LineString(l2)
        if line1.intersects(line2):
            return True 
        return False

class Simulation:
    def __init__(self):
        self.win = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.map = Map()
        # self.map.load_map_from_json('./pygame base/map1.json')
        # self.map.load_map_from_json('./pygame base/intersect_test.json')

        # Test for collisions
        # print(Collisions.are_lines_colliding(self.map.lines[1], self.map.lines[2]))
        
    def run(self):
        self.is_running = True
        self.is_drawing = False
        self.is_shift = False 
        self.last_pos = None
        self.current_mouse_pos = [0, 0]

        while self.is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.is_running = False

                # Check for Keypress
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.is_drawing = False
                        self.last_pos = None
                    if event.key == pygame.K_e:
                        self.map.export_map_to_json()

                    if event.key == pygame.K_LSHIFT:
                        self.is_shift = True 

                # Check for Key release
                if event.type == pygame.KEYUP:
                    if event. key == pygame.K_LSHIFT:
                        self.is_shift = False 

                # Check for Mouse Motion
                if event.type == pygame.MOUSEMOTION:
                    if self.is_shift:
                        if abs(event.pos[0] - self.last_pos[0]) < abs(event.pos[1] - self.last_pos[1]):
                            self.current_mouse_pos = [self.last_pos[0], event.pos[1]]
                        else:
                            self.current_mouse_pos = [event.pos[0], self.last_pos[1]]
                    else:
                        self.current_mouse_pos = event.pos

                # Check for mouse button
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if not self.is_drawing:
                        self.is_drawing = True 
                        self.last_pos = self.current_mouse_pos
                    else:
                        self.map.add_line(self.last_pos, self.current_mouse_pos)
                        self.last_pos = self.current_mouse_pos

            self.draw()
            self.clock.tick(FPS)
        
    def draw(self):
        self.clear()
        for line in self.map.lines:
            pygame.draw.line(self.win, '#aaaaaa', line[0], line[1], width=2)
        if self.is_drawing:
            pygame.draw.line(self.win, '#aaaaaa', self.last_pos, self.current_mouse_pos, width=2)
        pygame.display.flip()
        
    def clear(self):
        self.win.fill('#232323')
    
if __name__ == '__main__':
    sim = Simulation()
    sim.run()
    pygame.quit()