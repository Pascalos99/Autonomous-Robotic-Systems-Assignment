import pygame
from math import sin, cos, pi

BG_COLOR = pygame.Color(200,210,210)
WIDTH, HEIGHT = 800, 600
FPS = 60
DT = 0.05
H = 0.001
DV = 3
DW = 0.15

class DotDisplay:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.theta = 0.
        self.v = 0.
        self.w = 0.
        self.size = size
        self.color = (0, 0, 255)
        self.thickness = 1
        self.win = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.key_config = {
            pygame.K_w: lambda: self.setVW(self.v + DV, self.w),
            pygame.K_s: lambda: self.setVW(self.v - DV, self.w),
            pygame.K_a: lambda: self.setVW(self.v, self.w - DW),
            pygame.K_d: lambda: self.setVW(self.v, self.w + DW),
            pygame.K_x: lambda: self.setVW(0, 0)
        }
    
    def setVW(self, v2, w2):
        self.v = v2
        self.w = w2

    def draw(self):
        self.win.fill(BG_COLOR)
        pygame.draw.circle(self.win, self.color, (self.x, self.y), self.size, self.thickness)
        pygame.draw.line(self.win, '#000000', (self.x, self.y), (
            self.x + self.size * sin(self.theta + 0.5 * pi),
            self.y - self.size * cos(self.theta + 0.5 * pi),), width=2)
        pygame.display.flip()

    def step(self):
        if abs(self.w) > H:
            self.x = self.x + self.v * (sin(self.theta + self.w * DT) - sin(self.theta)) / self.w
            self.y = self.y - self.v * (cos(self.theta + self.w * DT) - cos(self.theta)) / self.w
        else:
            self.x = self.x + self.v * cos(self.theta) * DT
            self.y = self.y + self.v * sin(self.theta) * DT

        self.theta = self.theta + self.w * DT
        
    
    def run(self):
        is_running = True
        while is_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    is_running = False
                    pygame.quit()
                    quit()

                if event.type == pygame.KEYDOWN:
                    try:
                        if event.key in self.key_config.keys():
                            self.key_config[event.key]()
                    except KeyError:
                        pass

                if event.type == pygame.VIDEORESIZE:
                    self.win = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

            self.step()
            self.draw()
            self.clock.tick(FPS)

if __name__ == '__main__':
    disp = DotDisplay(400, 300, 30)
    disp.run()
    pygame.quit()