import pygame
from math import sin, cos, pi

BG_COLOR = pygame.Color(200,210,210)
WIDTH, HEIGHT = 800, 600
FPS = 60
DT = 0.05
H = 0.001
DA = 0.3
DW = 0.15

class DotDisplay:
    def __init__(self, x, y, size):
        self.xy_init = (x, y)
        self.x = x
        self.y = y
        self.theta = 0.
        self.v = 0.
        self.a = 0.
        self.w = 0.
        self.size = size
        self.color = (0, 0, 255)
        self.thickness = 1
        self.win = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.key_config = {
            pygame.K_w: lambda: self.setAVW(self.a + DA, self.v, self.w),
            pygame.K_s: lambda: self.setAVW(self.a - DA, self.v, self.w),
            pygame.K_a: lambda: self.setAVW(self.a, self.v, self.w - DW),
            pygame.K_d: lambda: self.setAVW(self.a, self.v, self.w + DW),
            pygame.K_x: lambda: self.setAVW(0, 0, 0),
            pygame.K_z: lambda: self.setAVW(0,self.v,self.w),
            pygame.K_r: lambda: self.setXY(*self.xy_init)
        }
    
    def setAVW(self, a2, v2, w2):
        self.a, self.v, self.w = a2, v2, w2

    def setXY(self, x2, y2):
        self.x, self.y = x2, y2

    def draw(self):
        self.win.fill(BG_COLOR)
        pygame.draw.circle(self.win, self.color, (self.x, self.y), self.size, self.thickness)
        pygame.draw.line(self.win, '#000000', (self.x, self.y), (
            self.x + self.size * sin(self.theta + 0.5 * pi),
            self.y - self.size * cos(self.theta + 0.5 * pi),), width=2)
        pygame.display.flip()

    def step(self):
        if abs(self.w) > H:
            self.x = self.x + self.a * (cos(self.theta + self.w * DT) - cos(self.theta)) / (self.w**2)
            self.x += ((self.a * DT + self.v) * sin(self.theta + self.w * DT) - self.v * sin(self.theta)) / self.w
            self.y = self.y + self.a * (sin(self.theta + self.w * DT) - sin(self.theta)) / (self.w**2)
            self.y -= ((self.a * DT + self.v) * cos(self.theta + self.w * DT) - self.v * cos(self.theta)) / self.w
        else:
            self.x = self.x + 0.5 * DT * (self.a * DT + 2.*self.v) * cos(self.theta)
            self.y = self.y + 0.5 * DT * (self.a * DT + 2.*self.v) * sin(self.theta)

        self.theta = self.theta + self.w * DT
        self.v = self.v + self.a * DT
        
    
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