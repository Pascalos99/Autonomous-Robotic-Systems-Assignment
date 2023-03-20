import pygame
from math import sin, cos, pi
import random

BG_COLOR = pygame.Color(200,210,210)
WIDTH, HEIGHT = 800, 600
FPS = 60
DT = 0.05
H = 0.001
DA = 0.3
DW = 0.15
history = 2000

class DotDisplay:
    def __init__(self, x, y, size):
        self.xy_init = (x, y)
        self.x = x
        self.y = y
        self.theta = 0.
        self.v = 0.
        self.a = 0.
        self.w = 0.
        self.Htrue = [(x, y, 0, 0)]
        self.Hpred = [(x, y, 0, 0)]
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
            pygame.K_r: lambda: self.setXY(*self.xy_init),
            pygame.K_c: lambda: self.resetHistory()
        }
    
    def setAVW(self, a2, v2, w2):
        self.a, self.v, self.w = a2, v2, w2

    def setXY(self, x2, y2):
        self.x, self.y = x2, y2

    def resetHistory(self):
        self.Htrue, self.Hpred = [(self.x, self.y, self.v, self.theta)], [(self.x, self.y, self.v, self.theta)]

    def draw(self):
        self.win.fill(BG_COLOR)
        pygame.draw.circle(self.win, self.color, (self.x, self.y), self.size, self.thickness)
        pygame.draw.line(self.win, '#ff0000', (self.x, self.y), (
            self.x + self.size * sin(self.theta + 0.5 * pi),
            self.y - self.size * cos(self.theta + 0.5 * pi),), width=2)
        for i in range(len(self.Htrue)-1)[-history:]:
            h1, h2 = self.Htrue[i][:2], self.Htrue[i+1][:2]
            p1, p2 = self.Hpred[i][:2], self.Hpred[i+1][:2]
            pygame.draw.line(self.win, '#00aa00', p1, p2)
            pygame.draw.line(self.win, '#000000', h1, h2)
            
        pygame.display.flip()

    def next_state(self, state, a, w):
        x, y, v, theta = state
        if abs(w) > H:
            x = x + a * (cos(theta + w * DT) - cos(theta)) / (w**2)
            x += ((a * DT + v) * sin(theta + w * DT) - v * sin(theta)) / w
            y = y + a * (sin(theta + w * DT) - sin(theta)) / (w**2)
            y -= ((a * DT + v) * cos(theta + w * DT) - v * cos(theta)) / w
        else:
            x = x + 0.5 * DT * (a * DT + 2.*v) * cos(theta)
            y = y + 0.5 * DT * (a * DT + 2.*v) * sin(theta)
        theta = theta + w * DT
        v = v + a * DT
        return x, y, v, theta
    
    def step(self):
        state = (self.x, self.y, self.v, self.theta)
        self.x, self.y, self.v, self.theta = self.next_state(state, self.a, self.w)
    
    def pred(self):
        # this is to 'simulate' what it might look like if we have an actual Karman filter with noise implemented
        return self.next_state(self.Hpred[-1], self.a * random.normalvariate(1, 0.25), self.w * random.normalvariate(1, 0.5))
    
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
            self.Htrue.append((self.x, self.y, self.v, self.theta))
            self.Hpred.append(self.pred())
            self.draw()
            self.clock.tick(FPS)

if __name__ == '__main__':
    disp = DotDisplay(400, 300, 30)
    disp.run()
    pygame.quit()