import pygame
import numpy as np
import random

BG_COLOR = pygame.Color(200,210,210)
WIDTH, HEIGHT = 800, 600
FPS = 60
DT = 0.05   # Delta T (time)
H = 0.001   # ?
DV = 3      # Delta V (velocity)
DW = 0.15   # Delta W (angular velocity)

VIEW_DIST = 100 # threshold for seeing landmarks
history = 2000  # number of history points

class Map:
    def __init__(self, type='sunflower'):
        self.num_landmarks = 25
        self.landmarks = []
        
        if type == 'random':
            self.init_random_landmarks()
        elif type == 'sunflower':
            self.init_sunflower_landmarks()
        
    def init_random_landmarks(self):
        for _ in range(self.num_landmarks):
            self.landmarks.append([
                random.uniform(0, WIDTH),
                random.uniform(0, HEIGHT)
            ])
        
    # Thank you: https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere 
    # This places all landmarks inside of a circle with same distance to each other
    def init_sunflower_landmarks(self):
        indices = np.arange(0, self.num_landmarks, dtype=float) + 0.5
        rs = np.sqrt(indices/self.num_landmarks)
        thetas = np.pi * (1 + 5**0.5) * indices

        for r, theta in zip(rs, thetas):
            self.landmarks.append([
                r * np.cos(theta) * WIDTH // 2 + WIDTH // 2, 
                r * np.sin(theta) * HEIGHT // 2 + HEIGHT // 2,
            ])
            
    def get_landmark_positions(self):
        return self.landmarks

class Simulation:
    def __init__(self, x, y):
        self.xy_init = (x, y)
        self.x = x
        self.y = y
        self.theta = 0.
        self.vel = 0.
        self.ang_vel = 0.
        self.history_true = [(x, y, 0)]
        self.history_pred = [(x, y, 0)]
        
        self.size = 30
        self.win = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.map = Map(type='sunflower')
        self.landmarks_in_view = []

        self.key_config = {
            pygame.K_w: lambda: self.setVW(self.vel + DV, self.ang_vel),
            pygame.K_s: lambda: self.setVW(self.vel - DV, self.ang_vel),
            pygame.K_a: lambda: self.setVW(self.vel, self.ang_vel - DW),
            pygame.K_d: lambda: self.setVW(self.vel, self.ang_vel + DW),
            pygame.K_x: lambda: self.setVW(0, 0), 
            pygame.K_r: lambda: self.setXY(*self.xy_init),
            pygame.K_c: lambda: self.resetHistory()
        }
    
    def setVW(self, vel, ang_vel):
        self.vel, self.ang_vel = vel, ang_vel
    
    def setXY(self, x, y):
        self.x, self.y = x, y
    
    def resetHistory(self):
        self.history_true = [(self.x, self.y, self.theta)]
        self.history_pred = [(self.x, self.y, self.theta)]

    def draw(self):
        self.win.fill(BG_COLOR)
        self.draw_landmarks()
        self.draw_player()
        pygame.display.flip()
        
    def draw_player(self):
        pygame.draw.circle(self.win, '#0000ff', (self.x, self.y), self.size, 1)
        pygame.draw.line(self.win, '#ff0000', (self.x, self.y), (
                self.x + self.size * np.sin(self.theta + 0.5 * np.pi),
                self.y - self.size * np.cos(self.theta + 0.5 * np.pi),
            ), 
            width=2)
        
        for i in range(len(self.history_true) - 1)[-history:]:
            true_pos_1, true_pos_2 = self.history_true[i][:2], self.history_true[i+1][:2]
            pred_pos_1, pred_pos_2 = self.history_pred[i][:2], self.history_pred[i+1][:2]
            pygame.draw.line(self.win, '#00aa00', true_pos_1, true_pos_2)
            pygame.draw.line(self.win, '#000000', pred_pos_1, pred_pos_2)
            
    def draw_landmarks(self):
        landmarks_pos = self.map.get_landmark_positions()
        for l_pos in landmarks_pos:
            pygame.draw.circle(self. win, '#000000', l_pos, 3, 3)
        for l_pos in self.landmarks_in_view:
            pygame.draw.line(self.win, '#000000', l_pos, (self.x, self.y))

    def next_state(self, state, vel, ang_vel):
        x, y, theta = state
        if abs(ang_vel) > H:
            x = x + vel * (np.sin(theta + ang_vel * DT) - np.sin(theta)) / ang_vel
            y = y - vel * (np.cos(theta + ang_vel * DT) - np.cos(theta)) / ang_vel
        else:
            x = x + vel * np.cos(theta) * DT
            y = y + vel * np.sin(theta) * DT
        theta = theta + ang_vel * DT
        return x, y, theta

    def step(self):
        state = (self.x, self.y, self.theta)
        self.x, self.y, self.theta = self.next_state(state, self.vel, self.ang_vel)

        # Collect landmarks in view distance
        landmarks_pos = self.map.get_landmark_positions()
        self.landmarks_in_view = []
        for l_pos in landmarks_pos:
            if np.sqrt((l_pos[0] - self.x)**2 + (l_pos[1] - self.y)**2) < VIEW_DIST:
                self.landmarks_in_view.append(l_pos)
    
    def pred(self):
        # this is to 'simulate' what it might look like if we have an actual Karman filter with noise implemented
        return self.next_state(self.history_pred[-1], self.vel * random.normalvariate(1, 0.5), self.ang_vel * random.normalvariate(1, 0.5))

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
            self.history_true.append((self.x, self.y, self.theta))
            self.history_pred.append(self.pred())
            self.draw()
            self.clock.tick(FPS)

if __name__ == '__main__':
    disp = Simulation(400, 300)
    disp.run()
    pygame.quit()