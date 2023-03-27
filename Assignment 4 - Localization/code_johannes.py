import pygame
import numpy as np
from scipy.optimize import least_squares, minimize
import random

# BG_COLOR = pygame.Color(200,210,210)
BG_COLOR = '#ffffff'
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
        self.v = 0.
        self.w = 0.

        # This matrix shows how certain we are of our current position. 
        # It will be updated every step
        self.covariance = np.array([[0.0001, 0.0, 0.0],
                                    [0.0, 0.0001, 0.0],
                                    [0.0, 0.0, 0.0001]])
        
        self.history_true = [(x, y, 0)]
        self.history_pred = [(x, y, 0)]
        self.history_corr = [(x, y, 0)]
        self.history_covariance = []
        
        self.size = 30
        self.win = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.map = Map(type='sunflower')
        self.landmarks_in_view = []

        self.key_config = {
            pygame.K_w: lambda: self.setVW(self.v + DV, self.w),
            pygame.K_s: lambda: self.setVW(self.v - DV, self.w),
            pygame.K_a: lambda: self.setVW(self.v, self.w - DW),
            pygame.K_d: lambda: self.setVW(self.v, self.w + DW),
            pygame.K_x: lambda: self.setVW(0, 0), 
            pygame.K_r: lambda: self.setXY(*self.xy_init),
            pygame.K_c: lambda: self.resetHistory()
        }
    
    def setVW(self, vel, ang_vel):
        self.v, self.w = vel, ang_vel
    
    def setXY(self, x, y):
        self.x, self.y = x, y
    
    def resetHistory(self):
        self.history_true = [(self.x, self.y, self.theta)]
        self.history_pred = [(self.x, self.y, self.theta)]

    def draw(self):
        self.win.fill(BG_COLOR)
        self.draw_landmarks()
        self.draw_player()
        self.draw_covariance()
        
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
            pygame.draw.line(self.win, '#000000', true_pos_1, true_pos_2, width=2)
            if i % 10 > 5:
                pygame.draw.line(self.win, '#000000', pred_pos_1, pred_pos_2, width=2)
            
    def draw_landmarks(self):
        landmarks_pos = self.map.get_landmark_positions()
        for l_pos in landmarks_pos:
            pygame.draw.circle(self. win, '#000000', l_pos, 3, 3)
        for l_pos in self.landmarks_in_view:
            pygame.draw.line(self.win, '#00bb00', l_pos, (self.x, self.y), width=2)

    def draw_covariance(self):
        for cov in self.history_covariance:
            # TODO rotation
            pygame.draw.ellipse(
                surface = self.win,
                color = '#000000',
                rect = np.abs(cov),
                width = 1,
            )

    def next_state(self, state, v, w):
        x, y, theta = state
        x = x + v * np.cos(theta) * DT
        y = y + v * np.sin(theta) * DT
        theta = theta + w * DT
        return x, y, theta

    def step(self):
        state = (self.x, self.y, self.theta)
        self.x, self.y, self.theta = self.next_state(state, self.v, self.w)

        # Collect landmarks that are in view distance
        landmarks_pos = self.map.get_landmark_positions()
        self.landmarks_in_view = []
        for l_pos in landmarks_pos:
            if np.sqrt((l_pos[0] - self.x)**2 + (l_pos[1] - self.y)**2) < VIEW_DIST:
                self.landmarks_in_view.append(l_pos)
    
    def pred(self):
        """PREDICTION STEP 1
        Formula: µ_t = A_t * µ_{t-1} + B_t * u_t
        (see Presentation 19 ARS, Slide 15)

            A_t:        Identity matrix (thus can be ignored)
            µ_{t-1}:    Previous pose
            B_t:        Update of position
            u_t:        Variables velocity and angular_velocity
        """
        prev_x, prev_y, prev_theta = self.history_pred[-1]
        new_pose = \
            np.array([ # µ_{t-1}
                [prev_x],
                [prev_y],
                [prev_theta]
            ]) + \
            np.array([ # B_t
                [DT * np.cos(prev_theta), 0.0],
                [DT * np.sin(prev_theta), 0.0],
                [0                      , DT]
            ]).dot(
                np.array([self.v, self.w]).reshape((2, 1)) # u_t
            )
        
        new_pose = [new_pose[0, 0], new_pose[1, 0], new_pose[2, 0]]

        # Add simulated noise to new pose:
        new_pose += np.array([
            random.normalvariate(0, 0.1),   # Noise on the x value of bot
            random.normalvariate(0, 0.1),   # Noise on the y value of bot
            random.normalvariate(0, 0.01)]) # Noise on rotation of bot


        """PREDICTION STEP 2
        Formula: Sigma_t = A_t * Sigma_{t-1} * A_t^T + R_t

            A_t:        Identity matrix (thus can be ignored)
            Sigma_{t-1}:Covariance matrix at time t (certainty of position)
            A_t^t:      Transposed identity matrix (can also be ignored)
            R_t:        Process noise (fixed)

        Effective Formula:
        Sigma_{t-1} + R_t
        """
        R_t = np.array([
            [random.normalvariate(0, 1), 0.0, 0.0],
            [0.0, random.normalvariate(0, 1), 0.0],
            [0.0, 0.0, random.normalvariate(0, 1)],
        ])
        self.covariance += R_t

        return new_pose


    def correct(self):
        # If there are no landmarks we cannot apply the correction step.
        if len(self.landmarks_in_view) <= 0:
            return self.history_pred[-1]
        
        # Locate current position of robot using landmarks.
        landmarks = np.array(self.landmarks_in_view)
        
        # ARS 19.21
        [distances] = np.array([np.sqrt((landmarks[:, 0] - self.x)**2 + (landmarks[:, 1] - self.y)**2)]) # r
        [bearings] = np.array([np.arctan2(landmarks[:, 1] - self.y, landmarks[:, 0] - self.x) - self.theta]) # Φ
        
        # Predict location of robot beased on landmarks.
        if len(landmarks) >= 3:
            # Location can be gotten from triangulation.
            predicted_x, predicted_y = self.trilaterate(landmarks[0], landmarks[1], landmarks[2], distances[0], distances[1], distances[2])
        else:
            # TODO: Locate current location from landmarks when less than 3.
            predicted_x = self.history_pred[-1][0]
            predicted_y = self.history_pred[-1][1]
            
        # TODO: Predict orientation (theta) of robot using first landmark.
        predicted_theta = self.history_pred[-1][2]
            
        # TODO: Add noise
        z_t = (
            np.array([predicted_x, predicted_y, predicted_theta])
            #    + np.array([random.normalvariate(0, 1), random.normalvariate(0, 1), random.normalvariate(0, 1)])
        ).T
        
        print("Predicted current location using landmarks:\n", z_t)
        print("Actual current location:\n", np.array([self.x, self.y, self.theta]), "\n")
            
        """
        CORRECTION STEP 1
        Formula: K_t = Sigma_t * (C_t)^T * (C_t * Sigma_t * (C_t)^T + Q_t)^-1
        
            Sigma_t = Coveriance matrix at interval t.
            C_t = Identity matrix.
            (C_t)^T = Transposed identity matrix so just identity matrix.
            Q_t = Noise
        """
        
        Sigma_t = self.covariance
        C_t = np.identity(3)
        Q_t = np.array([
            [random.normalvariate(0, 1), 0.0, 0.0],
            [0.0, random.normalvariate(0, 1), 0.0],
            [0.0, 0.0, random.normalvariate(0, 1)],
        ])
        
        K_t = Sigma_t @ C_t.T @ np.linalg.inv(C_t @ Sigma_t @ C_t.T + Q_t)
        
        """
        CORRECTION STEP 2
        Formula: µ_t = µ_t + K_t * (z_t - C_t * µ_t)
        
            µ_t = Predicted pose for time t.
            K_t = Calculated in previous step.
            z_t = Predicted current state at time t.
            C_t = Identity matrix
        """
        
        mu_t = self.history_pred[-1]
        
        mu_t = mu_t + K_t @ (z_t - C_t @ mu_t)
        
        """
        CORRECTION STEP 3
        Formula: Sigma_t = (I - K_t * C_t) * Sigma_t
        
            I = Identity matrix.
            K_t = Calculated step 1.
            C_t = Identity matrix.
            Sigma_t = Covariance matrix at interval t.
        """
        
        self.covariance = (np.identity(3) - K_t @ C_t) @ self.covariance

        return mu_t


    def triangulate_position(self):
        """
        For the assignment we can calculate the exact position of the robot from landmarks within sensor range.
        This can only be done if 3 or more landmarks are in the vision radius of the bot.
        After the position is triangulated, some noise is added to simulate sensor noise
        """
        lms = []    # List of landmarks (just for easier use, not necessary)
        dists = []  # List of distances for each landmark
        for i in range(3):
            lm = self.landmarks_in_view[i]
            lms.append(lm)
            dists.append(np.sqrt((lm[0] - self.x)**2 + (lm[1] - self.y)**2))
        new_position = self.trilaterate(lms[0], lms[1], lms[2], dists[0], dists[1], dists[2])

        # Add simulated sensor noise
        new_position[0] += random.normalvariate(1, 0.01)
        new_position[1] += random.normalvariate(1, 0.01)

        # After we got a new position, the bearing of the robot has to be adjusted as well
        # For that we choose the first landmark
        new_theta = self.theta 
    
        return new_position[0], new_position[1], new_theta # triangulated x and y, and last known theta

    # Created with ChatGPT
    def trilaterate(self, p1, p2, p3, d1, d2, d3):
        def objective(x):
            return np.sum([(np.linalg.norm(x - p1) - d1)**2,
                        (np.linalg.norm(x - p2) - d2)**2,
                        (np.linalg.norm(x - p3) - d3)**2])
        
        initial_guess = np.array([0, 0])
        result = minimize(objective, initial_guess)
        return result.x




    def run(self):
        is_running = True
        tick_counter = 0
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
            self.history_pred[-1] = self.correct()

            # Add new covariance for plotting
            if tick_counter % (1 * FPS) == 0:
                self.history_covariance.append((
                    self.history_pred[-1][0] - self.covariance[0, 0] // 2,
                    self.history_pred[-1][1] - self.covariance[1, 1] // 2,
                    self.covariance[0, 0],
                    self.covariance[1, 1]
                ))
            self.draw()

            
            pygame.display.flip()
            self.clock.tick(FPS)
            tick_counter += 1

if __name__ == '__main__':
    disp = Simulation(400, 300)
    disp.run()
    pygame.quit()