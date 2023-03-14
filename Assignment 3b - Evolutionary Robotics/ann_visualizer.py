import numpy as np
import pygame
from scipy.interpolate import interp1d

from neural_GA import ANN


class ANN_Visualizer:
    def __init__(self, ann: ANN, win: pygame.Surface) -> None:
        self.neuron_size = 10
        self.neuron_gap = self.neuron_size + 10
        self.layer_gap = 125
        self.top_dist = 50

        self.neuron_positions = []

        self.out_win = win
        self.num_inputs = ann.num_inputs
        self.hidden_layers = ann.hidden_layers
        self.num_outputs = ann.num_outputs

        self.weights = ann.network
        self.activations = ann.activations

        _ = self._draw_neurons(self._create_surface())
        self.connection_surface = self._create_surface()
        self.connection_surface = self._draw_connections(self.connection_surface)

    def generate_img(self):
        # Create Neuron Surface
        neuron_surface = self._create_surface()
        neuron_surface = self._draw_neurons(neuron_surface)

        s = self._create_surface()
        s.blit(self.connection_surface, (0, 0))
        s.blit(neuron_surface, (0, 0))
        return s

    def _create_surface(self):
        s = pygame.Surface(size=(
            2 * (2 + len(self.hidden_layers)) * self.neuron_size + (len(self.hidden_layers) + 1) * self.layer_gap,
            2 * self.top_dist + 2 * self.num_inputs * self.neuron_size + (self.num_inputs - 2) * self.neuron_gap)
        ).convert_alpha()
        s.fill([0, 0, 0, 0])  # make transparent
        return s

    def _draw_neurons(self, s):
        # Draw Input Neurons
        self.neuron_positions.append([])
        for i in range(self.num_inputs):
            circle_center = (
                self.neuron_size,
                self.top_dist + 2 * i * self.neuron_size + i * self.neuron_gap)

            self.neuron_positions[-1].append(circle_center)
            pygame.draw.circle(
                surface=s,
                color=self._get_neuron_color(type='inp', value=self.activations[0][i]),
                center=circle_center,
                radius=self.neuron_size)

        # Draw Hidden Neurons
        for i in range(len(self.hidden_layers)):
            self.neuron_positions.append([])
            for j in range(self.hidden_layers[i]):
                circle_center = (
                    (i + 2) * self.neuron_size + (i + 1) * self.layer_gap,
                    ((self._create_surface().get_rect().height) / 2)
                    - (self.hidden_layers[i] * 2 * self.neuron_size + (self.hidden_layers[i] - 2) * self.neuron_gap) / 2
                    + 2 * j * self.neuron_size + j * self.neuron_gap)

                self.neuron_positions[-1].append(circle_center)
                pygame.draw.circle(
                    surface=s,
                    color=self._get_neuron_color(type='hid', value=self.activations[i][j], layer=i),
                    center=circle_center,
                    radius=self.neuron_size)

        # Draw Output Neurons
        self.neuron_positions.append([])
        for i in range(self.num_outputs):
            circle_center = (
                (len(self.hidden_layers) + 2) * self.neuron_size + (len(self.hidden_layers) + 1) * self.layer_gap,
                ((self._create_surface().get_rect().height) / 2)
                - (self.num_outputs * 2 * self.neuron_size + (self.num_outputs - 2) * self.neuron_gap) / 2
                + 2 * i * self.neuron_size + i * self.neuron_gap)

            self.neuron_positions[-1].append(circle_center)
            c = pygame.draw.circle(
                surface=s,
                color=self._get_neuron_color(type='out', value=self.activations[-1][i],
                                             layer=len(self.activations) - 1),
                center=circle_center,
                radius=self.neuron_size)
        return s

    def _draw_connections(self, s):
        # Draw connections from
        for layer in range(len(self.neuron_positions) - 1):
            for i, neuron_in in enumerate(self.neuron_positions[layer]):
                for j, neuron_out in enumerate(self.neuron_positions[layer + 1]):
                    pygame.draw.line(
                        surface=s,
                        color=self._get_line_color(layer=layer, value=self.weights[layer][i][j]),
                        start_pos=(neuron_in),
                        end_pos=(neuron_out),
                        width=2)
        return s

    def _draw_recurrent_connections(self, s):
        pass  # TODO

    def _get_line_color(self, layer, value=None) -> pygame.Color:
        all_layer_weights = np.ndarray.flatten(self.weights[layer])
        conv = interp1d([min(all_layer_weights), max(all_layer_weights)], [50, 255])
        return pygame.Color([conv(value), conv(value), conv(value)])

    def _get_neuron_color(self, type, value=None, layer=0) -> pygame.Color:
        if type == 'inp':
            conv_r = interp1d([min(self.activations[layer]), max(self.activations[layer])], [50, 123])
            conv_g = interp1d([min(self.activations[layer]), max(self.activations[layer])], [50, 241])
            conv_b = interp1d([min(self.activations[layer]), max(self.activations[layer])], [50, 168])
            return pygame.Color([conv_r(value), conv_g(value), conv_b(value)])

        if type == 'hid':
            conv_r = interp1d([min(self.activations[layer]), max(self.activations[layer])], [50, 233])
            conv_g = interp1d([min(self.activations[layer]), max(self.activations[layer])], [50, 255])
            conv_b = interp1d([min(self.activations[layer]), max(self.activations[layer])], [50, 112])
            return pygame.Color([conv_r(value), conv_g(value), conv_b(value)])

        if type == 'out':
            try:
                conv_r = interp1d([-1., 1.], [50, 255])
                conv_g = interp1d([-1., 1.], [50, 112])
                conv_b = interp1d([-1., 1.], [50, 166])
                return pygame.Color([conv_r(value), conv_g(value), conv_b(value)])
            except ValueError:
                print(self.activations[layer])

    def animate(self, ann):
        self.weights = ann.network
        self.activations = ann.activations
        return self.generate_img()
