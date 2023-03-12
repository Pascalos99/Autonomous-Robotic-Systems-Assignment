import datetime
import json
import numpy as np

class Map:
    def __init__(self):
        self.lines = []
        self.line_width = 2
        self.wall_segments = []

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

            distance_x = abs(item['start_pos'][0] - item['end_pos'][0])
            distance_y = abs(item['start_pos'][1] - item['end_pos'][1])
            distance = np.sqrt(distance_x ** 2 + distance_y ** 2)

            if self.line_width % 2 != 0:
                first_width = (self.line_width - 1) // 2
                second_width = (self.line_width - 1) // 2
            else:
                first_width = (self.line_width - 1) // 2
                second_width = (self.line_width - 1) // 2 + 1

            dis_x_a = distance_x / distance
            dis_y_a = distance_y / distance

            edge_1 = [
                [(item['start_pos'][0] - (first_width * dis_y_a)), (item['start_pos'][1] + (second_width * dis_x_a))],
                [(item['end_pos'][0] - (first_width * dis_y_a)), (item['end_pos'][1] + (second_width * dis_x_a))]
            ]
            edge_2 = [
                [(item['end_pos'][0] - (first_width * dis_y_a)), (item['end_pos'][1] + (second_width * dis_x_a))],
                [(item['end_pos'][0] + (second_width * dis_y_a)), (item['end_pos'][1] - (first_width * dis_x_a))]
            ]
            edge_3 = [
                [(item['end_pos'][0] + (second_width * dis_y_a)), (item['end_pos'][1] - (first_width * dis_x_a))],
                [(item['start_pos'][0] + (second_width * dis_y_a)), (item['start_pos'][1] - (first_width * dis_x_a))]
            ]
            edge_4 = [
                [(item['start_pos'][0] + (second_width * dis_y_a)), (item['start_pos'][1] - (first_width * dis_x_a))],
                [(item['start_pos'][0] - (first_width * dis_y_a)), (item['start_pos'][1] + (second_width * dis_x_a))]
            ]

            self.wall_segments.append(edge_1)
            self.wall_segments.append(edge_2)
            self.wall_segments.append(edge_3)
            self.wall_segments.append(edge_4)