import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import matplotlib.widgets

fig, ax = plt.subplots()
fig.set_figwidth(9)
fig.set_figheight(6)
x, y = [0], [0]
x_width = 2 * np.pi

class Animator(FuncAnimation):
    def __init__(self, fig, func, frames=None, interval=25, init_func=None, fargs=None, save_count=None, **kwargs):
        self.i = 0
        self.min = 0
        self.max = 100
        self.runs = True 
        self.forwards = True
        self.fig = fig 
        self.func = func
        self.ampl = 1
        self.setup()
        FuncAnimation.__init__(
            self, self.fig, self.update, frames=self.play(), interval=interval, init_func=init_func, fargs=fargs, save_count=save_count, **kwargs)
        
    def setup(self):
        playerax = self.fig.add_axes([0.01, 0.83, 0.25, 0.05])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        pp = divider.append_axes('right', size='100%', pad=0.05)
        st = divider.append_axes('right', size='100%', pad=0.05)
        
        self.button_pause = matplotlib.widgets.Button(playerax, label='Stop')
        self.button_step = matplotlib.widgets.Button(pp, label='Step')
        self.button_reset = matplotlib.widgets.Button(st, label='Reset')

        self.button_pause.on_clicked(self.pause)
        self.button_step.on_clicked(self.onestep)
        self.button_reset.on_clicked(self.reset)
        
        variableax = self.fig.add_axes([0.1, 0.75, 0.16, 0.05])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(variableax)
        
        # t1 = divider.append_axes('bottom', size='100%', pad=0.05)
        self.input_a = matplotlib.widgets.TextBox(variableax, label='Amplitude:')
        self.input_a.on_submit(self.ampl_change)
        self.input_a.set_val("1")
        
    def play(self):
        while self.runs:
            self.i = self.i + self.forwards - (not self.forwards)
            if self.i > self.min and self.i < self.max:
                yield self.i
            else:
                self.stop()
                yield self.i

    def pause(self, event=None):
        if self.runs:
            self.button_pause.label.set_text('Start')
            self.runs = False
            self.event_source.stop()
        else:
            self.button_pause.label.set_text('Stop')
            self.runs = True 
            self.event_source.start()

    def stop(self, event=None):
        self.runs = False
        
    def onestep(self, event=None):
        self.i = self.i+self.forwards-(not self.forwards)
        self.update(self.i)
        self.fig.canvas.draw_idle()
        
    def reset(self, event=None):
        global x, y, ax 
        x = [0]
        y = [0]
        ax.set_xlim(0, x_width)
        
    def ampl_change(self, event=None):
        self.ampl = int(self.input_a.text)
        # self.reset()
        
    def update(self, i):
        self.func(self.i, self.ampl)

plot, = ax.plot([], [], color='b')
fig.subplots_adjust(left=0.33)
ax.set_xlim(0, x_width)
ax.set_ylim(-1, 1)

curr_ampl = 1

def step(i, ampl):
    global curr_ampl
    
    x.append(x[-1] + 0.1)
    y.append(np.sin(x[-1]) * ampl)
    
    if ax.get_xlim()[1] < x[-1] + 0.5:
        ax.set_xlim(x[-1] - x_width, x[-1] + 0.5)
    
    if ampl != curr_ampl:
        ax.set_ylim(-ampl, ampl)
        curr_ampl = ampl
    
    plot.set_data(x, y)
    
ani = Animator(fig, step, interval=50)
plt.show()
    
    