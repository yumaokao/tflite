#!/usr/bin/env python

# import sys
import argparse
import urwid
import numpy as np

VERSION = "0.3.0"


def exit_program():
    raise urwid.ExitMainLoop()


# GraphView: contains toolbox to update core BarGraph
#            handles palette and key inputs
class GraphView(urwid.WidgetWrap):
    """
    A class responsible for providing the application's interface and
    graph display.
    """
    palette = [
        ('body',         'black',      'light gray', 'standout'),
        ('header',       'white',      'dark red',   'bold'),
        ('screen edge',  'light blue', 'dark cyan'),
        ('main shadow',  'dark gray',  'black'),
        ('line',         'black',      'light gray', 'standout'),
        ('bg background','dark gray',  'black'),
        ('bg 1',         'black',      'dark blue', 'standout'),
        ('bg 1 smooth',  'dark blue',  'black'),
        ('bg 2',         'black',      'light blue', 'standout'),
        ('bg 2 smooth',  'light blue',  'black'),
        ('button normal','light gray', 'dark blue', 'standout'),
        ('button select','white',      'dark green'),
        ('line',         'black',      'light gray', 'standout'),
        ('pg normal',    'white',      'black', 'standout'),
        ('pg complete',  'white',      'dark magenta'),
        ('pg smooth',    'dark magenta','black')
        ]
    bar_modes = [(16, 8, '16 bins, cols 8'),
                 (32, 4, '32 bins, cols 4'),
                 (64, 2, '64 bins, cols 2'),
                 (128, 1, '128 bins, cols 1')]

    def __init__(self, anpy=None, amin=None, amax=None):
        urwid.WidgetWrap.__init__(self, self.main_window())
        # default value
        self.set_bar_mode(0)
        self.graph_num_hlines = 4
        self.tensor = None
        self.min = 0.0
        self.max = 0.0
        # default value
        if anpy is None:
            anpy = self.random_tensor()
        if amin is None:
            amin = anpy.min()
        if amax is None:
            amax = anpy.max()
        # self.set_tensor_with_minmax(anpy, amin, amax)
        self.w_bar_buttons[0].set_state(True)
        self.set_tensor_with_minmax(anpy, amin, amax)

    @staticmethod
    def _unhandled_keys(key):
        if key == 'Q':
            exit_program()

    def exit_program(self, w):
        raise urwid.ExitMainLoop()

    def set_bar_mode(self, mode):
        self.bar_mode = mode
        self.graph_num_bars = self.bar_modes[self.bar_mode][0]
        self.graph_bar_width = self.bar_modes[self.bar_mode][1]

    def set_tensor_with_minmax(self, tensor=None, tmin=None, tmax=None):
        if tensor is not None:
            self.tensor = tensor
        if tmin is not None:
            self.min = tmin
        if tmax is not None:
            self.max = tmax

        # if no tensor, do nothing
        if self.tensor is None:
            return
        # if no min/max, uses self.tensor's
        if self.min is None:
            self.min = self.tensor.min()
        if self.max is None:
            self.max = self.tensor.max()
        # sys.stderr.write('set_tensor_with_minmax: {}\n'.format(self.graph_num_bars))
        # sys.stderr.flush()
        xscale = (self.max - self.min) / self.graph_num_bars
        xzero = int(abs(self.min) / xscale)
        bins = [self.min + b * xscale for b in range(self.graph_num_bars + 1)]
        hist, bins = np.histogram(self.tensor, bins=bins)
        ymax = hist.max()
        ymin = hist.min()
        yscale = (ymax - ymin) / self.graph_num_hlines
        ylines = [ymin + l * yscale for l in range(self.graph_num_hlines + 1)]
        ylines.reverse()

        # prepare data
        l = []
        for b, n in enumerate(hist.tolist()):
            l.append([0, n] if b < xzero else [n, 0])
        self.w_graph.set_data(l, ymax + yscale / 2, ylines)
        self.w_graph.set_bar_width(self.graph_bar_width)

        # update w_scale
        scls = list(map(lambda s: (s, '{:>8.0f}'.format(s)), ylines))
        self.w_scale.set_scale(scls, ymax + yscale / 2)
        self.w_scale._invalidate()

        # update w_min, w_max
        self.w_min.set_text('{:.2f}'.format(self.min))
        self.w_max.set_text('{:.2f}'.format(self.max))

        # update w_qinfo
        qinfostr = 'uint8: zero={:.0f} scale={:.2f}, '
        qinfostr += 'x-axis: zero={:.0f} scale={:.2f}'
        u8s = (self.max - self.min) / 256.0
        u8z = int(abs(self.min) / u8s)
        qinfo = qinfostr.format(u8z, u8s, xzero, xscale)
        self.w_qinfo.set_text(qinfo)
        return True

    def random_tensor(self):
        # anpy = np.random.rand(1, 128, 128, 128)
        anpy = np.random.normal(size=(1, 128, 128, 128))
        sf = np.random.randint(1, 1024)
        zp = np.random.randint(-512, 512)
        anpy = anpy * sf - zp
        return anpy

    def on_random_button(self, button):
        anpy = self.random_tensor()
        self.set_tensor_with_minmax(anpy, anpy.min(), anpy.max())

    def prepare_on_bar_mode_change(self):
        def on_bar_mode_change(rb, state, userdata):
            # sys.stderr.write('on_bar_mode_change: {} state {}\n'.format(userdata, state))
            # sys.stderr.flush()
            if state:
                self.set_bar_mode(userdata)
                anpy = self.tensor
                if anpy is not None:
                    self.set_tensor_with_minmax(anpy, anpy.min(), anpy.max())
        return on_bar_mode_change

    # widgets
    def bar_graph(self, smooth=False):
        satt = {(1,0): 'bg 1 smooth', (2,0): 'bg 2 smooth'}
        w = urwid.BarGraph(['bg background','bg 1','bg 2'], satt=satt)
        return w

    def button(self, t, fn):
        w = urwid.Button(t, fn)
        w = urwid.AttrWrap(w, 'button normal', 'button select')
        return w

    def radio_button(self, g, l, fn, data):
        w = urwid.RadioButton(g, l, False, on_state_change=fn, user_data=data)
        w = urwid.AttrWrap(w, 'button normal', 'button select')
        return w

    def graph_controls(self):
        self.w_random_button = self.button('Normal', self.on_random_button)
        self.w_bar_buttons = []
        group = []
        for i, m in enumerate(self.bar_modes):
            rb = self.radio_button(group, m[2],
                                   self.prepare_on_bar_mode_change(), i)
            self.w_bar_buttons.append(rb)

        l = [urwid.Text('Random Generator', align='center'),
             urwid.Columns([('fixed', 10, self.w_random_button)]),
             urwid.Divider(),
             urwid.Text('Display Modes', align='center'),
            ]
        l.extend(self.w_bar_buttons)
        w = urwid.ListBox(urwid.SimpleListWalker(l))
        w = urwid.Padding(w, left=2, right=2)
        return w

    def main_window(self):
        self.w_graph = self.bar_graph()
        self.w_scale = urwid.GraphVScale([(8.0, '8.0'), (4.0, '4.0')], 12.0)
        self.w_min = urwid.Text('0.0', align='left')
        self.w_qinfo = urwid.Text('zero=0, scale=2.0', align='center')
        self.w_max = urwid.Text('32.0', align='right')

        footer = urwid.Columns([
                self.w_min,
                ('fixed', 108, self.w_qinfo),
                self.w_max
            ])
        pile0 = urwid.Pile([self.w_graph, ('pack', footer)])
        pile1 = urwid.Pile([self.w_scale, ('pack', urwid.Text('chan[:]'))])

        controls = self.graph_controls()
        cols = urwid.Columns([
                ('fixed', 8, pile1),
                ('fixed', 128, pile0),
                controls,
            ], dividechars=1, focus_column=2)
        self.top = urwid.Frame(cols)
        return self.top


class GraphController:
    """
    A class responsible for setting up the model and view and running
    the application.
    """
    def __init__(self, anpy=None, amin=None, amax=None):
        kwargs = {'anpy': anpy, 'amin': amin, 'amax': amax}
        self.view = GraphView(**kwargs)

    def main(self):
        self.loop = urwid.MainLoop(self.view, self.view.palette,
                                   unhandled_input=GraphView._unhandled_keys)
        self.loop.run()


def main():
    parser = argparse.ArgumentParser(description='numpy array histogram visualizer')
    parser.add_argument('npy', nargs='?', default=None,
            help='specify numpy array file')
    parser.add_argument('--min', nargs='?', default=None, type=float,
            help='specify min value for the numpy array to histogram')
    parser.add_argument('--max', nargs='?', default=None, type=float,
            help='specify max value for the numpy array to histogram')
    parser.add_argument('-v', '--version', action='version',
            version=VERSION, help='show version infomation')
    args = parser.parse_args()

    anpy = np.load(args.npy) if args.npy is not None else None
    kwargs = {'anpy': anpy, 'amin': args.min, 'amax': args.max}
    GraphController(**kwargs).main()


if '__main__'==__name__:
    main()
