# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from spike_analysis import *
import matplotlib.pyplot as plt

def run():
    t, raw = load_data(data_file='continuous.dat', channels=32)
    a = 0
    b = .05
    x = filter_data(raw)
    plot_channels(t, raw, 'd3_m3_d19iv_lfp', title='Day 3 in Media 3, Day 19 IV: LFP (-125\u03BCV to 125\u03BCV)', t_i=a, t_f=b)
    plot_channels(t, x, 'd3_m3_d19iv_filtered', title='Day 3 in Media 3, Day 19 IV: Filtered Neural Recording (-125\u03BCV to 125\u03BCV)', t_i=a, t_f=b)
    spike_data = get_spikes(x, 4, t_i=a, t_f=b)
    plot_channels(t, x, 'd3_m3_d19iv_spike_detection', title='Day 3 in Media 3, Day 19 IV: Spike Detection (-125\u03BCV to 125\u03BCV)', k=4, raster_on=True, spikes=spike_data, t_i=a, t_f=b)


if __name__ == '__main__':
    run()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
