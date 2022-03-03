from spike_analysis import *

def run():
    t, raw = load_data(data_file='continuous.dat', channels=32)
    x = filter_data(raw)
    t_tot = total_recording_time(t)
    minutes = t_tot.seconds//60

    for i in range(0, minutes):
        a = int(i*60)
        b = int((i+1)*60)
        print(a)
        plot_channels(t, raw, 'd3_m3_d19iv_lfp_'+str(a)+'_to_'+str(b)+'s', title='Day 3 in Media 3, Day 19 IV: LFP (-125\u03BCV to 125\u03BCV)', t_i=a, t_f=b)
        plot_channels(t, x, 'd3_m3_d19iv_filtered_'+str(a)+'_to_'+str(b)+'s', title='Day 3 in Media 3, Day 19 IV: Filtered Neural Recording (-125\u03BCV to 125\u03BCV)', t_i=a, t_f=b)
        spike_data = get_spikes(x, 4.25, t_i=a, t_f=b)
        plot_channels(t, x, 'd3_m3_d19iv_spike_detection_'+str(a)+'_to_'+str(b)+'s', title='Day 3 in Media 3, Day 19 IV: Spike Detection (-125\u03BCV to 125\u03BCV)', k=4, raster_on=True, spikes=spike_data, t_i=a, t_f=b)
        rasterize(spike_data,'d3_m3_d19iv_raster_'+str(a)+'_to_'+str(b)+'s', title='Day 3 in Media 3, Day 19 IV: Raster Plot')
        max_rate, mean_rate, rate_variance = get_spike_stats(spike_data, t_i=a, t_f=b)
        with open('spike_stats.txt', 'a') as f:
            f.write('['+str(max_rate[0])+', '+str(mean_rate)+', '+str(rate_variance)+'], ')

if __name__ == '__main__':
    run()