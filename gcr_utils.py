import pyabf
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from scipy.interpolate import CubicSpline
from scipy.signal import decimate, find_peaks, convolve
from sklearn.cluster import KMeans
from openTSNE import TSNE
import yaml
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from IPython.display import HTML

class DataLoader:
    def __init__(self):
        self.sampling_rate = None

    def load_data(self, file_path, exclude=True, channels=[], start_time=None, end_time=None):
        abf = pyabf.ABF(file_path)
        self.sampling_rate = abf.dataRate
        print(abf.protocol)
        #TODO: store stimulation amplitude and channel
        data = self._parse_data(abf, exclude, channels, start_time, end_time)
        return data

    def _parse_data(self, abf, exclude, channels, start_time, end_time):
        data = {}
        start_idx = 0 if start_time is None else self.get_index(start_time, abf.dataRate)
        end_idx = len(abf.sweepY) if end_time is None else self.get_index(end_time, abf.dataRate)

        timestamps = np.arange(start_idx, end_idx) / self.sampling_rate

        all_channels = list(range(abf.channelCount))
        if exclude:
            channels_to_process = [ch for ch in all_channels if ch not in channels]
        else:
            channels_to_process = channels

        for channel in channels_to_process:
            abf.setSweep(0, channel)
            data[channel] = {
                'sampling_rate': self.sampling_rate,
                'timestamps': timestamps,
                'values': abf.sweepY[start_idx:end_idx].copy()
            }
        return data

    def get_index(self, time, sampling_rate):
        # Convert times to indices
        return int(time * sampling_rate)

class Filter:
    def __init__(self):
        pass

    def stimulation_filter(self, data, stimulation_times):
        #TODO: detect stimulation -- use window and eliminate around crossing a certain threshold
        filtered_data = {}

        channels = list(data.keys())
        for channel in channels:

            for stims in stimulation_times:
                start_idx = np.where( stims[0] == data[channel]['timestamps'])[0]
                end_idx = np.where( stims[1] == data[channel]['timestamps'])[0]

                if len(start_idx) != 1 or len(end_idx) != 1:
                    continue
                else:
                    start_idx=start_idx[0]
                    end_idx=end_idx[0]

                data[channel]['values'][start_idx:end_idx] = np.zeros(end_idx - start_idx)

            filtered_data[channel] = {
                'sampling_rate': data[channel]['sampling_rate'],
                'timestamps': data[channel]['timestamps'],
                'values': data[channel]['values']
            }

        return filtered_data

    def temporal_zeroing(self, data):
        # Subtract the mean from each channel's data
        filtered_data = {}
        for channel in data:
            filtered_data[channel] = {
                'sampling_rate': data[channel]['sampling_rate'],
                'timestamps': data[channel]['timestamps'],
                'values': data[channel]['values'] - np.ones_like(data[channel]['values'])*np.mean(data[channel]['values'])
            }

        return filtered_data
    
    #this is not what we want at all lol
    #TODO: fix this
    def interchannel_zeroing(self, data):
        # Ensure that all channels have the same length of data
        lengths = [len(data[channel]['values']) for channel in data]
        if len(set(lengths)) > 1:
            raise ValueError("All channels must have the same length of data.")

        # Stack all channel data for easier mean calculation
        all_values = np.stack([data[channel]['values'] for channel in data], axis=0)

        # Calculate the mean across all channels at each time point
        mean_values = np.mean(all_values, axis=0)

        # Subtract the mean from each channel's data
        filtered_data = {}
        for channel in data:
            filtered_data[channel] = {
                'sampling_rate': data[channel]['sampling_rate'],
                'timestamps': data[channel]['timestamps'],
                'values': data[channel]['values'] - mean_values
            }

        return filtered_data
    
    def bandpass_filter(self, data, lowcut, highcut, order=5):

        filtered_data = {}
        for channel in data:
            nyquist = 0.5 * data[channel]['sampling_rate']
            low = lowcut / nyquist
            high = highcut / nyquist
            b, a = butter(order, [low, high], btype='band')
            filtered_values = filtfilt(b, a, data[channel]['values'])
            filtered_data[channel] = {
                'sampling_rate': data[channel]['sampling_rate'],
                'timestamps': data[channel]['timestamps'],
                'values': filtered_values
            }
        
        return filtered_data
    
    #TODO: channel filter to eliminate coordinated fake spikes

class SpikeDetector:
    def __init__(self):
        pass

    def threshold_detection(self, data, thresholds = [3,3], dead_time = 3, min_consecutive_time = .1, spike_time = 2):
        detected_spikes = {}
        
        for channel in data:
            min_consecutive_samples = int(min_consecutive_time * data[channel]['sampling_rate'] / 1000 )
            dead_samples = int(dead_time * data[channel]['sampling_rate'] / 1000 )  # Convert refractory period to samples
            noise_std = self.estimate_noise_std(data[channel]['values'])
            adjusted_threshold_neg = -noise_std * thresholds[0]
            adjusted_threshold_pos = noise_std * thresholds[1]
            spike_indices = np.where(np.logical_or( data[channel]['values'] <= adjusted_threshold_neg , data[channel]['values'] >= adjusted_threshold_pos))[0]
            spike_timestamps = []
            last_spike = -np.inf
            for idx in spike_indices:
                if np.abs(idx - last_spike) >= dead_samples and (np.all(data[channel]['values'][idx:idx+min_consecutive_samples] <= adjusted_threshold_neg) or np.all(data[channel]['values'][idx:idx+min_consecutive_samples] >= adjusted_threshold_pos)):
                    # Align to extreme
                    index = self.find_local_extreme(data[channel], idx, spike_time, thresholds)
                    spike_timestamps.append(data[channel]['timestamps'][index])
                    last_spike = index
            detected_spikes[channel] = np.array(spike_timestamps)

        return detected_spikes

    def estimate_noise_std(self, signal):
        # Use the median absolute deviation (MAD) to estimate noise standard deviation
        mad = np.median(np.abs(signal - np.median(signal)))
        noise_std = mad / 0.6745  # Donoho's rule
        return noise_std
    
    def find_local_extreme(self, signal, start_idx, spike_time, thresholds):
        window_size = int(spike_time * signal['sampling_rate'] / 1000)  # spike_time ms window size in samples
        start_idx = max(0, int(start_idx - window_size/2))
        end_idx = min(start_idx + window_size, len(signal['values']))
        min_id = start_idx + np.argmin(signal['values'][start_idx:end_idx])
        max_id = start_idx + np.argmax(signal['values'][start_idx:end_idx])
        min_val = signal['values'][min_id]
        max_val = signal['values'][max_id]
        if( min_val > 0 ):
            return max_id
        elif( max_val < 0):
            return min_id
        elif( np.abs(min_val)/thresholds[0] > np.abs(max_val)/thresholds[1] ):
            return min_id
        else:
            return max_id
    
class PostDetectionProcessing:
    def __init__(self):
        pass

    def statistical_elimination(self, data, spike_train, window):
        filtered_spike_train = {}

        avg_list = []
        std_list = []
        # First pass: calculate median absolute average and median standard deviation
        for channel in spike_train:
            half_window = int(window / 2 * data[channel]['sampling_rate'] / 1000)
            for spike in spike_train[channel]:
                spike_idx = np.where(data[channel]['timestamps'] == spike)[0][0]
                start_idx = max(0, spike_idx - half_window)
                end_idx = min(spike_idx + half_window, len(data[channel]['values']))
                waveform = data[channel]['values'][start_idx:end_idx]

                if len(waveform) == 0:
                    continue

                avg = np.mean(waveform)
                std = np.std(waveform)

                avg_list.append(abs(avg))
                std_list.append(std)

        median_abs_avg = np.median(avg_list)
        median_std = np.median(std_list)

        # Second pass: filter spikes based on thresholds
        for channel in spike_train:
            half_window = int(window / 2 * data[channel]['sampling_rate'] / 1000)
            filtered_spikes = []
            for spike in spike_train[channel]:
                spike_idx = np.where(data[channel]['timestamps'] == spike)[0][0]
                start_idx = max(0, spike_idx - half_window)
                end_idx = min(spike_idx + half_window, len(data[channel]['values']))
                waveform = data[channel]['values'][start_idx:end_idx]

                if len(waveform) == 0:
                    continue

                avg = np.mean(waveform)
                std = np.std(waveform)

                if abs(avg) <= 3 * median_abs_avg and std <= 4 * median_std:
                    filtered_spikes.append(spike)

            filtered_spike_train[channel] = np.array(filtered_spikes)

        return filtered_spike_train
    
    
    def multi_point_align_spikes(self, data, spike_train, window=2, interpolation_factor = 4, max_shift=10, n_peaks=3, n_bins=50):
        aligned_spike_train = {}
        max_spline_align_shift = interpolation_factor * max_shift

        for channel in spike_train:
            window_samples = int(window * data[channel]['sampling_rate'] / 1000)
            half_window = window_samples // 2
            aligned_spikes = []
            spike_indices = []
            min_indices = []
            max_indices = []
            for spike in spike_train[channel]:
                spike_idx = np.where(data[channel]['timestamps'] == spike)[0][0]
                start_idx = max(0, spike_idx - half_window)
                end_idx = min(spike_idx + half_window, len(data[channel]['values']))
                waveform = data[channel]['values'][start_idx:end_idx]

                if len(waveform) == 0:
                    continue

                # Interpolate waveform
                original_times = np.linspace(0, len(waveform) - 1, len(waveform))
                interpolation_times = np.linspace(0, len(waveform) - 1, len(waveform) * interpolation_factor)
                cs = CubicSpline(original_times, waveform)
                interpolated_waveform = cs(interpolation_times)

                spike_indices.append(spike_idx)
                min_indices.append(np.argmin(interpolated_waveform))
                max_indices.append(np.argmax(interpolated_waveform))

            min_indices = np.array(min_indices)
            max_indices = np.array(max_indices)
            spike_indices = np.array(spike_indices)
            min_values = np.array([interpolated_waveform[idx] for idx in min_indices])
            max_values = np.array([interpolated_waveform[idx] for idx in max_indices])

            ind_neg = min_values < max_values
            ind_min_neg = min_indices[ind_neg]
            ind_max_pos = max_indices[~ind_neg]

            # Histogram for negative group (min)
            hist_min, bins_min = np.histogram(ind_min_neg, bins=n_bins)
            cs_min = CubicSpline(bins_min[:-1], hist_min)
            smooth_hist_min = cs_min(np.linspace(bins_min[0], bins_min[-1], 5 * n_bins))
            peaks_min, _ = find_peaks(smooth_hist_min, distance=max_spline_align_shift // 2)
            peaks_min = np.array(peaks_min[:n_peaks])

            # Histogram for positive group (max)
            hist_max, bins_max = np.histogram(ind_max_pos, bins=n_bins)
            cs_max = CubicSpline(bins_max[:-1], hist_max)
            smooth_hist_max = cs_max(np.linspace(bins_max[0], bins_max[-1], 5 * n_bins))
            peaks_max, _ = find_peaks(smooth_hist_max, distance=max_spline_align_shift // 2)
            peaks_max = np.array(peaks_max[:n_peaks])

            time_shift = np.zeros(len(spike_indices)) + max_shift

            # Align negative group
            for i, idx in enumerate(spike_indices[ind_neg]):
                min_shift, ind_cluster = np.min(np.abs(peaks_min - ind_min_neg[i])), np.argmin(np.abs(peaks_min - ind_min_neg[i]))
                if min_shift > max_spline_align_shift:
                    continue
                n_shift = round((ind_min_neg[i] - peaks_min[ind_cluster]) / interpolation_factor)
                if n_shift > 0:
                    L = max_shift + n_shift
                    R = max_shift - n_shift
                    shifted_waveform = np.roll(data[channel]['values'][idx], n_shift)
                    time_shift[i] += n_shift / 2
                else:
                    L = max_shift - abs(n_shift)
                    R = max_shift + abs(n_shift)
                    shifted_waveform = np.roll(data[channel]['values'][idx], n_shift)
                    time_shift[i] -= n_shift / 2

                if len(shifted_waveform) >= interpolation_factor:
                    decimated_waveform = decimate(shifted_waveform, interpolation_factor, zero_phase=True)
                    aligned_spikes.append(decimated_waveform)
                else:
                    aligned_spikes.append(shifted_waveform)

            # Align positive group
            for i, idx in enumerate(spike_indices[~ind_neg]):
                min_shift, ind_cluster = np.min(np.abs(peaks_max - ind_max_pos[i])), np.argmin(np.abs(peaks_max - ind_max_pos[i]))
                if min_shift > max_spline_align_shift:
                    continue
                n_shift = round((ind_max_pos[i] - peaks_max[ind_cluster]) / interpolation_factor)
                if n_shift > 0:
                    L = max_shift + n_shift
                    R = max_shift - n_shift
                    shifted_waveform = np.roll(data[channel]['values'][idx], n_shift)
                    time_shift[i] += n_shift / 2
                else:
                    L = max_shift - abs(n_shift)
                    R = max_shift + abs(n_shift)
                    shifted_waveform = np.roll(data[channel]['values'][idx], n_shift)
                    time_shift[i] -= n_shift / 2

                if len(shifted_waveform) >= interpolation_factor:
                    decimated_waveform = decimate(shifted_waveform, interpolation_factor, zero_phase=True)
                    aligned_spikes.append(decimated_waveform)
                else:
                    aligned_spikes.append(shifted_waveform)


                decimated_waveform = decimate(shifted_waveform, interpolation_factor, zero_phase=True)
                aligned_spikes.append(decimated_waveform)

            new_spike_time = spike_indices + time_shift
            aligned_spikes = new_spike_time

            aligned_spike_train[channel] = np.array(aligned_spikes)

        return aligned_spike_train
    
    
    def align_spikes(self, data, spike_train, window=2, alignment='multi-point', interpolation_factor = 4):
        aligned_spike_train = {}

        for channel in spike_train:
            half_window = int(window / 2 * data[channel]['sampling_rate'] / 1000)
            aligned_spikes = []
            for spike in spike_train[channel]:
                spike_idx = np.where(data[channel]['timestamps'] == spike)[0][0]
                start_idx = max(0, spike_idx - half_window)
                end_idx = min(spike_idx + half_window, len(data[channel]['values']))
                waveform = data[channel]['values'][start_idx:end_idx]

                if len(waveform) == 0:
                    continue

                # Interpolate waveform
                original_times = np.linspace(0, len(waveform) - 1, len(waveform))
                interpolation_times = np.linspace(0, len(waveform) - 1, len(waveform) * interpolation_factor)
                cs = CubicSpline(original_times, waveform)
                interpolated_waveform = cs(interpolation_times)

                # Find the alignment point
                if alignment == 'min':
                    align_idx = np.argmin(interpolated_waveform)
                elif alignment == 'max':
                    align_idx = np.argmax(interpolated_waveform)
                else:
                    raise ValueError("Alignment must be either 'min' or 'max'")

                # Calculate the new spike timestamp based on alignment
                align_time_offset = (align_idx / len(interpolation_times) - 0.5) * window / 1000  # in seconds
                new_spike_time = data[channel]['timestamps'][spike_idx] + align_time_offset

                # Find the closest original timestamp
                closest_idx = np.searchsorted(data[channel]['timestamps'], new_spike_time)
                if closest_idx == len(data[channel]['timestamps']):
                    closest_idx -= 1
                aligned_spikes.append(data[channel]['timestamps'][closest_idx])

            aligned_spike_train[channel] = np.array(aligned_spikes)

        return aligned_spike_train


class SpikeSorter:
    def __init__(self):
        pass
    
    def get_waveforms(self, data, spikes, window = 2):
        waveforms={}

        channels = list(data.keys())
        for channel in channels:
            sampling_rate = data[channel]['sampling_rate']
            half_window_samples = int(window/2/1000 * sampling_rate)
            spike_times = spikes[channel]
            spike_windows = []

            for spike in spike_times:
                spike_idx = np.where(data[channel]['timestamps'] == spike)[0][0]
                if spike_idx - half_window_samples < 0 or spike_idx + half_window_samples > len(data[channel]['values']):
                    continue
                spike_window = data[channel]['values'][spike_idx - half_window_samples:spike_idx + half_window_samples+1]
                spike_windows.append(spike_window)

            waveforms[channel] = spike_windows

        return waveforms

    def apply_pca(self, waveforms, n_components=3):
        reduced_features = {}

        channels = list(waveforms.keys())
        for channel in channels:
            if waveforms[channel] is not None and len( waveforms[channel] ) > n_components:
                # Apply PCA using sklearn
                features = np.array(waveforms[channel])
                pca = PCA(n_components=n_components)
                reduced_features[channel] = pca.fit_transform(features)
            else:
                reduced_features[channel] = None

        return reduced_features
    
    def apply_tsne(self, waveforms, n_components=3):
        reduced_features = {}

        channels = list(waveforms.keys())
        for channel in channels:
            if waveforms[channel] is not None and len( waveforms[channel] ) > n_components:
                # Apply t-SNE using openTSNE
                tsne = TSNE(n_components=n_components, n_jobs=-1, random_state=42)
                reduced_features[channel] = tsne.fit(waveforms)
            else:
                reduced_features[channel] = None

        return reduced_features
    
    def apply_kmeans(self, features, n_clusters=3):
        clusters={}

        channels = list(features.keys())
        for channel in channels:
            if features[channel] is not None and len(features[channel])>n_clusters:
                # Cluster spikes using kmeans
                kmeans = KMeans(n_clusters=n_clusters)
                clusters[channel] = kmeans.fit_predict(features[channel])
            else:
                clusters[channel] = None

        return clusters

    def spike_sorting_pipeline(self, data, spikes, window = 2, n_components=3, n_clusters=3, dimensionality_reduction_method = 'pca', cluster = True):
        # Step 1: Get spike waveforms
        waveforms = self.get_waveforms(data, spikes, window)

        # Step 2: Apply dimensionality reduction
        # PCA
        if(dimensionality_reduction_method == 'pca'):
            reduced_features = self.apply_pca(waveforms, n_components)
        # t-SNE
        elif(dimensionality_reduction_method == 'tsne'):
            reduced_features = self.apply_pca(waveforms, n_components)
        else:
            print("This method is not valid -- please use \'pca\' or \'tsne\'.")
            return

        # Step 3: Apply k-means clustering
        if cluster:
            clusters=self.apply_kmeans(reduced_features, n_clusters)
        else:
            clusters=None

        return reduced_features, clusters

class SpikeStatistics:
    def __init__(self):
        pass

    def decode(self, spike_train, kernel, time_window, spike_window = 1):
        #time window should be 2d array with start and end time in [s]
        #conv kernel needs to be a valid function to convolve against binary spike train
        spike_train = self.binary_representation(spike_train, time_window)
        
        decoded_train = {}
        for channel, spikes in spike_train.items():
            total_bins = int((time_window[1] - time_window[0]) * 1000 / spike_window)

            decoded_train[channel] = {'sampling_rate':spike_window,
                                      'timestamps':np.linspace(time_window[0], time_window[1], total_bins, endpoint=False),
                                      'values':convolve(spikes, kernel, mode='same')}

        return decoded_train
    
    # Define a Gaussian kernel
    def gaussian_kernel(self, size, sigma):
        x = np.arange(-size // 2 + 1, size // 2 + 1)
        kernel = np.exp(-(x**2 / (2 * sigma**2)))
        return 1000 * kernel / kernel.sum()

    # Define an exponential kernel
    def exponential_kernel(self, size, tau):
        x = np.arange(0, size)
        kernel = np.exp(-x / tau)
        return 1000 * kernel / kernel.sum()

    # Define a square kernel
    def count_kernel(self, size):
        kernel = np.ones(size//2)
        kernel = np.concatenate((kernel, kernel[::-1]))
        return 1000 * kernel / kernel.sum()
    
    def binary_representation(self, spike_train, time_window, spike_window=1):
        # spike_window in [ms]
        # time_window in [s]
        updated_train = {}
        channels = list(spike_train.keys())
        total_bins = int( (time_window[1] - time_window[0]) * 1000 / spike_window )

        for channel in channels:
            spikes = np.zeros(total_bins)
            spike_times = spike_train[channel]

            for spike_time in spike_times:
                if time_window[0] <= spike_time < time_window[1]:
                    bin_index = int((spike_time - time_window[0]) * 1000 / spike_window)
                    spikes[bin_index] = 1

            updated_train[channel] = spikes

        return updated_train
    
    def calculate_frequency_after_stimulation(self, spike_train, time_window, stimulation_times, window_size=0.2, spike_window=1):
        binary_spike_train = self.binary_representation(spike_train, time_window, spike_window)
        frequencies = {}
        
        for channel, spikes in binary_spike_train.items():
            timestamps = np.linspace(time_window[0], time_window[1], len(spikes), endpoint=False)
            spike_count_list = []
            spike_counts_per_stim = []
            
            for stim in stimulation_times:
                start_time = stim[0]
                end_time = start_time + window_size
                
                # Find the indices for the window
                start_idx = np.searchsorted(timestamps, start_time, side='left')
                end_idx = np.searchsorted(timestamps, end_time, side='right')
                
                # Count the spikes in the window
                spike_count = np.sum(spikes[start_idx:end_idx])
                spike_counts_per_stim.append(spike_count)
                spike_count_list.append(spike_count / window_size)  # spikes per second for each stimulation

            # Calculate average frequency (spikes per second)
            average_frequency = np.mean(spike_count_list)
            frequencies[channel] = {
                'frequencies_per_stimulation': spike_count_list,
                'average_frequency': average_frequency
            }
            
        return frequencies
    
    def get_peak_times_and_differences(self, decoded_train, stimulation_times, window_size=0.2):
        peak_times = {}
        peak_differences = {}
        
        for channel, data in decoded_train.items():
            timestamps = data['timestamps']
            values = data['values']
            channel_peak_times = []
            channel_peak_differences = []
            
            for stim in stimulation_times:
                start_time = stim[1]
                end_time = start_time + window_size
                
                # Find the indices for the window
                start_idx = np.searchsorted(timestamps, start_time, side='left')
                end_idx = np.searchsorted(timestamps, end_time, side='right')
                
                # Find the time of the maximum value in the window
                if start_idx < len(values) and end_idx <= len(values):
                    window_values = values[start_idx:end_idx]
                    if len(window_values) > 0:
                        max_idx = np.argmax(window_values)
                        peak_time = np.round(timestamps[start_idx + max_idx],3)
                        channel_peak_times.append(peak_time)
                        # Calculate the difference between stim[1] and the peak time
                        peak_difference = np.round(peak_time - start_time,3)
                        channel_peak_differences.append(peak_difference)
            
            peak_times[channel] = channel_peak_times
            peak_differences[channel] = channel_peak_differences

        return peak_times, peak_differences

class Visualizer:
    def __init__(self):
        pass

    def multi_channel_plot(self, data, exclude=True, channels=[], spikes=None, use_absolute_time=True, stimulation_times=None, time_window=None):
        all_channels = list(data.keys())
        if exclude:
            channels_to_plot = [ch for ch in all_channels if ch not in channels]
        else:
            channels_to_plot = channels

        plt.figure(figsize=(10, len(channels_to_plot) * 2))
        for i, channel in enumerate(channels_to_plot):
            timestamps = data[channel]['timestamps']
            values = data[channel]['values']
            sampling_rate = data[channel]['sampling_rate']

            if time_window:
                start_idx = np.searchsorted(timestamps, time_window[0], side='left')
                end_idx = np.searchsorted(timestamps, time_window[1], side='right')
                timestamps = timestamps[start_idx:end_idx]
                values = values[start_idx:end_idx]

            time = timestamps if use_absolute_time else np.arange(len(values)) * sampling_rate
            plt.subplot(len(channels_to_plot), 1, i + 1)
            plt.plot(time, values, label=f'Channel {channel}')
            
            if spikes is not None and channel in spikes:
                for spike in spikes[channel]:
                    if time_window and (spike < time_window[0] or spike > time_window[1]):
                        continue
                    if use_absolute_time:
                        plt.axvline(x=spike, color='r', linestyle='--', linewidth=0.5)
                    else:
                        spike_idx = np.where(timestamps == spike)[0][0]
                        plt.axvline(x=spike_idx * sampling_rate, color='r', linestyle='--', linewidth=0.5)

            if stimulation_times is not None:
                for stim in stimulation_times:
                    if ((stim[0] < timestamps[-1] or stim[1] < timestamps[-1]) and (stim[0] > timestamps[0] or stim[1] > timestamps[0])):
                        if time_window and (stim[0] < time_window[0] or stim[1] > time_window[1]):
                            continue
                        if use_absolute_time:
                            plt.axvspan(stim[0], stim[1], color='k', alpha=0.5)
                        else:
                            start = np.where(timestamps == stim[0])[0][0] * sampling_rate
                            end = np.where(timestamps == stim[1])[0][0] * sampling_rate
                            plt.axvspan(start, end, color='k', alpha=0.5)
                            
            plt.xlabel('Time (s)' if use_absolute_time else 'Relative Time (s)')
            plt.ylabel('Amplitude')
            plt.title(f'Channel {channel} Data')
            plt.legend()
        plt.tight_layout()
        plt.show()


    def raster_plot(self, spikes):
        #NOTE: NOT YET TESTED
        plt.figure(figsize=(10, len(spikes) * 0.5))
        for i, (channel, spike_times) in enumerate(spikes.items()):
            plt.vlines(spike_times, i + 0.5, i + 1.5, color='black')
        plt.xlabel('Time (s)')
        plt.ylabel('Channel')
        plt.title('Raster Plot')
        plt.yticks(range(1, len(spikes) + 1), [f'Channel {ch}' for ch in spikes.keys()])
        plt.show()

    def overlay_spikes(self, data, channel, spikes, window=2, max_spikes=100):
        spikes_channel = np.random.choice(spikes[channel], max_spikes, replace=False) if len(spikes[channel]) > max_spikes else spikes[channel]

        window_samples = int(window/2 * data[channel]['sampling_rate']/1000)
        plt.figure(figsize=(4, 4))
        for spike in spikes_channel:
            spike_idx = np.where(data[channel]['timestamps'] == spike)[0][0]
            if spike_idx - window_samples < 0 or spike_idx + window_samples > len(data[channel]['values']):
                continue
            spike_window = data[channel]['values'][spike_idx - window_samples:spike_idx + window_samples+1]
            time_window = (data[channel]['timestamps'][spike_idx - window_samples:spike_idx + window_samples+1] - spike)*1000
            plt.plot(time_window, spike_window, alpha=0.5)
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.title(f'Overlaid Spikes for Channel {channel}')
        plt.show()
    
    def multi_channel_overlay_spikes(self, data, spikes, window=2, max_spikes=100):
        channels = list(data.keys())
        plt.figure(figsize=(4, len(channels) * 2))

        for i, channel in enumerate(channels):
            window_samples = int(window/2 * data[channel]['sampling_rate']/1000)
            plt.subplot(len(channels), 1, i + 1)
            spikes_channel = np.random.choice(spikes[channel], max_spikes, replace=False) if len(spikes[channel]) > max_spikes else spikes[channel]
            for spike in spikes_channel:
                spike_idx = np.where(data[channel]['timestamps'] == spike)[0][0]
                if spike_idx - window_samples < 0 or spike_idx + window_samples > len(data[channel]['values']):
                    continue
                spike_window = data[channel]['values'][spike_idx - window_samples:spike_idx + window_samples+1]
                time_window = (data[channel]['timestamps'][spike_idx - window_samples:spike_idx + window_samples+1] - spike)*1000
                plt.plot(time_window, spike_window, alpha=0.5)
            plt.xlabel('Time (ms)')
            plt.ylabel('Amplitude')
            plt.title(f'Overlaid Spikes for Channel {channel}')
        plt.tight_layout()
        plt.show()
    
    def plot_representation(self, features, clusters=None):
        #2D, 3D, clusters
        channels = list(features.keys())
        plt.figure(figsize=(10, len(channels) * 6))

        for i, channel in enumerate(channels):
            plt.subplot(len(channels), 1, i + 1)
            if features[channel] is not None:
                if clusters is not None and clusters[channel] is not None:
                    plt.scatter(features[channel][:, 0], features[channel][:, 1], c=clusters[channel], cmap='viridis', alpha=0.6)
                else:
                    plt.scatter(features[channel][:, 0], features[channel][:, 1], alpha=0.6)
                plt.xlabel('F1')
                plt.ylabel('F2')
                plt.title(f'Feature Representation of Spikes for Channel {channel}')
            else:
                plt.xlabel('F1')
                plt.ylabel('F2')
                plt.title(f'Feature Representation of Spikes for Channel {channel}')
        plt.tight_layout()
        plt.show()

    def plot_2d_electrodes(self, electrode_data):
        """
        Plot electrode positions on a 2D plane.
        
        Parameters:
        - electrode_data: Dictionary containing electrode data loaded from the YAML file.
        """
        positions = electrode_data['pos']
        size = electrode_data['size']
        
        plt.figure(figsize=(16, 12))
        ax = plt.gca()
        
        for idx, (x, y) in enumerate(positions):
            # Calculate the radius in plot units (assuming size is the diameter)
            radius = size / 2
            circle = plt.Circle((x, y), radius, edgecolor='blue', facecolor='blue', alpha=0.5)
            ax.add_patch(circle)
            plt.text(x, y, str(idx), fontsize=12, ha='center', va='center', color='white')
        
        plt.title('2D Electrode Positions')
        plt.xlabel('X position (um)')
        plt.ylabel('Y position (um)')
        plt.xlim(-size, max(pos[0] for pos in positions) + size)
        plt.ylim(-size, max(pos[1] for pos in positions) + size)
        ax.set_aspect('equal', adjustable='box')
        plt.show()
    
    def show_frequency_after_stim_2D(self, electrode_data, decoded_train, time_window, stimulation_times, stimulation_electrode):
        """
        Show a 2D animation of frequency over each electrode for a given time window.
        
        Parameters:
        - electrode_data: Dictionary containing electrode data loaded from the YAML file.
        - decoded_train: Decoded spike train data.
        - time_window: Tuple specifying the start and end times of the window (in seconds)
        """
        positions = electrode_data['pos']
        size = electrode_data['size']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        circles = []
        texts = []
        for idx, (x, y) in enumerate(positions):
            radius = size / 2
            circle = plt.Circle((x, y), radius, edgecolor='black', facecolor='white', alpha=0.5)
            ax.add_patch(circle)
            circles.append(circle)
            text = ax.text(x, y, '', fontsize=12, ha='center', va='center', color='black')
            texts.append(text)
        
        plt.title('2D Electrode Positions')
        plt.xlabel('X position (um)')
        plt.ylabel('Y position (um)')
        plt.xlim(-size, max(pos[0] for pos in positions) + size)
        plt.ylim(-size, max(pos[1] for pos in positions) + size)
        ax.set_aspect('equal', adjustable='box')
        
        start_time, end_time = time_window
        num_frames = int((end_time - start_time) * 1000)  # Number of frames for the window
        cmap = cm.get_cmap('coolwarm')

        # Calculate the maximum value across all channels for normalization
        max_value = max([max(decoded_train[channel]['values']) for channel in decoded_train])

        def update(frame):
            current_time = start_time + frame / 1000.0  # Current time in seconds

            for idx, (circle, text) in enumerate(zip(circles, texts)):
                if idx in decoded_train:
                    timestamps = decoded_train[idx]['timestamps']
                    values = decoded_train[idx]['values']
                    start_idx = np.searchsorted(timestamps, current_time, side='left')
                    if start_idx < len(values):
                        frequency = values[start_idx]  # Directly use the value
                        normalized_frequency = frequency / max_value  # Normalize the frequency
                        color = cmap(normalized_frequency)  # Get color from colormap
                        circle.set_facecolor(color)
                        text.set_text(f'{frequency:.2f}')
                elif idx == stimulation_electrode:
                    #check if it is stimulation time and make red during otherwise set as grey
                    if current_time >= stimulation_times[0] and current_time <= stimulation_times[1]:
                        circle.set_facecolor('red')
                    else:
                        circle.set_facecolor('white')
            
            ax.set_title(f'Time {current_time:.3f} s')
            return circles + texts

        anim = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

        return HTML(anim.to_jshtml())

    
    def show_frequency_after_stim_3D(self, electrode_data, decoded_train, time_window):
        #TODO: IN PROGRESS NOT WORKING YET
        """
        Show a 3D animation of frequency over each electrode for a given time window.
        
        Parameters:
        - electrode_data: Dictionary containing electrode data loaded from the YAML file.
        - decoded_train: Decoded spike train data.
        - time_window: Tuple specifying the start and end times of the window (in seconds)
        """
        positions = electrode_data['pos']
        size = electrode_data['size']
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        x = [pos[0] for pos in positions]
        y = [pos[1] for pos in positions]
        z = np.zeros(len(positions))  # Base of the bars
        dx = dy = np.ones(len(positions)) * size  # Width and depth of the bars
        dz = np.zeros(len(positions))  # Initial heights of the bars

        bars = ax.bar3d(x, y, z, dx, dy, dz, color='b', alpha=0.5)
        
        start_time, end_time = time_window
        num_frames = int((end_time - start_time) * 1000)  # Number of frames for the window
        
        def update(frame):
            current_time = start_time + frame / 1000.0  # Current time in seconds
            new_dz = np.zeros(len(positions))
            for channel in range(len(positions)):
                if channel in decoded_train:
                    timestamps = decoded_train[channel]['timestamps']
                    values = decoded_train[channel]['values']
                    start_idx = np.searchsorted(timestamps, current_time, side='left')
                    if start_idx < len(values):
                        frequency = values[start_idx]  # Directly use the value
                        new_dz[channel] = frequency
            
            ax.cla()  # Clear the axis
            ax.bar3d(x, y, z, dx, dy, new_dz, color='b', alpha=0.5)
            ax.set_title(f'Time {current_time:.3f} s')
            ax.set_xlabel('X position (um)')
            ax.set_ylabel('Y position (um)')
            ax.set_zlabel('Frequency (Hz)')
            ax.set_zlim(0, max(new_dz) * 1.2)  # Adjust z-axis limit for better visualization
            return bars

        anim = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)
        return HTML(anim.to_jshtml())
    
    def plot_peak_by_distance(self, peak_times, electrode_data, stimulation_electrode):
        print(electrode_data['pos'])
        print(peak_times)
        for idx in peak_times:
            #compute from stimulation electrode to electrode idx
            distance = np.linalg.norm( np.array( electrode_data['pos'][stimulation_electrode] ) - np.array( electrode_data['pos'][idx] ) )
            plt.scatter(distance*np.ones(len(peak_times[idx])), np.array(peak_times[idx])*1000)
        #make the plot prettier
        plt.xlabel('Distance from Stimulation Electrode (um)')
        plt.ylabel('Peak Time (ms)')
        plt.title('Peak Time by Distance from Stimulation Electrode')



# Example usage

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data



    

    
    #pca histogram
        

#connected information
#visualization of plots

#firing frequency vs stimulation amplitude (200ms after stimulation)

#plot spike frequency vs time after stimulation; estimate interelectrode delay