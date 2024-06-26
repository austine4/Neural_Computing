import pyabf
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from scipy.interpolate import CubicSpline
from scipy.signal import decimate, find_peaks
from sklearn.cluster import KMeans
from openTSNE import TSNE

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

class Visualizer:
    def __init__(self):
        pass

    def multi_channel_plot(self, data, exclude=True, channels=[], spikes=None, use_absolute_time=True, stimulation_times=None):
        all_channels = list(data.keys())
        if exclude:
            channels_to_plot = [ch for ch in all_channels if ch not in channels]
        else:
            channels_to_plot = channels

        plt.figure(figsize=(10, len(channels_to_plot) * 2))
        for i, channel in enumerate(channels_to_plot):
            time = data[channel]['timestamps'] if use_absolute_time else np.arange(len(data[channel]['values'])) * data[channel]['sampling_rate']
            plt.subplot(len(channels_to_plot), 1, i + 1)
            plt.plot(time, data[channel]['values'], label=f'Channel {channel}')
            if spikes is not None and channel in spikes:
                for spike in spikes[channel]:
                    if use_absolute_time:
                        plt.axvline(x=spike, color='r', linestyle='--', linewidth=0.5)
                    else:
                        spike_idx = np.where(data[channel]['timestamps'] == spike)[0][0]
                        plt.axvline(x=spike_idx*data[channel]['sampling_rate'], color='r', linestyle='--', linewidth=0.5)
            if stimulation_times is not None:
                for stim in stimulation_times:
                    if((stim[0] < data[channel]['timestamps'][-1] or stim[1] < data[channel]['timestamps'][-1]) and (stim[0] > data[channel]['timestamps'][0] or stim[1] > data[channel]['timestamps'][0])):
                        if use_absolute_time:
                            plt.axvspan(stim[0], stim[1], color='k', alpha=0.5)
                        else:
                            start = np.where(data[channel]['timestamps'] == stim[0])[0][0]*data[channel]['sampling_rate']
                            end = np.where(data[channel]['timestamps'] == stim[1])[0][0]*data[channel]['sampling_rate']
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
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.title(f'PCA of Spikes for Channel {channel}')
            else:
                plt.xlabel('PC1')
                plt.ylabel('PC2')
                plt.title(f'PCA of Spikes for Channel {channel}')
        plt.tight_layout()
        plt.show()
    
    
    #pca histogram
        

#connected information
#visualization of plots

#firing frequency vs stimulation amplitude (200ms after stimulation)

#plot spike frequency vs time after stimulation; estimate interelectrode delay