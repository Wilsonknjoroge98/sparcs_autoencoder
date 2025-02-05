from pydub import AudioSegment
from pydub.playback import play
import asyncio
import numpy as np
import logging
from tqdm import tqdm
import threading
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
from scipy.stats import skew, kurtosis
from scipy.signal import welch, butter, filtfilt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from sklearn.preprocessing import MinMaxScaler

class Utils:
        def __init__(self, autoencoder):
                self.autoencoder = autoencoder

        async def filter_samples(self, iq_samples) -> np.array:
                        """
                        Pass raw iq samples through a number of filters, normalizers, and frequency aligners 
                        and transform the result into a N x 128 array for batch processing

                        :param iq_samples:
                        IQ samples read from the SDR data stream.
                        :param center_freq:
                        Float-representation of center frequency currently being observed by the SDR sourcing the IQ signal, in megahertz
                        :param soi_start_freq:
                        Float of the spectrum-of-interest start frequency, in MHz. This is the visible window of the user's spectrogram.
                        :param soi_end_freq:
                        Float of the spectrum-of-interest end frequency, in MHz. This is the end-freq of the visible window of the user's spectrogram.
                        """

                        center_freq=446000000
                        soi_start_freq=445000000
                        soi_end_freq=447000000

                        # how far off the user's current spectrogram viewport is removed from the SDR's center frequency being sampled.
                        frequency_shift_hz = ((soi_start_freq + soi_end_freq) / 2) - center_freq
                        # how fast the SDR is sampling IQ data
                        sample_rate_hz = (soi_end_freq * 2) + 1
                        # set up filter coefficients for bandpass filter
                        # b, a = butter(4, Wn=[445000000, 446999999], fs=sample_rate_hz, btype='bandpass')
                        
                        processed_data = []
                        # loop through segments of iq samples

                        logger.info("Filtering IQ samples")
                        for sample in tqdm(iq_samples):
                                # remove dc offset i.e. center the signal (sample data type)
                                data = sample - np.mean(np.abs(sample))

                                # energy normalization
                                # data = data / np.sqrt(np.mean(np.abs(data)**2))

                                # array of sample timestamps
                                t = np.arange(len(data)) / sample_rate_hz


                                # adjust array shapes to match
                                data, t = self._adjust_array_shapes(data, t)


                                # shift the signal frequency by frequency_shift_hz
                                shifted_data = data * np.exp(1j*2*np.pi*frequency_shift_hz*t)
                        
                                # filters out frequencies outside the spectrum of interest.
                                # filt_data = filtfilt(b, a, shifted_data)


                                processed_data.append(shifted_data)
        
                        # stack the processed data into a single array
                        flattened_data = np.hstack(processed_data)

                        window_size = 128000 
                        step_size = 64000 

                        # use a sliding window with 50% overlap to create a 2D array of windows
                        windows = np.array([flattened_data[i:i+window_size] for i in range(0, len(flattened_data)-window_size, step_size)])

                        # reshape to (n_windows, 128000)
                        return windows.reshape(-1, 128000)
        
        def _adjust_array_shapes(self, arr1: np.array, arr2: np.array) -> tuple[np.array]:
                """
                Adjust the shapes of two arrays to match, truncating longer arrays to match shorter array dimensions.

                :param arr1:
                array
                :param arr2:
                array
                """
                # check to see if the dimensions are different
                if arr1.shape != arr2.shape:
                        # e.g (5, 8)
                        min_shape = np.minimum(arr1.shape, arr2.shape) # find combined minimum array shape, as a tuple of x- and y- dimensions.
                        # slice arrays to match smallest shape e.g arr[slice(0, 3), slice(0, 3)]
                        arr1 = arr1[tuple(slice(0, s) for s in min_shape)] 
                        arr2 = arr2[tuple(slice(0, s) for s in min_shape)] 
                return arr1, arr2
        
        def extract_features(self, filtered_data: np.array, sample_rate_hz: float):
                """
                Extract real-valued features from a 1D complex-valued IQ sample array.

                :param filtered_data: Filtered IQ samples (complex-valued NumPy array)
                :param sample_rate_hz: SDR sampling rate in Hz
                :return: 1D NumPy array of extracted features
                """
             
                if not np.iscomplexobj(filtered_data):
                        raise ValueError("Input data must be complex-valued (IQ samples).")

                
                magnitude = np.abs(filtered_data)  
                phase = np.angle(filtered_data) 

               
                feature_vector = [
                        # avg magnitude
                        np.mean(magnitude),
                        # variation in magnitude  
                        np.std(magnitude),
                        # more spikes or more dips? 
                        skew(magnitude),  
                        # how spikey
                        kurtosis(magnitude),  
                        np.var(phase),  
                        np.mean(np.gradient(phase)), 
                        # biggest difference between two samples 
                        np.ptp(magnitude),  
                        # biggest difference between two samples
                        np.ptp(phase),  
                ]

             
                freqs, psd = welch(magnitude, fs=sample_rate_hz, nperseg=128)
                spectral_entropy = -np.sum(psd * np.log(psd + 1e-12))  

                feature_vector.append(spectral_entropy)

                return np.array(feature_vector)
        
        async def send_sdr_command(self, stream, command, value):
                '''
                https://k3xec.com/rtl-tnp/
                send a request to the rtl_tnp server in order to adjust the settings of the remote device.

                :param command: A byte string denoted by b'' (e.g. b'0x01'). Each byte command maps to a unique definition.
                :param value: np unsigned integer represents the argument for the command (e.g. the rtl-tnp set frequency '0x01' command would require a argument frequency)
                '''
                try:
                        command_message = command + int(value).to_bytes(4, byteorder='big')
                
                        stream.write(command_message)
                        # wait until it is appropriate to resume writing to the stream
                        await stream.drain()
                except Exception as e:
                        error_message = f"Error in send_command: {str(e)}"
                        print(error_message)

        def get_reconstruction_error(self, data):
                reconstructed = self.autoencoder.predict(data)
                return np.mean(np.abs(data - reconstructed), axis=(1, 2))
        
        def reshape_features(self, features_arr: np.array):
                num_full_chunks = features_arr.shape[0] // 10

                data_trimmed = features_arr[:num_full_chunks * 10]

                return data_trimmed.reshape(num_full_chunks, 10, 9)
                
        def play_sound(self, sound_file):
                sound = AudioSegment.from_file(sound_file)
                play(sound)




class Test:
        def __init__(self, sdr_ip, sdr_port, sdr_freq, sdr_sample_rate, sdr_gain, num_samples, autoencoder):
                self.sdr_ip = sdr_ip
                self.sdr_port = sdr_port
                self.sdr_freq = sdr_freq
                self.sdr_sample_rate = sdr_sample_rate
                self.sdr_gain = sdr_gain
                self.num_samples = num_samples
                self.stream = None
                self.utils = Utils(autoencoder=autoencoder)    
        async def stream_samples(self):
                """
                Open an SDR device from IP and port numbers. While a "streaming task" is active, collect IQ data
                from raw uint8 data from the SDR data socket stream.

                For each sample read in, pass data to the "initial recorder" and "monitor recorder" based on whether
                the recorders are "active"

                Blocks returning until "streaming task" is cancelled / concluded.
                """
                try:
                        reader, writer = await asyncio.open_connection(self.sdr_ip, self.sdr_port)
                        self.stream = writer

                        # number of cycles per second measured in Hz
                        await self.utils.send_sdr_command(self.stream, b'\x01', np.uint32(self.sdr_freq))
                        # number of samples read and imaginary per second (must be at least 2x frequency), this was f'd up
                        await self.utils.send_sdr_command(self.stream, b'\x02', np.uint32(self.sdr_sample_rate))
                        # manual gain control 
                        await self.utils.send_sdr_command(self.stream, b'\x03', np.uint32(1))
                        # gain in tenths of a dB
                        await self.utils.send_sdr_command(self.stream, b'\x04', np.uint32(self.sdr_gain))

                      
        
                        logger.info("Starting to stream IQ samples")
                        await asyncio.sleep(2)
                        iq_samples = []
                        while len(iq_samples) < self.num_samples:
                                # read raw data from sdr up to 1024 bytes
                                data = await reader.read(1024)

                                if not data:
                                        logger.warning("No data received. Check device and try again.")
                                        break

                                if len(data) != 1024:
                                        logger.warning(f"Received malformed sample of length: {len(data)}")
                                        continue 

                                # interpret a buffer as a 1-dimensional array
                                raw_data = np.frombuffer(data, dtype=np.uint8)

                                print(np.mean(raw_data))
                                
                                iq_samples.append(raw_data)
                                
                        
                        self.stream.close()
                        await writer.wait_closed()
                        print(iq_samples)
                        return iq_samples
                except Exception as e:
                        error_message = f"Error while streaming from SDR: {str(e)}"
                        print(error_message)

        async def detect(self):         
                iq_samples = await self.stream_samples()

                # shape (n_samples, 128000)
                # 128000 is an arbitraty batch size number for processing
                # each batch represents a collection of filtered IQ data

                filtered_samples = await self.utils.filter_samples(iq_samples=iq_samples)

                logger.info("Extracting features from IQ samples")

                feature_list = []
                # extract 9 features from each batch of filtered IQ data
                for filt_data in tqdm(filtered_samples): 
                        features = self.utils.extract_features(filtered_data=filt_data, sample_rate_hz=2048000)
                        feature_list.append(features)

                # shape (-1, 9)
                feature_arr = np.array(feature_list)


                logger.info("Features extracted from IQ samples")

                # shape (num_samples, sequence_length, num_features)
                train_data = self.utils.reshape_features(feature_arr)

                errors = self.utils.get_reconstruction_error(train_data)

                print('mean', np.mean(errors))
                print('std', np.std(errors))
                print('max', np.max(errors))
                print('min', np.min(errors))
        async def train(self):
                iq_samples = await self.stream_samples()

                # shape (n_samples, 128000)
                # 128000 is an arbitraty batch size number for processing
                # each batch represents a collection of filtered IQ data
                print(iq_samples)
                filtered_samples = await self.utils.filter_samples(iq_samples=iq_samples)

                logger.info("Extracting features from IQ samples")
                
                feature_list = []
                # extract 9 features from each batch of filtered IQ data
                for filt_data in tqdm(filtered_samples):  
                        features = self.utils.extract_features(filtered_data=filt_data, sample_rate_hz=2048000)
                        feature_list.append(features)
                
                # shape (-1, 9)
                feature_arr = np.array(feature_list)

                scaler = MinMaxScaler()
                scaled_feature_arr = scaler.fit_transform(feature_arr)

                logger.info("Features extracted from IQ samples")

                # shape (num_samples, sequence_length, num_features)
                train_data = self.utils.reshape_features(scaled_feature_arr)


                print('train data shape', train_data.shape)

                # to what extent are feature arrays batched
                sequence_length = 10

                # number of features in each component
                num_features = 9

                latent_dim = 5

                inputs = Input(shape=(sequence_length, num_features))
                encoded = LSTM(latent_dim, activation='relu', return_sequences=False)(inputs)
                decoded = RepeatVector(sequence_length)(encoded)
                decoded = LSTM(num_features, activation='linear', return_sequences=True)(decoded)

                autoencoder = Model(inputs, decoded)
                autoencoder.compile(optimizer='adam', loss='mse')

                # autoencoder = load_model('autoencoder.keras')

                autoencoder.fit(train_data, train_data, epochs=50, batch_size=32, validation_split=0.1)

                autoencoder.save('autoencoder_light.keras')


class Stream:
        def __init__(self, sdr_ip, sdr_port, sdr_freq, sdr_sample_rate, sdr_gain, num_samples, autoencoder, iq_queue, stop_event):
            self.sdr_ip = sdr_ip
            self.sdr_port = sdr_port
            self.sdr_freq = sdr_freq
            self.sdr_sample_rate = sdr_sample_rate
            self.sdr_gain = sdr_gain
            self.num_samples = num_samples
            self.stream = None
            self.utils = Utils(autoencoder=autoencoder)
            self.iq_queue = iq_queue
            self.stop_event = stop_event


        async def stream_samples(self):
                """
                Open an SDR device from IP and port numbers. While a "streaming task" is active, collect IQ data
                from raw uint8 data from the SDR data socket stream.

                For each sample read in, pass data to the "initial recorder" and "monitor recorder" based on whether
                the recorders are "active"

                Blocks returning until "streaming task" is cancelled / concluded.
                """
                try:
                         
                        reader, writer = await asyncio.open_connection(self.sdr_ip, self.sdr_port)
                        self.stream = writer

                        # number of cycles per second measured in Hz
                        await self.utils.send_sdr_command(self.stream, b'\x01', np.uint32(self.sdr_freq))
                        # number of samples read and imaginary per second (must be at least 2x frequency), this was f'd up
                        await self.utils.send_sdr_command(self.stream, b'\x02', np.uint32(self.sdr_sample_rate))
                        # manual gain control 
                        await self.utils.send_sdr_command(self.stream, b'\x03', np.uint32(1))
                        # gain in tenths of a dB
                        await self.utils.send_sdr_command(self.stream, b'\x04', np.uint32(self.sdr_gain))

                        logger.info("Starting to stream IQ samples")
                        await asyncio.sleep(2)
                        while not self.stop_event.is_set():
                                iq_samples = []
                                while len(iq_samples) < self.num_samples:
                                        # read raw data from sdr up to 1024 bytes
                                        data = await reader.read(1024)

                                        if not data:
                                                logger.warning("No data received. Check device and try again.")
                                                break

                                        if len(data) != 1024:
                                                # logger.warning(f"Received malformed sample of length: {len(data)}")
                                                continue 

                                        # interpret a buffer as a 1-dimensional array
                                        raw_data = np.frombuffer(data, dtype=np.uint8)
                                        
                                        iq_samples.append(raw_data)

                                        if len(iq_samples) == self.num_samples:
                                                await self.iq_queue.put(iq_samples)
                                        # print('queue size', self.iq_queue.qsize())
                except Exception as e:
                        error_message = f"Error while streaming from SDR: {str(e)}"
                        print(error_message)

        async def detect(self):
                while not self.stop_event.is_set():
                        try:           
                                # iq_samples = await self.stream_samples()
                                iq_samples = await self.iq_queue.get()

                                # shape (n_samples, 128000)
                                # 128000 is an arbitraty batch size number for processing
                                # each batch represents a collection of filtered IQ data

                                filtered_samples = await self.utils.filter_samples(iq_samples=iq_samples)

                                logger.info("Extracting features from IQ samples")

                                feature_list = []
                                # extract 9 features from each batch of filtered IQ data
                                for filt_data in tqdm(filtered_samples): 
                                        features = self.utils.extract_features(filtered_data=filt_data, sample_rate_hz=2048000)
                                        feature_list.append(features)

                                # shape (-1, 9)
                                feature_arr = np.array(feature_list)


                                logger.info("Features extracted from IQ samples")

                                # shape (num_samples, sequence_length, num_features)
                                train_data = self.utils.reshape_features(feature_arr)

                                errors = self.utils.get_reconstruction_error(train_data)

                                print('mean', np.mean(errors))
                                print('std', np.std(errors))
                                print('max', np.max(errors))
                                print('min', np.min(errors))

                                # if (np.mean(errors) > 60000):
                                #         threading.Thread(target=self.utils.play_sound, args=('beep.mp3',)).start()
                                #         threading.Thread(target=self.utils.play_sound, args=('beep.mp3',)).start()
                                #         threading.Thread(target=self.utils.play_sound, args=('beep.mp3',)).start()

                                
                        # return anomalies
                        except asyncio.TimeoutError:
                                continue

        async def train(self):
                while not self.stop_event.is_set():
                        try: 
                                iq_samples = await self.iq_queue.get()

                                # shape (n_samples, 128000)
                                # 128000 is an arbitraty batch size number for processing
                                # each batch represents a collection of filtered IQ data
                                filtered_samples = await self.utils.filter_samples(iq_samples=iq_samples)

                                
                                logger.info("Extracting features from IQ samples")
                                
                                feature_list = []
                                # extract 9 features from each batch of filtered IQ data
                                for filt_data in tqdm(filtered_samples):  
                                        features = self.utils.extract_features(filtered_data=filt_data, sample_rate_hz=2048000)
                                        feature_list.append(features)
                                
                                # shape (-1, 9)
                                feature_arr = np.array(feature_list)

                                scaler = MinMaxScaler()
                                scaled_feature_arr = scaler.fit_transform(feature_arr)

                                logger.info("Features extracted from IQ samples")

                                # shape (num_samples, sequence_length, num_features)
                                train_data = self.utils.reshape_features(scaled_feature_arr)


                                # print('train data shape', train_data.shape)

                                # # to what extent are feature arrays batched
                                # sequence_length = 10

                                # # number of features in each component
                                # num_features = 9

                                # latent_dim = 5

                                # inputs = Input(shape=(sequence_length, num_features))
                                # encoded = LSTM(latent_dim, activation='relu', return_sequences=False)(inputs)
                                # decoded = RepeatVector(sequence_length)(encoded)
                                # decoded = LSTM(num_features, activation='linear', return_sequences=True)(decoded)

                                # autoencoder = Model(inputs, decoded)
                                # autoencoder.compile(optimizer='adam', loss='mse')

                                self.utils.autoencoder.fit(train_data, train_data, epochs=50, batch_size=32, validation_split=0.1)

                                autoencoder.save('autoencoder_light.keras')
                        except asyncio.TimeoutError:
                                continue
                


if __name__ == "__main__":
        async def main():
                logger.info('loading model')
                autoencoder = load_model('autoencoder.keras')

                iq_queue = asyncio.Queue(maxsize=100000) 
                stop_event = asyncio.Event()

                stream = Stream(sdr_ip='192.168.3.157', sdr_port=1234, sdr_freq=446000000, sdr_sample_rate=2048000, sdr_gain=10, num_samples=20000, autoencoder=autoencoder, iq_queue=iq_queue, stop_event=stop_event)
                # test = Test(sdr_ip='192.168.3.157', sdr_port=1234, sdr_freq=446000000, sdr_sample_rate=2048000, sdr_gain=10, num_samples=10000, autoencoder=autoencoder)
          
                producer_task = asyncio.create_task(stream.stream_samples()) 
                consumer_tasks = [asyncio.create_task(stream.train()) for _ in range(10)]
                
                await asyncio.sleep(120)

                stop_event.set()
                await producer_task
                await asyncio.gather(*consumer_tasks)
                
        asyncio.run(main())







