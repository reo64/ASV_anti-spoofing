#import wave
import numpy as np
import scipy.signal
import librosa
from scipy.fftpack import realtransforms
import matplotlib.pyplot as plt

# Linear Frequency Cepstral Coefficients(LFCCs)

#beginning of class LFCC
class LFCC(object):

#public method
    def __init__(self, y, sr):
        self.waveform = y
        self.sr = sr
        self.nfft = None
        self.window_length = None
        self.hop_length = None
    
    def get_audio_signal(self):
        return self.waveform
    
    def get_sampling_rate(self):
        return self.sr
    
    def extract_feature(self, p=0.97, window_length=20, hop_length=None, nfft=512, nchannels=20, ndim=20, delta=False):
        
        # Define a convolute preEmphasis filter
        self.waveform = self._preEmphasis(self.waveform, p)
        # NFFT (if NFFT is greater than self.frame_length fft_frame is padded with zero, truncated to NFFT if not)
        self.nfft = nfft
        # Window length in ms
        self.window_length = window_length
        # Frame length (number of points in frame)
        self.frame_length = int((self.sr * self.window_length)/1000)
        # Hop length (number of points to overlap)
        if hop_length is None:
            self.hop_length = self.frame_length//2
        else:
            self.hop_length = int((self.sr * hop_length)/1000)
        # Make a spectrogram
        spec = self._stft(wave=self.waveform, frame_length=self.frame_length, nfft=self.nfft, step=self.hop_length)
        # n: fft_bin, d: cycle time
        freq = np.fft.fftfreq(n=self.nfft//2, d=1.0/self.sr)
        
        """
        for i, sp in enumerate(spec):
            plt.plot(freq, sp)
        plt.show()
        """
        """
        import scipy.io.wavfile as wav
        import scipy.signal as signal
        import librosa

        f, t, Zxx = signal.stft(self._waveform, fs=self._sr)
        #plt.pcolormesh(t, f, , shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud')
        plt.show()

        #DB = librosa.amplitude_to_db(Zxx, ref=np.max)
        #lirosa.display.specshow(DB, sr=self._sr, hop_length=self._nfft, x_axis='time', y_axis='log')
        #plt.colorbar(format='%+2.0fdB')
        """

        #linearfilterbank
        df = self.sr / self.nfft # frequency resolution
        ###print("sampling rate: {}, freq resolution: {}".format(self.sr, df))
        filterbank = self._linearFilterBank(nchannels=nchannels)
        ###print(filterbank.shape)

        """
        for c in np.arange(0, nchannels):
            plt.plot(np.arange(0, self._nfft//2) * df, filterbank[c])
        plt.show()
        """
        eps = 2.2204e-16
        # Power spectrum
        power_spec = spec**2
        # apply linearfilterbanks for each vector, then sum all and take log
        linear_spec = np.log10(np.dot(power_spec, filterbank.T)+eps)
        ###print("linear_spec:", linear_spec.shape)
        
        # obtain a cepstrum by applying discrete cosine transform to log-linear spectrum
        cepstrum = self._dct(linear_spec).T
        
        # cepstrum = (n-dimensional feature, shift), nceps is the number of features to use
        lfcc = cepstrum[:ndim]
        
        if delta == False:
            return lffc
        
        delta = self._delta(lfcc)
        double_delta = self._delta(delta)
        
        combined = np.vstack((lfcc, delta, double_delta))
        
        return combined
    
    #pre-emphasis filter
    def _preEmphasis(self, signal, p):
        #signal := voice_signal, p := coefficient
        #make FIR filter such that coefficients are (1.0, p)
        return scipy.signal.lfilter([1.0, -p], 1.0, signal)
    
    #convert hz to mel
    def _hz2mel(self, f):
        return 1125.0 * np.log(f/700.0 + 1.0)
    
    #convert mel to hz
    def _mel2hz(self, m):
        return 700.0 * (np.exp(m/1125.0) - 1.0)
    
    #Short Time Fourier Transform: STFT
    def _stft(self, wave, frame_length, nfft, step):
        # wave: waveform (numpy.ndarray)
        # window_length: hammingWindow (numpy.ndarray)
        # nfft: number of points to perform FFT
        # nstep: overlapping length
        wave_length = wave.shape[0]
        
        # Decide a frame_length and hammingwindow
        hamming_window = np.hamming(frame_length)
        
        print("step_length:", step)
        nshift = (wave_length - frame_length + step) // step
        print("wave_length: {}, frame_length: {}, nshift: {}".format(wave_length, frame_length, nshift))
        
        #padded_wave = np.zeros(wavelen+step)
        #padded_wave = np.zeros(windowlen+(shift*step))
        #padded_wave[:wavelen] = wave # waveform has to be reformed to fit the sliding window 
        
        # X: Spectrum has to be (shift, nfft//2)
        X = np.array([]).reshape(0, nfft//2)
        for m in range(nshift):
            start = step * m
            #padded_frame = np.pad(wave[start:start+frame_length]*hamming_window, (nfft-frame_length)//2, constant_values=0)
            #print(padded_frame.shape)
            #x = np.fft.fft(padded_frame, n=nfft, norm=None)[:nfft//2]
            x = np.fft.fft(wave[start:start+frame_length]*hamming_window, n=nfft, norm=None)[:nfft//2]
            X = np.vstack((X, x))
            """
            if m == 120:
                plt.plot(np.arange(0, windowlen), wave[start:start+windowlen]*window)
                plt.show()
                plt.plot(np.fft.fftfreq(n=self._nfft, d=1.0/self._sr)[:self._nfft//2], np.abs(x)**2)
                plt.show()
            """
        #print("X.shape:", X.shape)
        # return magnified spectrum
        return np.abs(X)#/self.nfft
    
    #generate linearfilterbank
    def _linearFilterBank(self, nchannels=40):
        freq_min = 0
        freq_max = self.sr//2
        linear_centers = np.linspace(freq_min, freq_max, nchannels+2)
        bins = np.floor((self.nfft+1)*linear_centers/self.sr)
        #print(bins)

        filterbank = np.zeros((nchannels, self.nfft//2))
        for m in range(1, nchannels+1):
            f_m_minus = int(bins[m - 1])    # left
            f_m = int(bins[m])              # center
            f_m_plus = int(bins[m + 1])     # right

            for k in range(f_m_minus, f_m):
                filterbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
            for k in range(f_m, f_m_plus):
                filterbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])

        return filterbank
    
    #discrete cosine transform
    def _dct(self, mspec):
        return realtransforms.dct(mspec, type=2, norm='ortho', axis=-1)
    
    #delta-LFCC, double-delta-LFCC
    def _delta(self, x):
        #expect shape x = (nceps, frame_size)
        delta = np.zeros(x.shape)
        for t in range(delta.shape[1]):
            index_t_minus_one, index_t_plus_one = t-1, t+1
            if index_t_minus_one < 0:
                index_t_minus_one = 0
            if index_t_plus_one >= delta.shape[1]:
                index_t_plus_one = delta.shape[1]-1
            
            delta[:,t] = (x[:, index_t_plus_one] - x[:, index_t_minus_one])
        return delta

#end of class LFCC

if __name__ == "__main__":
    import librosa
    
    y, sr = librosa.load("utterance3.wav")
    lfcc = LFCC(y, sr).extract_feature().T
    
#    y, sr = librosa.load("utterance3.wav", sr=22050)
#    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12).T

    print(lfcc[100], lfcc.shape)
#   print(mfcc[100], mfcc.shape)