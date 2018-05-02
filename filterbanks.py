"""
Created 06/03/2018
@author: Will Wilkinson
"""

import numpy as np


class FilterBank(object):
    """
    Based on Josh McDermott's Matlab filterbank code:
    http://mcdermottlab.mit.edu/Sound_Texture_Synthesis_Toolbox_v1.7.zip

    leny = filter bank length in samples
    fs = sample rate
    N = number of frequency channels / subbands (excluding high-&low-pass which are added for perfect reconstruction)
    low_lim = centre frequency of first (lowest) channel
    high_lim = centre frequency of last (highest) channel
    """
    def __init__(self, leny, fs, n, low_lim, high_lim, q=2, ac_int=np.array([2, 4, 8, 16])):
        self.leny = leny
        self.fs = fs
        self.N = n
        self.low_lim = low_lim
        self.high_lim, self.freqs, self.nfreqs = self.check_limits(leny, fs, high_lim)
        self.Q = q
        self.ac_int = ac_int
        self.filters = np.zeros([self.nfreqs + 1, n + 2])
        self.subbands = np.zeros([self.nfreqs + 1, n + 2])

    @staticmethod
    def check_limits(leny, fs, high_lim):
        if np.remainder(leny, 2) == 0:
            nfreqs = leny / 2
            max_freq = fs / 2
        else:
            nfreqs = (leny - 1) / 2
            max_freq = fs * (leny - 1) / 2 / leny
        freqs = np.linspace(0, max_freq, nfreqs + 1)
        if high_lim > fs / 2:
            high_lim = max_freq
        return high_lim, freqs, int(nfreqs)

    def generate_subbands(self, signal):
        if signal.shape[0] == 1:  # turn into column vector
            signal = np.transpose(signal)
        n = self.filters.shape[1] - 2
        signal_length = signal.shape[0]
        filt_length = self.filters.shape[0]
        # watch out: numpy fft acts on rows, whereas Matlab fft acts on columns
        fft_sample = np.transpose(np.asmatrix(np.fft.fft(signal)))
        # generate negative frequencies in right place; filters are column vectors
        if np.remainder(signal_length, 2) == 0:  # even length
            fft_filts = np.concatenate([self.filters, np.flipud(self.filters[1:filt_length - 1, :])])
        else:  # odd length
            fft_filts = np.concatenate([self.filters, np.flipud(self.filters[1:filt_length, :])])
        # multiply by array of column replicas of fft_sample
        tile = np.dot(fft_sample, np.ones([1, n + 2]))
        fft_subbands = np.multiply(fft_filts, tile)
        # ifft works on rows; imag part is small, probably discretization error?
        self.subbands = np.transpose(np.real(np.fft.ifft(np.transpose(fft_subbands))))


class EqualRectangularBandwidth(FilterBank):
    def __init__(self, leny, fs, n, low_lim, high_lim):
        super(EqualRectangularBandwidth, self).__init__(leny, fs, n, low_lim, high_lim)
        # make cutoffs evenly spaced on an erb scale
        erb_low = self.freq2erb(self.low_lim)
        erb_high = self.freq2erb(self.high_lim)
        erb_lims = np.linspace(erb_low, erb_high, self.N + 2)
        self.cutoffs = self.erb2freq(erb_lims)
        self.filters = self.make_filters(self.N, self.nfreqs, self.freqs, self.cutoffs)

    @staticmethod
    def freq2erb(freq_hz):
        n_erb = 9.265 * np.log(1 + np.divide(freq_hz, 24.7 * 9.265))
        return n_erb

    @staticmethod
    def erb2freq(n_erb):
        freq_hz = 24.7 * 9.265 * (np.exp(np.divide(n_erb, 9.265)) - 1)
        return freq_hz

    def make_filters(self, n, nfreqs, freqs, cutoffs):
        cos_filts = np.zeros([nfreqs + 1, n])
        for k in range(n):
            l_k = cutoffs[k]
            h_k = cutoffs[k + 2]  # adjacent filters overlap by 50%
            l_ind = np.min(np.where(freqs > l_k))
            h_ind = np.max(np.where(freqs < h_k))
            avg = (self.freq2erb(l_k) + self.freq2erb(h_k)) / 2
            rnge = self.freq2erb(h_k) - self.freq2erb(l_k)
            # map cutoffs to -pi/2, pi/2 interval
            cos_filts[l_ind:h_ind + 1, k] = np.cos((self.freq2erb(freqs[l_ind:h_ind + 1]) - avg) / rnge * np.pi)
        # add lowpass and highpass to get perfect reconstruction
        filters = np.zeros([nfreqs + 1, n + 2])
        filters[:, 1:n + 1] = cos_filts
        # lowpass filter goes up to peak of first cos filter
        h_ind = np.max(np.where(freqs < cutoffs[1]))
        filters[:h_ind + 1, 0] = np.sqrt(1 - np.power(filters[:h_ind + 1, 1], 2))
        # highpass filter goes down to peak of last cos filter
        l_ind = np.min(np.where(freqs > cutoffs[n]))
        filters[l_ind:nfreqs + 1, n + 1] = np.sqrt(1 - np.power(filters[l_ind:nfreqs + 1, n], 2))
        return filters


class Linear(FilterBank):
    def __init__(self, leny, fs, n, low_lim, high_lim):
        super(Linear, self).__init__(leny, fs, n, low_lim, high_lim)
        self.cutoffs = np.linspace(self.low_lim, self.high_lim, self.N + 2)
        self.filters = self.make_filters(self.N, self.nfreqs, self.freqs, self.cutoffs)

    @staticmethod
    def make_filters(n, nfreqs, freqs, cutoffs):
        cos_filts = np.zeros([nfreqs + 1, n])
        for k in range(n):
            l_k = cutoffs[k]
            h_k = cutoffs[k + 2]  # adjacent filters overlap by 50%
            l_ind = np.min(np.where(freqs > l_k))
            h_ind = np.max(np.where(freqs < h_k))
            avg = (l_k + h_k) / 2
            rnge = h_k - l_k
            # map cutoffs to -pi/2, pi/2 interval
            cos_filts[l_ind:h_ind + 1, k] = np.cos((freqs[l_ind:h_ind + 1] - avg) / rnge * np.pi)
        # add lowpass and highpass to get perfect reconstruction
        filters = np.zeros([nfreqs + 1, n + 2])
        filters[:, 1:n + 1] = cos_filts
        # lowpass filter goes up to peak of first cos filter
        h_ind = np.max(np.where(freqs < cutoffs[1]))
        filters[:h_ind + 1, 0] = np.sqrt(1 - np.power(filters[:h_ind + 1, 1], 2))
        # highpass filter goes down to peak of last cos filter
        l_ind = np.min(np.where(freqs > cutoffs[n]))
        filters[l_ind:nfreqs + 1, n + 1] = np.sqrt(1 - np.power(filters[l_ind:nfreqs + 1, n], 2))
        return filters


class ConstQCos(FilterBank):
    def __init__(self, leny, fs, n, low_lim, high_lim, q):
        super(ConstQCos, self).__init__(leny, fs, n, low_lim, high_lim, q)
        self.center_freqs = 2 ** np.linspace(np.log2(low_lim), np.log2(high_lim), n)
        self.filters = self.make_filters(self.N, self.nfreqs, self.freqs, self.center_freqs, q)

    @staticmethod
    def make_filters(n, nfreqs, freqs, center_freqs, q):
        cos_filts = np.zeros([nfreqs + 1, n])
        for k in range(0, n):
            bw = center_freqs[k] / q
            lo = center_freqs[k] - bw
            hi = center_freqs[k] + bw
            l_ind = np.min(np.where(freqs > lo))
            h_ind = np.max(np.where(freqs < hi))
            avg = center_freqs[k]
            rnge = hi - lo
            cos_filts[l_ind:h_ind + 1, k] = np.cos((freqs[l_ind:h_ind + 1] - avg) / rnge * np.pi)
        temp = np.sum(cos_filts ** 2, 1)
        filters = cos_filts / np.sqrt(np.mean(temp[np.where(np.logical_and(freqs >= center_freqs[3], freqs <= center_freqs[-4]))]))
        return filters


class LinConstQCos(FilterBank):
    def __init__(self, leny, fs, n, low_lim, high_lim, q):
        super(LinConstQCos, self).__init__(leny, fs, n, low_lim, high_lim, q)
        self.center_freqs = np.linspace(low_lim, high_lim, n)
        self.Q_center_freqs = 2 ** np.linspace(np.log2(low_lim), np.log2(high_lim), n)
        self.filters = self.make_filters(self.N, self.nfreqs, self.freqs, self.center_freqs, q, self.Q_center_freqs)

    @staticmethod
    def make_filters(n, nfreqs, freqs, center_freqs, q, q_centre_freqs):
        cos_filts = np.zeros([nfreqs + 1, n])
        low_cut = q_centre_freqs - q_centre_freqs / q / 2
        hi_cut = q_centre_freqs + q_centre_freqs / q / 2
        low_bw = hi_cut[4] - low_cut[4]
        overlap = hi_cut[4] - low_cut[4+1]
        hi_bw = hi_cut[4+1] - low_cut[4+1]
        prop_overlap_hi = overlap / low_bw
        prop_overlap_low = overlap / hi_bw
        avg_overlap = (prop_overlap_hi + prop_overlap_low) / 2
        spacing = (center_freqs[-1] - center_freqs[0]) / (n - 1)
        bw = spacing / (1 - avg_overlap)
        for k in range(0, n):
            lo = center_freqs[k] - bw
            hi = center_freqs[k] + bw
            if lo <= 0:
                lo = 0
                l_ind = np.min(np.where(freqs > lo))
                h_ind = np.max(np.where(freqs < hi))
                center = center_freqs[k]
                c_ind = np.max(np.where(freqs < center))
                rnge1 = (center - lo) * 2
                rnge2 = (hi - center) * 2
                cos_filts[l_ind:c_ind+1, k] = np.cos((freqs[l_ind:c_ind+1] - center) / rnge1 * np.pi)
                cos_filts[c_ind:h_ind+1, k] = np.cos((freqs[c_ind:h_ind+1] - center) / rnge2 * np.pi)
            else:
                l_ind = np.min(np.where(freqs > lo))
                h_ind = np.max(np.where(freqs < hi))
                avg = center_freqs[k]
                rnge = hi - lo
                cos_filts[l_ind:h_ind+1, k] = np.cos((freqs[l_ind:h_ind+1] - avg) / rnge * np.pi)
        temp = np.sum(np.transpose(cos_filts) ** 2)
        filters = cos_filts / np.sqrt(np.mean(temp[np.where(center_freqs[3] <= freqs <= center_freqs[-4])]))
        return filters


class EnvAutocorrelation(FilterBank):
    def __init__(self, leny, fs, n, low_lim, high_lim, q, ac_int):
        super(EnvAutocorrelation, self).__init__(leny, fs, n, low_lim, high_lim, q, ac_int)
        self.N = ac_int.shape[0]
        nyquist = fs / 2
        self.filters, self.Hz_cutoffs = self.make_filters(self.N, nyquist, ac_int, self.nfreqs, self.freqs)

    @staticmethod
    def make_filters(n, nyquist, lags_smp, nfreqs, freqs):
        filters = np.zeros([nfreqs + 1, n])
        hz_cutoffs = np.zeros(n)
        for k in range(0, n):
            if k == 0:
                high_cutoff = 1 / (4 * lags_smp[k] / 1000)
                low_cutoff = 0.5 * 1 / (4 * lags_smp[k] / 1000)
            else:
                high_cutoff = 1 / (4 * (lags_smp[k] - lags_smp[k-1]) / 1000)
                low_cutoff = 0.5 * 1 / (4 * (lags_smp[k] - lags_smp[k-1]) / 1000)
            if high_cutoff > nyquist:
                filters[:, k] = 1
            else:
                l_ind = np.min(np.where(freqs > low_cutoff))
                h_ind = np.max(np.where(freqs < high_cutoff))
                if l_ind < h_ind:
                    filters[0:l_ind, k] = 1
                    filters[l_ind:h_ind+1, k] = np.cos((freqs[l_ind:h_ind+1] - freqs[l_ind]) / (freqs[l_ind] - freqs[h_ind]) * np.pi / 2)
                else:
                    filters[0:l_ind, k] = 1
            hz_cutoffs[k] = high_cutoff
        return filters, hz_cutoffs


class OctaveCos(FilterBank):
    def __init__(self, leny, fs, n, low_lim, high_lim):
        super(OctaveCos, self).__init__(leny, fs, n, low_lim, high_lim)
        if high_lim > fs / 2:
            high_lim = self.freqs[-1]
        cutoffs = high_lim / 2 ** np.linspace(0, 20, 21)
        self.cutoffs = np.flipud(cutoffs[np.where(cutoffs > low_lim)][:, None])
        self.center_freqs = self.cutoffs[:-1]
        self.N = self.center_freqs.shape[0]
        self.filters, _, _ = self.make_filters(self.N, self.nfreqs, self.freqs, self.cutoffs, self.center_freqs)

    @staticmethod
    def make_filters(n, nfreqs, freqs, cutoffs, center_freqs):
        filters = np.zeros([nfreqs + 1, n])
        for k in range(0, n):
            lo = center_freqs[k] / 2
            hi = cutoffs[k] * 2
            l_ind = np.min(np.where(freqs > lo))
            h_ind = np.max(np.where(freqs < hi))
            avg = (np.log2(lo) + np.log2(hi)) / 2
            rnge = (np.log2(hi) - np.log2(lo))
            filters[l_ind:h_ind+1, k] = np.cos((np.log2(freqs[l_ind:h_ind+1]) - avg) / rnge * np.pi)
        return filters, center_freqs, freqs


class LinearOctaveCos(FilterBank):
    def __init__(self, leny, fs, n, low_lim, high_lim):
        super(LinearOctaveCos, self).__init__(leny, fs, n, low_lim, high_lim)
        if high_lim > fs / 2:
            high_lim = self.freqs[-1]
        self.center_freqs = np.linspace(low_lim, high_lim, n + 1)
        self.cutoffs = np.concatenate([0, self.center_freqs])
        self.filters, self.center_freqs = self.make_filters(self.N, self.nfreqs, self.freqs, self.cutoffs, self.center_freqs)

    @staticmethod
    def make_filters(n, nfreqs, freqs, cutoffs, center_freqs):
        filters = np.zeros([nfreqs + 1, n])
        for k in range(0, n):
            if k == 0:
                lo = cutoffs[k]
                hi = cutoffs[k + 2]
                l_ind = np.min(np.where(freqs > lo))
                h_ind = np.max(np.where(freqs < hi))
                center = cutoffs[k + 1]
                c_ind = np.max(np.where(freqs < center))
                rnge1 = (center - lo) * 2
                rnge2 = (hi - center) * 2
                filters[l_ind:c_ind+1, k] = np.cos((freqs[l_ind:c_ind+1] - center) / rnge1 * np.pi)
                filters[c_ind:h_ind+1, k] = np.cos((freqs[c_ind:h_ind+1] - center) / rnge2 * np.pi)
            else:
                lo = cutoffs[k]
                hi = cutoffs[k + 2]
                l_ind = np.min(np.where(freqs > lo))
                h_ind = np.max(np.where(freqs < hi))
                avg = (lo + hi) / 2
                rnge = hi - lo
                filters[l_ind:h_ind+1, k] = np.cos((freqs[l_ind:h_ind+1] - avg) / rnge * np.pi)
        center_freqs = center_freqs[0:-1]
        return filters, center_freqs
