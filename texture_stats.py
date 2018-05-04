import numpy as np
from numpy import transpose as tp
import scipy.signal as sig
import scipy.stats as scistat
import filterbanks as fb


class SoundTexture(object):
    """
    Based on Josh McDermott's Matlab toolbox:
    http://mcdermottlab.mit.edu/Sound_Texture_Synthesis_Toolbox_v1.7.zip

    y = audio file
    fs = sample rate
    """
    def __init__(self, y, fs):
        self.y = y
        self.fs = fs
        # default settings:
        self.desired_rms = .01
        self.audio_sr = 20000
        self.n_audio_channels = 30
        self.low_audio_f = 20
        self.hi_audio_f = 10000
        self.use_more_audio_filters = 0
        self.lin_or_log_filters = 1
        self.env_sr = 400
        self.n_mod_channels = 20
        self.low_mod_f = 0.5
        self.hi_mod_f = 200
        self.use_more_mod_filters = 0
        self.mod_filt_Q_value = 2
        self.use_zp = 0
        self.low_mod_f_c12 = 1
        self.compression_option = 1
        self.comp_exponent = .3
        self.log_constant = 10 ** -12
        self.match_env_hist = 0
        self.match_sub_hist = 0
        self.n_hist_bins = 128
        self.manual_mean_var_adjustment = 0
        self.max_orig_dur_s = 7
        self.desired_synth_dur_s = 5
        self.measurement_windowing = 2
        self.imposition_windowing = 1
        self.win_steepness = .5
        self.imposition_method = 1
        self.sub_imposition_order = 1
        self.env_ac_intervals_smp = np.array([1, 2, 3, 4, 5, 6, 7, 9, 11, 14, 18, 22, 28, 36, 45, 57, 73, 92, 116, 148, 187, 237, 301])  # in samples
        self.sub_ac_undo_win = 1
        self.sub_ac_win_choice = 2
        self.num_sub_ac_period = 5
        # allocate memory:
        self.mod_c2 = []
        self.mod_c1 = []
        self.env_c = []
        self.subband_ac = []
        self.mod_power_center_freqs = []
        self.mod_c2_center_freqs = []
        self.mod_c1_center_freqs = []
        self.audio_cutoffs_hz = []
        self.subband_mean = np.zeros(self.n_audio_channels + 2)
        self.subband_var = np.zeros(self.n_audio_channels + 2)
        self.subband_skew = np.zeros(self.n_audio_channels + 2)
        self.subband_kurt = np.zeros(self.n_audio_channels + 2)
        self.env_mean = np.zeros(self.n_audio_channels + 2)
        self.env_var = np.zeros(self.n_audio_channels + 2)
        self.env_skew = np.zeros(self.n_audio_channels + 2)
        self.env_kurt = np.zeros(self.n_audio_channels + 2)
        self.subband_hist = np.zeros([self.n_audio_channels + 2 + 1, self.n_hist_bins])
        self.subband_bins = np.zeros([self.n_audio_channels + 2 + 1, self.n_hist_bins])
        self.env_hist = np.zeros([self.n_audio_channels + 2, self.n_hist_bins])
        self.env_bins = np.zeros([self.n_audio_channels + 2, self.n_hist_bins])
        self.env_ac = np.zeros([self.n_audio_channels + 2, self.env_ac_intervals_smp.shape[0]])
        self.mod_power = np.zeros([self.n_audio_channels + 2, self.n_mod_channels])
        self.subband_ac_power = np.zeros(self.n_audio_channels + 2)
        # calculate stats:
        self.orig_sound, self.ds_factor = self.format_orig_sound()
        self.measurement_win = self.set_measurement_window(self.orig_sound.shape[0], self.measurement_windowing)
        self.measure_texture_stats(self.orig_sound, self.measurement_win)

    def format_orig_sound(self):
        orig_sound = self.y
        if orig_sound.ndim == 2:
            orig_sound = (orig_sound[:, 0] + orig_sound[:, 1]) / 2  # if stereo convert to mono
        if self.fs != self.audio_sr:
            orig_sound = sig.resample(orig_sound, int(orig_sound.shape[0] * self.audio_sr / self.fs))
        if np.remainder(orig_sound.shape[0], 2) == 1:
            orig_sound = np.concatenate([orig_sound, np.array([0])])
        ds_factor = self.audio_sr / self.env_sr
        new_l = int(np.floor((orig_sound.shape[0] / ds_factor / 2) * ds_factor * 2))
        orig_sound = orig_sound[:new_l]
        orig_sound = orig_sound / np.sqrt(np.mean(np.square(orig_sound))) * self.desired_rms
        return orig_sound, ds_factor

    def set_measurement_window(self, sound_length, windowing_option):
        if windowing_option == 1:
            measurement_win = np.ones([int(sound_length / self.ds_factor), 1])
        elif windowing_option == 2:
            temp = self.make_windows_rcos_flat_no_ends(int(sound_length / self.ds_factor), int(np.round(sound_length / self.audio_sr)), self.win_steepness)
            measurement_win = np.sum(temp, 1)
        else:
            raise Exception('measurement_win must be 1 or 2')
        return measurement_win

    @staticmethod
    def make_windows_rcos_flat_no_ends(signal_length_smp, num_secs, ramp_prop):
        num_secs = num_secs + 2
        if ramp_prop == 0.5:
            ramp_length_smp = int(np.floor(signal_length_smp / (num_secs - 1)))
            flat_length_smp = 0
        elif ramp_prop < 0.5:
            flat_length = signal_length_smp / (num_secs * (1 - ramp_prop) / (1 - 2 * ramp_prop) - ramp_prop / (1 - 2 * ramp_prop))
            ramp_length_smp = int(np.floor(flat_length * ramp_prop / (1 - 2 * ramp_prop)))
            flat_length_smp = int(np.floor(flat_length))
        else:
            raise Exception('ramp_prop must be less than .5')
        windows = np.zeros([signal_length_smp, num_secs])
        windows[:flat_length_smp, 0] = 2
        windows[flat_length_smp: flat_length_smp + ramp_length_smp, 0] = np.cos(np.linspace(1, ramp_length_smp, num=ramp_length_smp) / ramp_length_smp * np.pi) + 1
        start_pt = flat_length_smp
        for n in range(0, num_secs - 2):
            windows[start_pt:start_pt+ramp_length_smp, n+1] = np.cos(np.linspace(-ramp_length_smp+1, 0, num=ramp_length_smp) / ramp_length_smp * np.pi) + 1
            windows[start_pt+ramp_length_smp:start_pt+ramp_length_smp+flat_length_smp, n+1] = 2
            windows[start_pt+ramp_length_smp+flat_length_smp:start_pt+2*ramp_length_smp+flat_length_smp, n+1] = np.cos(np.linspace(1, ramp_length_smp, num=ramp_length_smp) / ramp_length_smp * np.pi) + 1
            start_pt = start_pt + flat_length_smp + ramp_length_smp
        windows[start_pt:start_pt+ramp_length_smp, num_secs-1] = np.cos(np.linspace(-ramp_length_smp + 1, 0, num=ramp_length_smp) / ramp_length_smp * np.pi) + 1
        windows[start_pt + ramp_length_smp:signal_length_smp, num_secs-1] = 2
        windows = windows[:, 1:-1]
        windows = windows / 2
        return windows

    @staticmethod
    def stat_central_moment_win(x, n, win, x_mean=-99):
        win = win / np.sum(win)
        if x_mean == -99:
            x_mean = np.sum(win * x)
        if n == 1:
            m = x_mean
        elif n == 2:
            m = np.sum(win * ((x - x_mean) ** 2))
            m = np.sqrt(m) / x_mean
        elif n == 3:
            m2 = np.sum(win * ((x - x_mean) ** 2))
            m = np.sum(win * ((x - x_mean) ** 3)) / (m2 ** (3.0 / 2.0))
        elif n == 4:
            m2 = np.sum(win * ((x - x_mean) ** 2))
            m = np.sum(win * ((x - x_mean) ** 4)) / (m2 ** 2)
        else:
            raise Exception('input value of n not recognised')
        return m

    @staticmethod
    def shift_s(s, num_samples):
        if num_samples == 0:
            new_s = s
        elif num_samples < 0:
            new_s = np.concatenate([s[-num_samples:], np.zeros(-num_samples)])
        else:
            new_s = np.concatenate([np.zeros(num_samples), s[:-num_samples]])
        return new_s

    def stat_env_ac_scaled_win(self, f_env, sample_spacing, use_zp, win):
        if use_zp != 0:
            raise Exception('zero padding not implemented')
        win = win / np.sum(win)
        ac_values = np.zeros(sample_spacing.shape[0])
        for p in range(0, sample_spacing.shape[0]):
            num_samp = sample_spacing[p]
            meanf_env = np.mean(f_env[:, p])
            mf_env = f_env[:, p] - meanf_env
            env_var = np.mean(mf_env ** 2)
            ac_values[p] = np.sum(win * (self.shift_s(mf_env, -num_samp) * self.shift_s(mf_env, num_samp))) / env_var
        return ac_values

    @staticmethod
    def stat_var_win(s, win):
        win = win / np.sum(win)
        w_var = np.sum(win * (s - np.sum(win * s)) ** 2)
        return w_var

    def stat_mod_power_win(self, s, mod_subbands, use_zp, win):
        if use_zp != 0:
            raise Exception('zero padding not implemented')
        win = win / np.sum(win)
        s_var = self.stat_var_win(s, win)
        mp = np.sum(np.dot(win[:, None], np.ones([1, mod_subbands.shape[1]])) * (mod_subbands ** 2), 0) / s_var
        return mp

    @staticmethod
    def stat_mod_c2_win(subbands, use_zp, win):
        if use_zp != 0:
            raise Exception('zero padding not implemented')
        win = win / np.sum(win)
        analytic_subbands = np.transpose(sig.hilbert(np.transpose(subbands)))
        n = analytic_subbands.shape[1]
        c2 = np.zeros([n-1, 2])
        for k in range(0, n-1):
            c = (analytic_subbands[:, k] ** 2) / np.abs(analytic_subbands[:, k])
            sig_cw = np.sqrt(np.sum(win * (np.real(c) ** 2)))
            sig_fw = np.sqrt(np.sum(win * (np.real(analytic_subbands[:, k+1]) ** 2)))
            c2[k, 0] = np.sum(win * np.real(c) * np.real(analytic_subbands[:, k+1])) / (sig_cw * sig_fw)
            c2[k, 1] = np.sum(win * np.real(c) * np.imag(analytic_subbands[:, k + 1])) / (sig_cw * sig_fw)
        return c2

    @staticmethod
    def stat_corr_filt_win_full(f_envs, use_zp, win):
        if use_zp != 0:
            raise Exception('zero padding not implemented')
        win = win / np.sum(win)
        cbc_value = np.zeros([f_envs.shape[1], f_envs.shape[1]])
        meanf_envs = np.mean(f_envs, 0)[None, :]
        mf_envs = f_envs - np.dot(np.ones([f_envs.shape[0], 1]), meanf_envs)
        env_stds = np.sqrt(np.mean(mf_envs ** 2, 0))[None, :]
        cbc_value[:, :] = np.dot(np.transpose((np.dot(win[:, None], np.ones([1, f_envs.shape[1]]))) * mf_envs), mf_envs) / np.dot(np.transpose(env_stds), env_stds)
        return cbc_value

    @staticmethod
    def autocorr_mult(x):
        xf = np.transpose(np.fft.fft(np.transpose(x)))
        xf2 = np.abs(xf) ** 2
        cx2 = np.transpose(np.real(np.fft.ifft(np.transpose(xf2))))
        cx = np.zeros_like(cx2)
        for j in range(0, cx2.shape[1]):
            cx[:, j] = np.fft.fftshift(cx2[:, j])
        return cx

    def autocorr_mult_zp(self, s, win_choice, undo_win):
        n = s.shape[1] - 2
        s_l = s.shape[0]
        wt = np.linspace(1, s_l, num=s_l) / s_l
        if win_choice == 1:  # hanning
            w = 0.5 - 0.5 * np.cos(2 * np.pi * wt)
        elif win_choice == 2:  # rect
            w = np.ones_like(wt)
        elif win_choice == 3:  # hamming
            w = 0.54 - 0.46 * np.cos(2 * np.pi * wt)
        elif win_choice == 4:  # hamming
            w = 0.6 - 0.4 * np.cos(2 * np.pi * wt)
        elif win_choice == 5:  # welch
            w = np.sin(np.pi * wt)
        else:
            raise Exception('window type not recognised')
        s_w = s * np.dot(np.transpose(w[None, :]), np.ones([1, n+2]))
        s_wp = np.vstack([np.zeros([int(s_l / 2), int(n + 2)]), s_w, np.zeros([int(s_l / 2), int(n + 2)])])
        w_p = np.vstack([np.zeros([int(w.shape[0] / 2), 1]), w[:, None], np.zeros([int(w.shape[0] / 2), 1])])
        ac = self.autocorr_mult(s_wp)
        if undo_win:
            w_ac = self.autocorr_mult(w_p)
            ac = ac / np.dot(w_ac, np.ones([1, int(n + 2)]))
        ac = ac[int(s_l / 2):int(3 * s_l / 2), :]
        return ac

    def measure_texture_stats(self, sample_sound, measurement_win):
        # Construct the filter banks
        if self.use_more_audio_filters == 0:
            if self.lin_or_log_filters == 1 or self.lin_or_log_filters == 2:
                filt_bank = fb.EqualRectangularBandwidth(self.orig_sound.shape[0], self.audio_sr, self.n_audio_channels, self.low_audio_f, self.hi_audio_f)
            elif self.lin_or_log_filters == 3 or self.lin_or_log_filters == 4:
                filt_bank = fb.Linear(self.orig_sound.shape[0], self.audio_sr, self.n_audio_channels, self.low_audio_f, self.hi_audio_f)
            else:
                raise Exception('filter type not recognised')
        else:
            raise Exception('double and quadruple audio filters not implemented')
        self.audio_cutoffs_hz = filt_bank.cutoffs
        filt_bank.generate_subbands(sample_sound)
        subbands = filt_bank.subbands  # [:, 1:-1]
        subband_envs = tp(np.absolute(sig.hilbert(tp(subbands))))
        if self.compression_option == 1:
            subband_envs = subband_envs ** self.comp_exponent
        elif self.compression_option == 2:
            subband_envs = np.log10(subband_envs + self.log_constant)
        subband_envs = sig.resample(subband_envs, int(subband_envs.shape[0] / self.ds_factor))
        subband_envs[subband_envs < 0] = 0
        if self.use_zp == 1:
            mod_filt_length = subband_envs.shape[0] * 2
        elif self.use_zp == 0:
            mod_filt_length = subband_envs.shape[0]
        else:
            raise Exception('use_zp input not recognised')
        if self.lin_or_log_filters == 1 or self.lin_or_log_filters == 3:
            const_q_bank = fb.ConstQCos(mod_filt_length, self.env_sr, self.n_mod_channels, self.low_mod_f, self.hi_mod_f, self.mod_filt_Q_value)
        elif self.lin_or_log_filters == 2 or self.lin_or_log_filters == 4:
            const_q_bank = fb.LinConstQCos(mod_filt_length, self.env_sr, self.n_mod_channels, self.low_mod_f, self.hi_mod_f, self.mod_filt_Q_value)
        else:
            raise Exception('lin_or_log_filters input not recognised')
        env_ac_bank = fb.EnvAutocorrelation(mod_filt_length, self.env_sr, self.n_mod_channels, self.low_mod_f, self.hi_mod_f, self.mod_filt_Q_value, self.env_ac_intervals_smp)
        octave_bank = fb.OctaveCos(mod_filt_length, self.env_sr, self.n_mod_channels, self.low_mod_f_c12, self.hi_mod_f)
        if self.lin_or_log_filters == 1 or self.lin_or_log_filters == 3:
            mod_c1_bank = octave_bank
            c1_ind = 1
        elif self.lin_or_log_filters == 2 or self.lin_or_log_filters == 4:
            mod_c1_bank = fb.LinearOctaveCos(mod_filt_length, self.env_sr, self.n_mod_channels, self.low_mod_f_c12, self.hi_mod_f)
            c1_ind = 0
        else:
            raise Exception('filter type not recognised')
        # Now calculate the stats
        self.subband_mean = np.mean(subbands, 0)
        self.subband_var = np.var(subbands, 0)
        self.mod_c2 = np.zeros([self.n_audio_channels + 2, octave_bank.N - 1, 2])
        self.mod_c1 = np.zeros([subband_envs.shape[1], subband_envs.shape[1], mod_c1_bank.N - c1_ind])
        for j in range(0, self.n_audio_channels + 2):
            self.subband_skew[j] = scistat.skew(subbands[:, j])
            self.subband_kurt[j] = scistat.kurtosis(subbands[:, j], fisher=False)
            self.env_mean[j] = self.stat_central_moment_win(subband_envs[:, j], 1, measurement_win)
            self.env_var[j] = self.stat_central_moment_win(subband_envs[:, j], 2, measurement_win, self.env_mean[j])
            self.env_skew[j] = self.stat_central_moment_win(subband_envs[:, j], 3, measurement_win, self.env_mean[j])
            self.env_kurt[j] = self.stat_central_moment_win(subband_envs[:, j], 4, measurement_win, self.env_mean[j])
            temp, bins = np.histogram(subbands[:, j], self.n_hist_bins)
            temp = temp.astype(float, copy=False)
            bins = bins.astype(float, copy=False)
            bins = (bins[:-1] + bins[1:]) / 2  # get bin centres
            self.subband_hist[j, :self.n_hist_bins] = temp / np.sum(temp)
            self.subband_bins[j, :self.n_hist_bins] = bins
            temp, bins = np.histogram(subband_envs[:, j], self.n_hist_bins)
            temp = temp.astype(float, copy=False)
            bins = bins.astype(float, copy=False)
            bins = (bins[:-1] + bins[1:]) / 2  # get bin centres
            self.env_hist[j, :self.n_hist_bins] = temp / np.sum(temp)
            self.env_bins[j, :self.n_hist_bins] = bins
            env_ac_bank.generate_subbands(subband_envs[:, j])
            f_env = env_ac_bank.subbands
            self.env_ac[j, :] = self.stat_env_ac_scaled_win(f_env, self.env_ac_intervals_smp, self.use_zp, measurement_win)
            const_q_bank.generate_subbands(subband_envs[:, j])
            mod_subbands = const_q_bank.subbands
            self.mod_power[j, :] = self.stat_mod_power_win(subband_envs[:, j], mod_subbands, self.use_zp, measurement_win)
            self.mod_power_center_freqs = const_q_bank.center_freqs
            octave_bank.generate_subbands(subband_envs[:, j])
            mod_c2_subbands = octave_bank.subbands
            self.mod_c2[j, :, :] = self.stat_mod_c2_win(mod_c2_subbands, self.use_zp, measurement_win)
            self.mod_c2_center_freqs = octave_bank.center_freqs[:-1]
        # compute subband envelope, modulation band correlations
        self.env_c = self.stat_corr_filt_win_full(subband_envs, self.use_zp, measurement_win)
        f_envs = np.zeros_like(subband_envs)
        for k in range(0, mod_c1_bank.N - c1_ind):
            for i in range(0, subband_envs.shape[1]):
                mod_c1_bank.generate_subbands(subband_envs[:, i])
                f_envs[:, i] = mod_c1_bank.subbands[:, k + c1_ind]  # exclude first
            self.mod_c1[:, :, k] = self.stat_corr_filt_win_full(f_envs, self.use_zp, measurement_win)
        self.mod_c1_center_freqs = mod_c1_bank.center_freqs
        # subband autocorrelation
        sub_ac_n_smp = np.round(self.num_sub_ac_period / self.audio_cutoffs_hz * self.audio_sr)
        sub_ac_n_smp[sub_ac_n_smp > self.num_sub_ac_period / 20.0 * self.audio_sr] = self.num_sub_ac_period / 20.0 * self.audio_sr
        temp = self.autocorr_mult_zp(subbands, self.sub_ac_win_choice, self.sub_ac_undo_win)
        l2 = subbands.shape[0]
        c2 = l2 / 2
        for k in range(0, self.n_audio_channels + 2):
            self.subband_ac.append(temp[int(c2 - sub_ac_n_smp[k]):int(c2 + sub_ac_n_smp[k] + 1), k])
            self.subband_ac_power[k] = np.sum(self.subband_ac[k] ** 2)  # used in SNR calculation
        amp_hist, amp_bins = np.histogram(sample_sound, self.n_hist_bins)
        amp_bins = (amp_bins[:-1] + amp_bins[1:]) / 2  # get bin centres
        self.subband_hist[self.n_audio_channels + 2, :self.n_hist_bins] = amp_hist
        self.subband_bins[self.n_audio_channels + 2, :self.n_hist_bins] = amp_bins
