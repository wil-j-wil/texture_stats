import numpy as np
import librosa as lb
import texture_stats
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.pyplot import figure, clf, plot, subplot, title, semilogy, semilogx, gca, imshow, xlabel, ylabel, show, rcParams

proj_path = ''
file_name = 'stim118_puppy_whining'
file_name_2 = 'stim7_applause'

# statistical analysis of two audio signals
y, fs = lb.load(proj_path + 'audio/' + file_name + '.wav', sr=None)
y_2, fs_2 = lb.load(proj_path + 'audio/' + file_name_2 + '.wav', sr=None)

sound_texture = texture_stats.SoundTexture(y, fs)
sound_texture_2 = texture_stats.SoundTexture(y_2, fs_2)

##
# plot the stats
rcParams.update({'font.size' : 6})

figure(1)
clf()
subplot(221)
plot(sound_texture.subband_mean, 'b', lw=0.7)
plot(sound_texture_2.subband_mean, 'r', lw=0.7)
title('Subband Mean', fontweight='bold')
gca().ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
subplot(222)
plot(sound_texture.subband_var, 'b', lw=0.7)
plot(sound_texture_2.subband_var, 'r', lw=0.7)
title('Subband Variance', fontweight='bold')
gca().ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
subplot(223)
plot(sound_texture.subband_skew, 'b', lw=0.7)
plot(sound_texture_2.subband_skew, 'r', lw=0.7)
title('Subband Skew', fontweight='bold')
xlabel('Subband number')
gca().ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
subplot(224)
plot(sound_texture.subband_kurt, 'b', lw=0.7)
plot(sound_texture_2.subband_kurt, 'r', lw=0.7)
title('Subband Kurtosis', fontweight='bold')
xlabel('Subband number')
gca().ticklabel_format(axis='y', style='sci', scilimits=(-3,3))

figure(2)
clf()
subplot(221)
plot(sound_texture.env_mean, 'b', lw=0.7)
plot(sound_texture_2.env_mean, 'r', lw=0.7)
title('Subband Envelope Mean', fontweight='bold')
subplot(222)
plot(sound_texture.env_var, 'b', lw=0.7)
plot(sound_texture_2.env_var, 'r', lw=0.7)
title('Subband Envelope Variance', fontweight='bold')
subplot(223)
plot(sound_texture.env_skew, 'b', lw=0.7)
plot(sound_texture_2.env_skew, 'r', lw=0.7)
title('Subband Envelope Skew', fontweight='bold')
xlabel('Subband number')
subplot(224)
plot(sound_texture.env_kurt, 'b', lw=0.7)
plot(sound_texture_2.env_kurt, 'r', lw=0.7)
title('Subband Envelope Kurtosis', fontweight='bold')
xlabel('Subband number')

figure(3)
clf()
for k in range(0, sound_texture.env_bins.shape[0]):
    subplot(6, 6, k + 1)
    semilogy(sound_texture.env_bins[k, :], sound_texture.env_hist[k, :], 'b', lw=0.7)
    semilogy(sound_texture_2.env_bins[k, :], sound_texture_2.env_hist[k, :], 'r', lw=0.7)
    gca().set_xlim([-0.01, 0.35])
    gca().set_ylim([10**-4, 10**-1])
    if np.remainder(k, 6) != 0:
        gca().get_yaxis().set_ticks([])
    else:
        ylabel('Prob. of Occ.')
    if k < 26:
        gca().get_xaxis().set_ticks([])
    else:
        xlabel('Env. Magn.')
    if k == 2:
        title('Envelope Histogram Comparison (Env. Magnitude x Prob. of Occurrence)', fontweight='bold')

figure(4)
clf()
for k in range(0, sound_texture.subband_bins.shape[0] - 1):
    subplot(6, 6, k + 1)
    semilogy(sound_texture.subband_bins[k, :], sound_texture.subband_hist[k, :], 'b', lw=0.7)
    semilogy(sound_texture_2.subband_bins[k, :], sound_texture_2.subband_hist[k, :], 'r', lw=0.7)
    gca().set_xlim([-0.035, 0.035])
    gca().set_ylim([10**-5, 10**0])
    if np.remainder(k, 6) != 0:
        gca().get_yaxis().set_ticks([])
    else:
        ylabel('Prob. of Occ.')
    if k < 26:
        gca().get_xaxis().set_ticks([])
    else:
        xlabel('Filt. Resp.')
    if k == 2:
        title('Subband Histogram Comparison (Filter Response x Prob. of Occurrence)', fontweight='bold')

figure(5)
clf()
for k in range(0, sound_texture.mod_power.shape[0]):
    subplot(6, 6, k + 1)
    semilogx(sound_texture.mod_power_center_freqs, sound_texture.mod_power[k, :], 'b', lw=0.7)
    semilogx(sound_texture_2.mod_power_center_freqs, sound_texture_2.mod_power[k, :], 'r', lw=0.7)
    if np.remainder(k, 6) != 0:
        gca().get_yaxis().set_ticks([])
    else:
        ylabel('Mod. Power')
    if k < 26:
        gca().get_xaxis().set_ticks([])
    else:
        xlabel('Mod Freq. (Hz)')
    if k == 2:
        title('Modulation Power Comparison (Mod Freq (Hz) x Channel Power)', fontweight='bold')

figure(6)
clf()
for k in range(0, sound_texture.mod_power.shape[0]):
    subplot(6, 6, k + 1)
    # plot(sound_texture.subband_ac[k], 'b', lw=0.7)
    # plot(sound_texture_2.subband_ac[k], 'r', lw=0.7)
    if k == 2:
        title('Subband Autocorrelation (TODO)', fontweight='bold')

figure(7)
clf()
for k in range(0, sound_texture.env_ac.shape[0]):
    subplot(6, 6, k + 1)
    semilogx(sound_texture.env_ac_intervals_smp, sound_texture.env_ac[k, :], 'b', lw=0.7)
    semilogx(sound_texture_2.env_ac_intervals_smp, sound_texture_2.env_ac[k, :], 'r', lw=0.7)
    gca().set_ylim([-1, 1])
    if np.remainder(k, 6) != 0:
        gca().get_yaxis().set_ticks([])
    else:
        ylabel('Autocorr.')
    if k < 26:
        gca().get_xaxis().set_ticks([])
    else:
        xlabel('Lag (ms)')
    if k == 2:
        title('Envelope Autocorrelation Comparison (Lag (ms) x Autocorr)', fontweight='bold')

figure(8)
clf()
for k in range(0, sound_texture.env_ac.shape[0]):
    subplot(6, 6, k + 1)
    semilogx(sound_texture.mod_c2_center_freqs, sound_texture.mod_c2[k, :, 0], 'bo-', lw=0.7, ms=3)
    semilogx(sound_texture.mod_c2_center_freqs, sound_texture.mod_c2[k, :, 1], 'bx-', lw=0.7, ms=5)
    semilogx(sound_texture_2.mod_c2_center_freqs, sound_texture_2.mod_c2[k, :, 0], 'ro-', lw=0.7, ms=3)
    semilogx(sound_texture_2.mod_c2_center_freqs, sound_texture_2.mod_c2[k, :, 1], 'rx-', lw=0.7, ms=5)
    gca().set_ylim([-1, 1])
    if np.remainder(k, 6) != 0:
        gca().get_yaxis().set_ticks([])
    else:
        ylabel('C2 corr.')
    if k < 26:
        gca().get_xaxis().set_ticks([])
    else:
        xlabel('Mod Freq. (Hz)')
    if k == 2:
        title('Mod. C2 Comparison (Mod Freq. (Hz) x C2 Correlation, dots=real, crosses=imag)', fontweight='bold')

figure(9)
clf()
for k in range(0, sound_texture.mod_c1.shape[2]):
    subplot(3, 4, 2 * k + 1)
    imshow(sound_texture.mod_c1[:, :, k])
    title('(A) %sHz  channel'%sound_texture.mod_c1_center_freqs[k + 1][0])
    if np.remainder(2 * k, 4) != 0:
        gca().get_yaxis().set_ticks([])
    if k < 4:
        gca().get_xaxis().set_ticks([])
    subplot(3, 4, 2 * k + 2)
    imshow(sound_texture_2.mod_c1[:, :, k])
    if k > 0:
        title('(B) %sHz  channel' % sound_texture.mod_c1_center_freqs[k + 1][0])
    gca().get_yaxis().set_ticks([])
    if k < 4:
        gca().get_xaxis().set_ticks([])
    if k == 0:
        title('C1 Comparison', fontweight='bold')

figure(10)
clf()
subplot(441)
title('Power (TODO)')
subplot(442)
semilogx(sound_texture.audio_cutoffs_hz, sound_texture.env_mean, 'b', lw=0.7)
semilogx(sound_texture_2.audio_cutoffs_hz, sound_texture_2.env_mean, 'r', lw=0.7)
title('Env. Mean')
subplot(443)
semilogx(sound_texture.audio_cutoffs_hz, sound_texture.env_var, 'b', lw=0.7)
semilogx(sound_texture_2.audio_cutoffs_hz, sound_texture_2.env_var, 'r', lw=0.7)
title('Env. Var.')
subplot(444)
semilogx(sound_texture.audio_cutoffs_hz, sound_texture.env_skew, 'b', lw=0.7)
semilogx(sound_texture_2.audio_cutoffs_hz, sound_texture_2.env_skew, 'r', lw=0.7)
title('Env. Skew.')
subplot(445)
imshow(np.flipud(sound_texture.mod_power))
ylabel('Mod. Power')
gca().get_xaxis().set_ticks([])
subplot(446)
imshow(np.flipud(sound_texture_2.mod_power))
ylabel('Mod. Power')
gca().get_xaxis().set_ticks([])
gca().get_yaxis().set_ticks([])
subplot(4, 8, 13)
imshow(np.flipud(sound_texture.mod_c2[:, :, 0]))
ylabel('Mod. C2 Re-Re')
gca().get_xaxis().set_ticks([])
gca().get_yaxis().set_ticks([])
subplot(4, 8, 14)
imshow(np.flipud(sound_texture.mod_c2[:, :, 1]))
ylabel('Mod. C2 Re-Im')
gca().get_xaxis().set_ticks([])
gca().get_yaxis().set_ticks([])
subplot(4, 8, 15)
imshow(np.flipud(sound_texture_2.mod_c2[:, :, 0]))
ylabel('Mod. C2 Re-Re')
gca().get_xaxis().set_ticks([])
gca().get_yaxis().set_ticks([])
subplot(4, 8, 16)
imshow(np.flipud(sound_texture_2.mod_c2[:, :, 1]))
ylabel('Mod. C2 Re-Im')
gca().get_xaxis().set_ticks([])
gca().get_yaxis().set_ticks([])
subplot(449)
imshow(np.flipud(sound_texture.env_c))
ylabel('Env C')
gca().get_xaxis().set_ticks([])
subplot(4, 4, 10)
imshow(np.flipud(sound_texture_2.env_c))
ylabel('Env C')
gca().get_xaxis().set_ticks([])
gca().get_yaxis().set_ticks([])
subplot(4, 4, 11)
imshow(np.flipud(sound_texture.mod_c1[:, :, 2]))
ylabel('Mod C1 (Sub 3)')
gca().get_xaxis().set_ticks([])
gca().get_yaxis().set_ticks([])
subplot(4, 4, 12)
imshow(np.flipud(sound_texture_2.mod_c1[:, :, 2]))
ylabel('Mod C1 (Sub 3)')
gca().get_xaxis().set_ticks([])
gca().get_yaxis().set_ticks([])
subplot(4, 4, 13)
imshow(np.flipud(sound_texture.mod_c1[:, :, 3]))
ylabel('Mod C1 (Sub 4)')
subplot(4, 4, 14)
imshow(np.flipud(sound_texture_2.mod_c1[:, :, 3]))
ylabel('Mod C1 (Sub 4)')
gca().get_yaxis().set_ticks([])
subplot(4, 4, 15)
imshow(np.flipud(sound_texture.mod_c1[:, :, 4]))
ylabel('Mod C1 (Sub 5)')
gca().get_yaxis().set_ticks([])
subplot(4, 4, 16)
imshow(np.flipud(sound_texture_2.mod_c1[:, :, 4]))
ylabel('Mod C1 (Sub 5)')
gca().get_yaxis().set_ticks([])

show()
