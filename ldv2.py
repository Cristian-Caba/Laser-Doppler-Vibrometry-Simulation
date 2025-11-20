import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Parameters
INPUT_FILENAME = "test01_20s.wav"           
OUT_FILENAME = "simulated.wav"
TARGET_SR = 22050

# Window modes (frequency Hz, Q factor, relative gain)
# MODES = [
#     (250.0, 30.0, 1.0),
#     (600.0, 50.0, 0.8),
#     (1400.0, 80.0, 0.6),
#     (3000.0, 40.0, 0.3)
# ]
MODES = [(45.0, 25.0, 1.0),    
    (120.0, 40.0, 0.9),   
    (280.0, 60.0, 0.7),     
    (450.0, 50.0, 0.5),   
    (850.0, 80.0, 0.3)]

IMPULSE_LEN_S = 0.2                   # length of window impulse response (s)
NOISE_LEVEL = 1e-4                    # additive noise level (relative)
DOPPLER_GAIN = 3.5                  # scale factor of output
ACOUSTIC_TO_DISP_GAIN = 1e-7           # meters per Pascal (tunable, sets displacement magnitude)
SPECKLE_AMPLITUDE = 0.02             # speckle intensity
LP_CUTOFF_APPROX = 4000            # lp cutoff, simulating real ldv
# ----------------------------------------------------------

def ensure_mono(arr):
    if arr.ndim == 1:
        return arr
    return np.mean(arr, axis=1)

def resample_linear(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x
    duration = x.shape[0] / sr_in
    n_out = int(round(duration * sr_out))
    if n_out < 1:
        return np.zeros(0, dtype=np.float32)
    t_old = np.linspace(0.0, duration, x.shape[0], endpoint=False)
    t_new = np.linspace(0.0, duration, n_out, endpoint=False)
    return np.interp(t_new, t_old, x).astype(np.float32)

def make_window_impulse_response(sr, length_s, modes):
    t = np.linspace(0, length_s, int(np.ceil(length_s * sr)), endpoint=False)
    h = np.zeros_like(t)
    for f0, Q, gain in modes:
        # damping time tau approximated from Q: tau = Q / (pi * f0)
        tau = Q / (np.pi * f0 + 1e-12)
        envelope = np.exp(-t / tau)
        h += gain * envelope * np.sin(2 * np.pi * f0 * t)
    # gentle fade at the end
    fade_len = int(0.01 * sr)
    if 0 < fade_len < h.size:
        h[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)
    # normalize energy so relative gains control the overall shape
    norm = np.sqrt(np.sum(h**2) + 1e-20)
    if norm > 0:
        h /= norm
    return h, t

def save_float32_wav(path, data, sr):
    data_clipped = np.clip(data, -1.0, 1.0).astype(np.float32)
    wavfile.write(path, sr, data_clipped)




if os.path.exists(INPUT_FILENAME):
    try:
        sr_in, wav = wavfile.read(INPUT_FILENAME)
    
        if wav.dtype == np.int16:
            wav = wav.astype(np.float32) / 32768.0
        elif wav.dtype == np.int32:
            wav = wav.astype(np.float32) / (2**31)
        elif wav.dtype in (np.float32, np.float64):
            wav = wav.astype(np.float32)
        else:
            wav = wav.astype(np.float32)
        wav = ensure_mono(wav)
        audio = resample_linear(wav, sr_in, TARGET_SR)
        print(f"Loaded '{INPUT_FILENAME}' (SR {sr_in}) -> resampled to {TARGET_SR} Hz")
    except Exception as e:
        print("No file", e)


sr = TARGET_SR
dt = 1.0 / sr
t_audio = np.arange(audio.size) / sr

pressure = audio * 0.02

h, h_t = make_window_impulse_response(sr, IMPULSE_LEN_S, MODES)

displacement = np.convolve(pressure, h, mode='full') * ACOUSTIC_TO_DISP_GAIN
displacement = displacement[:pressure.size]

velocity = np.concatenate(([0.0], np.diff(displacement))) / dt

vib = DOPPLER_GAIN * velocity

vib += NOISE_LEVEL * np.std(vib) * np.random.randn(vib.size)
vib *= (1.0 + SPECKLE_AMPLITUDE * np.random.randn(vib.size))

lp_len = max(1, int(sr / LP_CUTOFF_APPROX))
vib = np.convolve(vib, np.ones(lp_len)/lp_len, mode='same')


if np.max(np.abs(vib)) > 0:
    vib_norm = vib / (1.05 * np.max(np.abs(vib)))
else:
    vib_norm = vib


save_float32_wav(OUT_FILENAME, vib_norm, sr)
print(f"Saved simulated LDV output to: {OUT_FILENAME}")


plt.figure(figsize=(12, 8))


plt.subplot(2, 1, 1)
plt.title("Waveform Comparison: Source Room vs. LDV Output")


original_norm = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio

plt.plot(t_audio, original_norm, label='Original Audio (Source)', alpha=0.6)
plt.plot(t_audio, vib_norm, label='Simulated LDV Signal (Glass)', alpha=0.7)

plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Normalized Amplitude")
plt.grid(True, alpha=0.3)


plt.subplot(2, 2, 3)
plt.title("Original Spectrogram")
plt.specgram(original_norm, NFFT=1024, Fs=sr, noverlap=512)
plt.ylabel("Frequency (Hz)")
plt.xlabel("Time (s)")


plt.subplot(2, 2, 4)
plt.title("LDV Output Spectrogram (Glass Response)")
plt.specgram(vib_norm, NFFT=1024, Fs=sr, noverlap=512)
plt.xlabel("Time (s)")
plt.yticks([])

plt.tight_layout()
plt.show()
# Note on methodology: Code development employed generative AI programming for
# initial implementation, followed by manual refinement to guarantee
# physical fidelity in the LDV simulation outputs.

