from torch import exp, arange, empty, cfloat
import torchaudio
import matplotlib.pyplot as plt
import celluloid
def fft(x,N, s):
    B = x. shape [0]
    if N == 1: return x[:, :1]
    X = empty(B, N, dtype=cfloat).view(B, 2, -1)
    X[:, 0] = fft (x, N // 2, 2 * s)
    X[:, 1] = fft(x [:, s:], N // 2, 2 * s)
    q = exp(-((2j * 3.1415) / N) * arange(N // 2))*X[:, 1]
    X[:, 0], X[:, 1] = X[:, 0] + q, X[:, 0] - q
    return X.view(B, N)
# Sound Processing

sound = torchaudio. load("Sia.mp3")[0][0, :]
STEP = 1000; WINDOW = 512; TIME = 800
i = arange(WINDOW) [None, :] + STEP * arange(TIME)[:, None]
out = (10 * fft(sound[i], WINDOW, 1).abs() .square().log10())
out = out.view(-1, 16, 32) .mean(-1)
# Make Video

fig, ax = plt.subplots(); camera = celluloid.Camera(fig)
for i in range(TIME):
    ax.bar(arange(16), (-out[i]).relu(), color="purple" )
    camera.snap()
animation = camera.animate()
animation.save('ani.mp4')