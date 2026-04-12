from __future__ import annotations
import numpy as np
import soundfile as sf
import librosa
from scipy import signal
from pathlib import Path

""" CONVOLUCION SOBRE UNA CANCION DE JUICE WRLD. USED TO * Lo-Fi Noise (Cassette Tape).

.........................................................*#@@*.:.......:.:.:..-.:.:.::..............
.#*++#*++*#****+#**#*****###%##%#*%*##+##*#**%#*##%%@@@@@@...@#*=-.-=%@#%%*%%##*%#%%##########%%%##.
.*=+**+***#*#+*##+##*#*###*##*%*##%*##%#%*#####%%#%%%@=...:@@@@@@@@@@@+-=+%###%#%#%#%#%#%%##%#%####.
.+%*+*#**#**#**#**#*#%**%##*%**%#*#%#*#%#%##*%###*##@@..:@@........@..@@@*=:=*@%%*@#@%###%%%##%%%%#.
.*+*****+*#*#**#**#**##+#*%##%#%*#%*#%####%#%%*##%@@@..@..............@@.@@@@*++@%%%#%%%@#%%%%%#*%%.
.***#+**#*+%**#+*+#***#%%#*%#%#*#%%*#%#%#######@#%@@......@@@@@@@@@@@@#:@-..:*@@%##@###%###@%%##%%%.
.*+****#+%*#+%##+##*#*%*#%%#%%#%*###%##%#@@%#%%@@@@...@......*#%%%%@%#%##@@@@##**@#%#@%#%@*#*#@@%%%.
.#+#***#=%*#**#**##****#**#**##%##%@*%###@%@@@@@@......:.@@@@@@@#***#@@%%%@%@%%%@%%%%#@##%@%@%###@@.
.**=#+****+*#+#*%**+#**@#%%%%%@@@@#*##@@@@%.......:......@@@@@@#@@@@*=+#%%@##%%#%#%@#@*%##%%##%##%%.
.*#+******#*##*****#%#*##%@@@@@.#@@@@@@+@......#@@.....@@@%..+..@...@@*%##%%%@@%#%*%%%%%@%%%#%##@%@.
.***+#***########**##*###*+..%......:....=+.+...:.@@............:@@+%#@##%%#%%%@#@%%@@%%%%#%@%%%###.
.**********#*#**%####*#@@:-@.......-.#...-:........@@.......@-#.....@=@#%%%%%#%%%%%%%%%#%#%%#@##%%%.
.#*++#**#***#+***#**%*@@..@@=..:.@.......+%.......@.#.*.:.....#.@@.@%.##@@%@%#@%#%%%%%@#%@%#@##%#%%.
.*=###*#+*****#*#*%#%@@...@%-...+-........@......@@.@....:....#.....@.+####@%#%%@#%%#%@%@%%@%@#%%%#.
.***+#+##*#####**#+@@@...@@..+...%........%%.......@.@...=.....%*...@@@:+#@%%%%%%%@@%@@%@#%%*%@%%#@.
.***#*++#*****#**%@@.....@@...@--....:#@@@@=....@..@.@@.........+@...=+@@%*#@%#%%%@%@%%%%%%@@%#@%%@.
.+*****#***#*#*##@@.......@...+:.:........@@*@@@@-..@:.@@%.......@=@@-...@@=*%%@%#%%%#@%##@%#@%%%@%.
.*****#*+#+***#%*@....%..@@...@@.=*%#@-....@#.%@#@@...@.@@@@@.....@.@@@@.@@@#%%%%#%%%##%@@%%%%%@%%%.
.***+*#**%*##*#@@@...@...@@...-@..-.%.++..@#.@@@@@@@...@...@@@@....@@@@..@==*%##@%@%%#@#%%#%%%###%#:
.***#+#*###***@+@...........@.@#..-:@@@@...@+@@:...@..+@@@@@@.%@...=@@@.@@@%@#%%%##@%%@%%%%%%%%%%%@.
.+**+*##**#+#@@*..:*..:..:....@@@.@@@=#@@@@@@@@@@%@@.*@@@@@@@@..@=.#@#.@@##%%@%%%%@%%@%%#@%@*#%#%#%.
.******#**#*#@....@....::..@-@..@......@*.=.@@@@@@@...**@@@@@@@@*@..@.@@%%%%#%###%%#%#@%#%#%%%#%%@@.
.****+**+%**@@...@-%...:.-.@....@...........@@@@*....-#*@@@@.@@@@@@.@@%@#%%#%%%@%#%###%%%%%%##%#%*%.
.**+**#+#*#@@....@.:...::........:...........@@@%@...@..@.....@@@@#..@:*%%#%@%%#%#@#@%####%#####%%%.
.#**#*#+**@@....+@....-.::-::.#@@#@@@@@.....#@:@@@@@....%=..@@@=@@@%..@.*#%####%%%%%#%#%%##%%%##%#%.
.**#**+*##@..=..#@.@..:::...::....@@@@@.....@*@@@@@@@@@@@@@@@@@@:.=@*.@@:%%%##@%##@#@#%%@%#%*##%**#.
.*****#***@......*.@..:.::.:-........%#.-...@@@@@@@@@@@@@@@@@@@@@@--=..@=+#%####%#%@*%%#%%#####%%%%.
.***#*****@@@...#@....::::::-.:.:+=@%..#............@@@@@@@@@@@@@@%*%+.%@.:*%#%##%##%%#%#%*#####*@*.
.**+****#+#%@....@....:::.-::.:.....@#%...@....@@@@..@@@@@@@@@@@@@@@@*.@@@@@+*%#%%#%%#%##%%#%%%#*##.
.##*#*#+*#*#@.=..%..:.:...:.::..=@@#---...@..*.@...@@@@@@@@@@@@@@@@@@+.:.@.@=########%*%#%%#%#*%##%.
.**##*****+*@.:..=....-:-::::..@@#+=+-.........@..@@@..@@@@@@@@@@@@@*.=..@@@.*%#%%%###########%####.
.**#*#****#*@.@..@.......-:-:.............@..@+-@@@@@@@@.@@@@@@@@@-@%+@.@@@@######%#%##%###%#%%%%#%.
.*%*###***+#@@@@.=@.......:=.:.:...+=#@@@=..%...........@@@@@@@@=:.@@*@.*@@@#+%####*#%#####%#%#%##%.
.+**#**+*+*+**%@@@@........-+-.:::.=.:#-...:%*=..@%.......@@@@@@@+@@%%@*....+##%%%*#####%#####%####.
.*+*+##+#**#+*++++@@........=:.-.::...:....=.*.+@@@@@@@@@@@@@#@@@**%@@@@.@@@##*###%#*###*##*#**#%*#.
.#*#+**#+**+******@@.........+.::.:.:.....-=@..@@@=....@@@@@+%.-#%+@*@#@.@**##%#**%#%#%###%*###%#%%.
.**#*##**#*******+@......-...=::.::::.:..@..................+@@@%-.*@*@%.@%%*##%*%##**#%*##%#%#**#*.
.*+*#*+*++*#*****@@.@.........-::-:.::.::...:.:..@%@@%@@@@@@+.=--*@#@@++.@*#%%#%*%*%#%%*%**##**###%.
.+*+*+**+*+**#*+*@..@........::-:.:::::.::::-.-#@.=@@@@@@@@@@@=@+#-@-..:.@%#*#####*##*##%*###*#**#*.
.*++**+*+*+#**#**@..@%:..........::.:::-::-:::.*.=.+...@@@@@@@@@@*@-%*=..@@%%##*##%#####*#%*#*%**#*.
.*++**+***++**++#@@@@@@..-.....@..::-:.:::-:....-.@@@.:..@.@@@@#:.:.....::@##*#%****%*####****+*#**.
.+**+=**++*+**+******#@......@@.....:::-:.:.:::.:-..@...@....=.:......=:*.@###**#####+#*##*%*######.
.*=*#++*+*+*********++@.::.@.@@@@@..:-::::::-.:.#..=...+...........-:=@-@.@##%##****######*###***#*.
.*+*++*+=#**+=*+#+****@...=...@*%@.:::.:.:::::.-.*.:.:.........+:.::.+-#@.*@#*+##*#*******##*%+*#**.
.++**=+++*+**+******+#@.@.@.@%@*+@.-:-:-:.::..:......--.:::::.*.##++@::#=..@#*#*#*#****##**#**#****.
.*+=++*++*-+****+*+*#*@...@@@@**+@.-:.::-:::-.::--=-.:.-.........:.=.-@@*..@@#*#*+***#+#*#**+#*+**#.
.+=+*++++***=*+*+*++*+@..@@#%#***@.-.::..:..:.:.::.-.-:........*=*=*#*++.-=.@@@#****###+*##*+*+*+**.
.**==+++++**#*+*#**+**@...@++***@@.-::::.:=..:.:--:.:.:.....+@@%#++=##@@@=@:..@@*#*#*#***#++***#+**.
.+=+**=*++++*+*****+#*@...@*#*##@..-..:::.::--::::-......%@@@%#+*%..@*@%@*@@...*.:**+**+*#+*#+*+*+*.
.+++=+++++*=**++**+#**@.@.@#+**+@..::-::.-..:.::.::::.=+###:#:@@.@=@@@@.@@.....@@...-*+*+++*++**+#+.
.=*==+*=+++*+*+++*+++*@@@.@@*%**@.:::::-:.:::.:.::..:*+%=-@::.:.+-*@-#@@@...=-.@@@@@...:#+*****++++.
.+++++++++*+++***=****#%@..@*=#@@.::.-..:::::.::::.:...-@@=@+*%#*@@@@@....=.....:@@@@@@...++=+++*++.
.=*+++++*=*++#++***+*#**@@@@@@@@...::.:.:....:......=@...@#@@@@*@......=....=+-..%@@@@@@@....==++++.
.=+=+++++=*+++++=+*#+*#@@@@.......:::-::::-:::.::................:......-@......+.@@@@@@@@@@........
......................................................................................@@@@@@@@@@@@..

"""
#Funciones

def normalizar_audio(x: np.ndarray, peak: float= 0.98) -> np.ndarray:
    """Normaliza el audio a un target peak evitando la division por cero."""
    max_val = np.max(np.abs(x))
    if max_val < 1e-12: # Evitar división por cero
        return x.copy()
    return peak * x / max_val

def stereo_a_mono(x: np.ndarray) -> np.ndarray:
    """Convierte un audio estéreo a mono si es necesario."""
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)

def make_lofi_ir(
        sr: int,
        duration_sec: float = 0.35,
        decay: float = 10.0,
        pre_delay_ms: float = 12.0,
        seed: int = 7,
) -> np.ndarray:
    """
    Crea una respuesta al impulso artificial con las siguientes características:
    -Delay corto
    -Decrecimiento exponencial
    -Cola pequeña y ruidosa
    -Algunas reflexiones tempranas para dar carácter
    """
    rng = np.random.default_rng(seed)

    n = int(duration_sec * sr)
    if n <= 8:
        raise ValueError("Duración del IR demasiado corta para el sample rate dado.")

    t = np.arange(n) / sr
    ir = np.zeros(n, dtype=np.float64)

    pre_delay = int(sr * pre_delay_ms / 1000.0)
    pre_delay = min(pre_delay, n - 1)

    # Direct impulse
    ir[pre_delay] = 1.0

    # Exponentially decaying noisy tail
    tail_len = n - pre_delay
    env = np.exp(-decay * np.linspace(0, 1, tail_len))
    noise_tail = rng.normal(0.0, 1.0, tail_len) * env * 0.15
    ir[pre_delay:] += noise_tail

    # Early reflections
    reflection_times_ms = [22, 37, 61, 88]
    reflection_gains = [0.45, 0.28, 0.18, 0.10]
    for ms, g in zip(reflection_times_ms, reflection_gains):
        idx = pre_delay + int(sr * ms / 1000.0)
        if idx < n:
            ir[idx] += g

    # Make it darker / more lofi with a low-pass
    b, a = signal.butter(2, 2500 / (sr / 2), btype="low")
    ir = signal.lfilter(b, a, ir)

    # Normalize IR energy
    ir /= np.sqrt(np.sum(ir**2) + 1e-12)
    return ir.astype(np.float32)


def add_hiss(x: np.ndarray, amount: float = 0.01, seed: int = 11) -> np.ndarray:
    """Añade ruido de banda ancha."""
    rng = np.random.default_rng(seed)
    hiss = rng.normal(0.0, 1.0, size=x.shape).astype(np.float32)
    return x + amount * hiss


def add_crackle(
    x: np.ndarray,
    sr: int,
    num_clicks_per_sec: float = 4.0,
    max_click_amp: float = 0.08,
    seed: int = 13,
) -> np.ndarray:
    """Añade breves impulsos de crujido similares a los del vinilo."""
    rng = np.random.default_rng(seed)
    y = x.copy()

    total_clicks = int(num_clicks_per_sec * len(x) / sr)
    if total_clicks <= 0:
        return y

    click_positions = rng.integers(0, len(x), size=total_clicks)

    for pos in click_positions:
        width = rng.integers(1, 12)  # very short
        amp = rng.uniform(-max_click_amp, max_click_amp)
        end = min(pos + width, len(y))
        y[pos:end] += amp * np.hanning(end - pos)

    return y


def lowpass_lofi(x: np.ndarray, sr: int, cutoff_hz: float = 4500.0) -> np.ndarray:
    """Aplicar un filtro low-pass suave para dar tono lofi."""
    b, a = signal.butter(3, cutoff_hz / (sr / 2), btype="low")
    return signal.lfilter(b, a, x).astype(np.float32)


def highpass_cleanup(x: np.ndarray, sr: int, cutoff_hz: float = 60.0) -> np.ndarray:
    """Remove very low rumble."""
    b, a = signal.butter(2, cutoff_hz / (sr / 2), btype="high")
    return signal.lfilter(b, a, x).astype(np.float32)


def apply_wow_flutter(
    x: np.ndarray,
    sr: int,
    wow_rate_hz: float = 0.35,
    flutter_rate_hz: float = 6.0,
    wow_depth_ms: float = 3.0,
    flutter_depth_ms: float = 0.6,
) -> np.ndarray:
    """
    Inestabilidad de tono/tiempo tipo cinta mediante lectura de retardo modulado.
    Solo valores pequeños, de lo contrario se distorsiona demasiado.
    """
    n = len(x)
    t = np.arange(n) / sr

    delay_ms = (
        wow_depth_ms * np.sin(2 * np.pi * wow_rate_hz * t)
        + flutter_depth_ms * np.sin(2 * np.pi * flutter_rate_hz * t)
    )
    delay_samples = delay_ms * sr / 1000.0

    read_idx = np.arange(n) - delay_samples
    read_idx = np.clip(read_idx, 0, n - 2)

    i0 = np.floor(read_idx).astype(int)
    frac = read_idx - i0
    y = (1.0 - frac) * x[i0] + frac * x[i0 + 1]
    return y.astype(np.float32)


def wet_dry_mix(dry: np.ndarray, wet: np.ndarray, wet_ratio: float = 0.35) -> np.ndarray:
    """Mezclar señales secas y húmedas."""
    min_len = min(len(dry), len(wet))
    dry = dry[:min_len]
    wet = wet[:min_len]
    return ((1.0 - wet_ratio) * dry + wet_ratio * wet).astype(np.float32)


def process_track(input_path: str, output_path: str) -> None:
    """
    Pipeline de convolución lofi completo.
    """
    in_path = Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {input_path}")

    # Load
    x, sr = librosa.load(in_path, sr=None, mono=False)
    x = x.T if x.ndim == 2 else x  # librosa puede retornar (channels, samples)
    x = stereo_a_mono(x).astype(np.float32)

    # Opcional: trabaje primero en un segmento corto para iterar más rápido durante el desarrollo.
    max_seconds = 45
    x = x[: sr * max_seconds]

    # Cleanup basico para eliminar rumbles y artefactos de baja frecuencia que no queremos amplificar.
    dry = highpass_cleanup(x, sr)
    dry = lowpass_lofi(dry, sr, cutoff_hz=5000.0)

    # Crear IR artificial
    ir = make_lofi_ir(sr, duration_sec=0.4, decay=9.0, pre_delay_ms=10.0)

    # Convolucion
    wet = signal.fftconvolve(dry, ir, mode="full").astype(np.float32)

    # Recortar wet a la longitud original para facilitar la mezcla.
    wet = wet[: len(dry)]

    # Add artifacts
    wet = add_hiss(wet, amount=0.008)
    wet = add_crackle(wet, sr, num_clicks_per_sec=3.5, max_click_amp=0.03)
    wet = apply_wow_flutter(
        wet,
        sr,
        wow_rate_hz=0.28,
        flutter_rate_hz=5.5,
        wow_depth_ms=2.2,
        flutter_depth_ms=0.35,
    )

    # Mezcla final
    y = wet_dry_mix(dry, wet, wet_ratio=0.42)
    y = normalizar_audio(y, peak=0.98)

    # Guardar resultado
    sf.write(output_path, y, sr)
    print(f"Archivo guardado en: {output_path}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent

    input_path = BASE_DIR.parent / "wavs" / "experiment3" / "jwusedto.wav"
    output_path = BASE_DIR.parent / "wavs" / "experiment3" / "usedto_convolved.wav"

    process_track(str(input_path), str(output_path))