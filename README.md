# DailySignals

Repositorio personal de experimentos y pequeñas implementaciones
relacionadas con **Signals and Systems** (Oppenheim).

Basado en el curso del MIT OpenCourseWare.
Objetivo: práctica, exploración y registro personal.

## Contenido
- Señales sinusoidales
- Exponenciales complejas
- Señales elementales (unit step, impulse)
- Operaciones sobre señales
- Representación en tiempo discreto y continuo
- Convolución

## Setup / Requisitos

Este proyecto utiliza **Python 3** y librerías estándar para procesamiento digital de señales (DSP).

---

### 1. Crear entorno virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 2. Instalar dependencias
```bash
pip install numpy scipy soundfile librosa matplotlib
```

### 3. Verificar Instalación
Puedes comprobar que todo funciona ejecutando:
```bash
python -c "import numpy, scipy, librosa, soundfile, matplotlib"
```
Si no hay errores, el entorno está listo.

---

## Quick Start

```bash
git clone <repo>
cd DailySignals
python -m venv venv
source venv/bin/activate
pip install numpy scipy soundfile librosa matplotlib
python experiments/experiment3.py
```

## Estructura del proyecto

```text
DailySignals/
├── experiments/
│   └── programa.py
├── wavs/
│   └── experiment3/
│       └── audio.wav
├── plots/
|   └── figura.png
```

---

## Ejemplo: Convolución Lo-Fi

Este experimento aplica convolución a una señal de audio:

- Se construye una respuesta al impulso artificial
- Se realiza convolución mediante FFT
- Se añaden efectos (hiss, crackle, wow/flutter)
- Se mezcla señal dry/wet

Conceptualmente:

y[n] = x[n] * h[n]

---

## Notas

- Proyecto con fines educativos y experimentales
- Basado en conceptos de sistemas LTI
- Se recomienda trabajar con fragmentos cortos de audio

---