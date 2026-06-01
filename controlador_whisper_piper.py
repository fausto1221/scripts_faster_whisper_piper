import os 
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" # deshabilita la advertencia de symlinks de HuggingFace   

"""
En Windows, HuggingFace requiere el uso de los enlaces sibolicos, los cuales no se crean automaticamente en carpetas
generadas por el usuario, en este caso, no puede crear una carpeta en webots, ya que no tiene los permisos suficiente o el programa no
los gestiona del todo, esto no quiere decir que no funcione, simplemente evita que salga un error constante en la consola
"""                                                        

from controller import Robot ### unicamente dentro de Webots
import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel
import time 
import wave
from io import BytesIO
from piper.voice import PiperVoice

##### CONFIGURACIÓN PIPER TTS #####
VOICE_NAME = "es_MX-claude-high"

MODELS_DIR = os.path.join(os.path.expanduser("~"), "piper_voices")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, f"{VOICE_NAME}.onnx")
CONFIG_PATH = os.path.join(MODELS_DIR, f"{VOICE_NAME}.onnx.json")

def descargar_voz_si_no_existe():
    if os.path.exists(MODEL_PATH) and os.path.exists(CONFIG_PATH):
        print(f"Voz Piper {VOICE_NAME} ya está lista.")
        return
    print(f"Descargando voz Piper {VOICE_NAME}...")
    try:
        import subprocess
        subprocess.run([
            "python", "-m", "piper.download_voices",
            VOICE_NAME,
            "--download-dir", MODELS_DIR
        ], check=True)
        print("Voz Piper descargada.")
    except Exception as e:
        print("Error al descargar Piper:", e)
        exit(1)

def hablar(texto: str):
### Reproduce texto con Piper###
    voice = PiperVoice.load(MODEL_PATH, CONFIG_PATH)
    
    wav_io = BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(voice.config.sample_rate)
        voice.synthesize_wav(texto, wav_file)
    
    wav_io.seek(0)
    with wave.open(wav_io, "rb") as wav_file:
        audio_data = wav_file.readframes(wav_file.getnframes())
        sample_rate = wav_file.getframerate()
    
    audio = np.frombuffer(audio_data, dtype=np.int16)
    sd.play(audio, samplerate=sample_rate)
    sd.wait()

##### CONFIGURACIÓN FASTER-WHISPER #####
samplerate = 16000
block_duration = 0.5
chunk_duration = 2.0
channels = 1

frames_per_block = int(samplerate * block_duration)
frames_per_chunk = int(samplerate * chunk_duration)

audio_queue = queue.Queue()
audio_buffer = []
command_queue = queue.Queue()

print("Cargando Whisper large-v3 en CUDA...")
model = WhisperModel("large-v3", device="cuda", compute_type="float32")

###### ROBOT UR5 #####
robot = Robot()
timestep = int(robot.getBasicTimeStep())

joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
               "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

motors = []
for name in joint_names:
    motor = robot.getDevice(name)
    motor.setVelocity(2.5)
    motors.append(motor)

HOME = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]

print("Moviendo a HOME inicial...")
for i, m in enumerate(motors):
    m.setPosition(HOME[i])
time.sleep(2.0)

##### PROGRAMACIÓN/AJUSTE DE SECUENCIAS #####

def move_to_joints(target_angles, duration=4.0):
    steps = int(duration * 1000 / timestep)
    current = [m.getTargetPosition() for m in motors]
    for step in range(steps + 1):
        t = step / steps
        for i in range(6):
            pos = current[i] + t * (target_angles[i] - current[i])
            motors[i].setPosition(pos)
        robot.step(timestep)
    for i, angle in enumerate(target_angles):
        motors[i].setPosition(angle)

def sequence_1():  # Frente - Derecha (baja más y gira más)
    print("---Secuencia 1: Tomar de frente y colocar a la derecha")
    hablar("Ejecutando simulación uno: tomando de frente y colocando a la derecha.")
    
    move_to_joints([0.0, -2.65, -1.35, -1.4, 0.0, 0.0], 3.2)   # Bajar más de frente
    time.sleep(0.8)
    move_to_joints([0.0, -1.45, 0.9, -1.6, 0.0, 0.0], 2.6)     # Levantar pieza
    move_to_joints([1.55, -1.45, 0.9, -1.6, 0.0, 0.0], 3.4)    # Girar MÁS a la derecha
    move_to_joints([1.55, -2.55, -1.1, -1.7, 0.0, 0.0], 2.9)   # Bajar más para colocar
    time.sleep(0.9)
    move_to_joints(HOME, 4.5)
    hablar("¡Simulación uno completada con éxito!.")

def sequence_2():  # Frente - Izquierda (baja más y gira más)
    print("---Secuencia 2: Tomar de frente y colocar a la izquierda")
    hablar("Ejecutando simulación dos: tomando de frente y colocando a la izquierda.")
    
    move_to_joints([0.0, -2.65, -1.35, -1.4, 0.0, 0.0], 3.2)   # Bajar más de frente
    time.sleep(0.8)
    move_to_joints([0.0, -1.45, 0.9, -1.6, 0.0, 0.0], 2.6)     # Levantar
    move_to_joints([-1.55, -1.45, 0.9, -1.6, 0.0, 0.0], 3.4)   # Girar MÁS a la izquierda
    move_to_joints([-1.55, -2.55, -1.1, -1.7, 0.0, 0.0], 2.9)  # Bajar más para colocar
    time.sleep(0.9)
    move_to_joints(HOME, 4.5)
    hablar("¡Simulación dos completada con éxito!.")

def sequence_3():  # Atrás - Derecha (baja más y gira más)
    print("---Secuencia 3: Tomar de atrás y colocar a la derecha")
    hablar("Ejecutando simulación tres: tomando de atrás y colocando a la derecha.")
    
    move_to_joints([0.0, -1.7, 2.1, -0.5, 0.0, 0.0], 3.3)      # Posición atrás más baja
    time.sleep(0.8)
    move_to_joints([0.0, -1.35, 1.25, -0.9, 0.0, 0.0], 2.7)    # Levantar de atrás
    move_to_joints([1.65, -1.5, 0.95, -1.65, 0.0, 0.0], 3.6)   # Girar MÁS a la derecha
    move_to_joints([1.65, -2.6, -1.05, -1.75, 0.0, 0.0], 3.0)  # Bajar más para colocar
    time.sleep(0.9)
    move_to_joints(HOME, 4.8)
    hablar("¡Simulación tres completada con éxito!.")

def sequence_4():  # Atrás - Izquierda (baja más y gira más)
    print("---Secuencia 4: Tomar de atrás y colocar a la izquierda")
    hablar("Ejecutando simulación cuatro: tomando de atrás y colocando a la izquierda.")
    
    move_to_joints([0.0, -1.7, 2.1, -0.5, 0.0, 0.0], 3.3)      # Posición atrás más baja
    time.sleep(0.8)
    move_to_joints([0.0, -1.35, 1.25, -0.9, 0.0, 0.0], 2.7)    # Levantar de atrás
    move_to_joints([-1.65, -1.5, 0.95, -1.65, 0.0, 0.0], 3.6)  # Girar MÁS a la izquierda
    move_to_joints([-1.65, -2.6, -1.05, -1.75, 0.0, 0.0], 3.0) # Bajar más para colocar
    time.sleep(0.9)
    move_to_joints(HOME, 4.8)
    hablar("¡Simulación cuatro completada con éxito!.")

# ==================== PROCESADOR DE COMANDOS ====================
def process_voice_command(cmd):
    cmd = cmd.lower().strip()
    print(f"Comando recibido: {cmd}")

    if any(w in cmd for w in ["home", "reposo", "inicio", "reset"]):
        hablar("Entendido, regresando a posición home.")
        for i, m in enumerate(motors):
            m.setPosition(HOME[i])
        return

    if any(w in cmd for w in ["parar", "stop", "detener", "quieto"]):
        hablar("Entendido, deteniendo el movimiento.")
        for m in motors:
            m.setPosition(m.getPosition())
        return

    if any(w in cmd for w in ["simulación 1", "simulacion 1", "simulacion uno", "secuencia uno", "secuencia 1"]):
        sequence_1()
        return

    if any(w in cmd for w in ["simulación 2", "simulacion 2", "simulacion dos", "secuencia dos", "secuencia 2"]):
        sequence_2()
        return

    if any(w in cmd for w in ["simulación 3", "simulacion 3", "simulacion tres", "secuencia tres", "secuencia 3"]):
        sequence_3()
        return

    if any(w in cmd for w in ["simulación 4", "simulacion 4", "simulacion cuatro", "secuencia cuatro", "secuencia 4"]):
        sequence_4()
        return

    # Movimientos manuales simples
    """En esta parte encontramos los comandos individuales, se pueden asignar diferentes palabras o el movimiento de los motores"""
    
    delta = 0.45
    if any(x in cmd for x in ["izquierda", "rotar izquierda"]) and "hombro" in cmd:
        hablar("Girando hombro hacia la izquierda.")
        motors[0].setPosition(motors[0].getTargetPosition() + delta)
    elif any(x in cmd for x in ["derecha", "rotar derecha"]) and "hombro" in cmd:
        hablar("Girando hombro hacia la derecha.")
        motors[0].setPosition(motors[0].getTargetPosition() - delta)
    elif any(w in cmd for w in ["levantar", "arriba", "subir"]):
        hablar("Levantando el brazo.")
        motors[1].setPosition(motors[1].getTargetPosition() - delta)
    elif any(w in cmd for w in ["bajar", "abajo"]):
        hablar("Bajando el brazo.")
        motors[1].setPosition(motors[1].getTargetPosition() + delta)
    else: 
        print("¡¡¡Comando no reconocido!!!")
        
# ==================== THREADS ====================
def audio_callback(indata, frames, time, status):
    if status: print(status)
    audio_queue.put(indata.copy())

def recorder():
    with sd.InputStream(samplerate=samplerate, channels=channels,
                        callback=audio_callback, blocksize=frames_per_block):
        print("...Escuchando micrófono...")
        while True:
            sd.sleep(100)

def transcriber():
    global audio_buffer
    while True:
        block = audio_queue.get()
        audio_buffer.append(block)
        total_frames = sum(len(b) for b in audio_buffer)
        
        if total_frames >= frames_per_chunk:
            audio_data = np.concatenate(audio_buffer)[:frames_per_chunk]
            audio_buffer = []
            audio_data = audio_data.flatten().astype(np.float32)

            segments, _ = model.transcribe(
                audio_data, language="es", beam_size=1,
                vad_filter=True, vad_parameters=dict(min_silence_duration_ms=600)
            )

            for segment in segments:
                texto = segment.text.strip()
                if texto:
                    print(f"Transcrito: {texto}")
                    command_queue.put(texto)

# ==================== INICIO ====================
if __name__ == "__main__":
    descargar_voz_si_no_existe()
    
    print("\n=== UR5 Control por Voz con Piper TTS (Version Ajustada) ===\n")
    print("Comandos disponibles:")
    print("  simulación 1 = frente >> derecha")
    print("  simulación 2 = frente >> izquierda")
    print("  simulación 3 = atrás >> derecha")
    print("  simulación 4 = atrás >> izquierda")
    print("  home / parar")
    
    threading.Thread(target=recorder, daemon=True).start()
    threading.Thread(target=transcriber, daemon=True).start()

    while robot.step(timestep) != -1:
        if not command_queue.empty():
            cmd = command_queue.get()
            process_voice_command(cmd)