import os
import numpy as np
import sounddevice as sd
import wave
from io import BytesIO
from piper.voice import PiperVoice


VOICE_NAME = "es_MX-claude-high"

# Directorio donde se almacenan los modelos descargados.
# Se crea automáticamente si no existe.
MODELS_DIR = os.path.join(os.path.expanduser("~"), "piper_voices")
os.makedirs(MODELS_DIR, exist_ok=True)


MODEL_PATH = os.path.join(MODELS_DIR, f"{VOICE_NAME}.onnx")    # Ruta al archivo del modelo neuronal (formato ONNX).



CONFIG_PATH = os.path.join(MODELS_DIR, f"{VOICE_NAME}.onnx.json") # Ruta al archivo de configuración JSON del modelo


FRASE_FIJA = "Esta es una prueba de Piper." #####Texto que se sintetizará y reproducirá cada vez que se ejecuta el script.##########




def descargar_voz_si_no_existe():
    
    # Si ambos archivos ya están presentes, no hay nada que descargar.
    if os.path.exists(MODEL_PATH) and os.path.exists(CONFIG_PATH):
        print(f"La voz: {VOICE_NAME} ya está lista.\n")
        return

    print(f"Descargando voz {VOICE_NAME}...\n")
    try:
        import subprocess
        # Invoca el descargador oficial de Piper como subproceso.
        # --download-dir indica dónde guardar los archivos descargados.
        subprocess.run([
            "python", "-m", "piper.download_voices",
            VOICE_NAME,
            "--download-dir", MODELS_DIR
        ], check=True)  # check=True lanza CalledProcessError si falla
        print("Descarga completada.\n")
    except Exception as e:
        # Si la descarga falla, muestra instrucciones manuales y termina.
        print("Error descargando la voz:", e)
        print(
            f"Prueba manualmente:\n"
            f"python -m piper.download_voices {VOICE_NAME} "
            f"--download-dir \"{MODELS_DIR}\""
        )
        exit(1)


def reproducir_prueba():

    print(">Cargando modelo...")
    # Carga el modelo ONNX y su configuración JSON 
    voice = PiperVoice.load(MODEL_PATH, CONFIG_PATH)

    print(">Generando audio...")

    # Buffer en memoria que actuará como archivo WAV temporal
    
    wav_io = BytesIO() #  Buffer en memoria que actuará como archivo WAV temporal. se usa BytesIO evita escribir archivos temporales en el disco

    # Abre el buffer como archivo WAV en modo escritura para configurar
    # los parámetros del stream y recibir las muestras sintetizadas
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(1)                            # Mono (1 canal)
        wav_file.setsampwidth(2)                               # 16 bits = 2 bytes por muestra
        wav_file.setframerate(voice.config.sample_rate)             # Frecuencia del modelo

        
        voice.synthesize_wav(FRASE_FIJA, wav_file) # Sintetiza el texto y escribe los frames PCM en el archivo WAV

    
    wav_io.seek(0)
    with wave.open(wav_io, "rb") as wav_file:
        
        audio_data = wav_file.readframes(wav_file.getnframes()) # Lee todos los frames en crudo (bytes PCM)
       
        sample_rate = wav_file.getframerate()  # Guarda la frecuencia de muestreo para pasarla a sounddevice

    
    
    audio = np.frombuffer(audio_data, dtype=np.int16) # Convierte el buffer de bytes a un arreglo numpy de enteros de 16 bits

    print(">Reproduciendo: Esta es una prueba de Piper.")

    
    sd.play(audio, samplerate=sample_rate) # Envía el arreglo al dispositivo de audio predeterminado del sistema

    
    sd.wait()

    print(">Reproducción terminada.\n") # Bloquea la ejecución hasta que la reproducción termine completamente


if __name__ == "__main__":
    print(" PRUEBA DE PIPER\n")

    descargar_voz_si_no_existe()     # Paso 1, se asegura que el modelo esté disponible en disco


    
    print("Iniciando reproducción automática...\n") # Paso 2, sintetiza y reproduce la frase de prueba 
    reproducir_prueba()
