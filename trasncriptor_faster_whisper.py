import sounddevice as sd                    # Captura de audio desde el micrófono
import numpy as np         
import queue                                # Cola "thread-safe" para comunicar los dos hilos
import threading                                # Ejecución concurrente de recorder y transcriber
from faster_whisper import WhisperModel  
 

# Parámetros de configuración
 
 
SAMPLERATE = 16_000         #  frecuencia de muestreo requerida por Whisper
BLOCK_DURATION = 0.5            # Segundos por bloque que entrega sounddevice al callback
CHUNK_DURATION = 2              # Segundos de audio acumulado antes de llamar a Whisper
CHANNELS = 1                # Mono; Whisper espera una sola pista de audio
 
# Número de muestras equivalentes a cada duración
FRAMES_PER_BLOCK = int(SAMPLERATE * BLOCK_DURATION)   
FRAMES_PER_CHUNK = int(SAMPLERATE * CHUNK_DURATION)  
 

# Estado compartido entre hilos 

 
audio_queue = queue.Queue()             # Bloques de audio "raw" producidos por recorder
audio_buffer = []                   # Acumulador temporal en el hilo para la transcripscion, osea el transcriber
 

#### Carga del modelo ####

"""
En esta parte, device tiene que ser cuda para aprovechar la GPU, si se va a correr con CPU, el valor de device debera ser "cpu"
El valor que recomienda el proyecto es float16, se trabajo con float 32, pero el otro no debe dar problemas e incluso debe ser mas rapido
pueden cambiar este valor segun lo vean conveniente
"""

model = WhisperModel("large-v3", device="cuda", compute_type="float32") 
 

 
def audio_callback(indata, frames, time, status):
    """
    Callback es llamado automáticamente por sounddevice en cada "BLOCK_DURATION" (segundos).
 

    Funcionamiento:
    Copia el array para no depender del búfer interno de sounddevice
    y lo pone en la cola para que el transcriber lo use y se "consuma".
    """
    if status:
        print(status)                  # Avisar sobre underflows/overflows si ocurren
    audio_queue.put(indata.copy())     
 
 
def recorder():
    """
    Abre el stream de entrada de audio y lo mantiene activo indefinidamente,
    corre en un hilo un daemon, por lo que se cierra automáticamente cuando el proceso principal termina (en este caso. al pulsar Ctrl+C).
    """
    with sd.InputStream(
        samplerate=SAMPLERATE,
        channels=CHANNELS,
        callback=audio_callback,
        blocksize=FRAMES_PER_BLOCK,
    ):
        print("Escuchando... pulsa CTRL+C para detenerlo.")
        while True:
            sd.sleep(100)   # Mantiene vivo el contexto; el callback se ejecuta en otro hilo interno
 
# Hilo principal, el del transcriber
 
def transcriber():
    """
    Consume bloques de la cola, acumula audio y transcribe cada vez que
    se alcanza  el "CHUNK_DURATION" en segundos, basicamente: Bloquea en audio_queue.get() hasta recibir un bloque, añade el bloque 
    al buffer local y si el buffer ya tiene >= FRAMES_PER_CHUNK descarta el excedente y vacia el buffer.
    """
           
 
                                                    ###Nota sobre Beam Size###
    """       
    En este caso beam_size equivale a 1 (beam_size=1) lo que es una búsqueda "greedy" (sin beam search), lo que maximiza velocidad
    a costa de una ligera reducción en calidad.  Aumentar a 5 mejora resultados pero puede limitar la velocidad.
    """
    
    global audio_buffer
 
    while True:
        
        block = audio_queue.get()          # Bloquea hasta que haya datos disponibles
        audio_buffer.append(block)
 
        
        total_frames = sum(len(b) for b in audio_buffer)
 
        if total_frames >= FRAMES_PER_CHUNK: #extraer exactamente un "chunk"
                                                        
            audio_data = np.concatenate(audio_buffer)[:FRAMES_PER_CHUNK]
 
           
            audio_buffer = []
 
            
            
            audio_data = audio_data.flatten().astype(np.float32)
 
            
            segments, _ = model.transcribe(
                audio_data,
                language="es",          #Fuerza el idioma español se puede quitar para detección automática
                beam_size=1,         #La mayor velocidad posible
            )
 
            # Imprime cada segmento detectado (sin marcas de tiempo)
            for segment in segments:
                print(segment.text)
 

threading.Thread(target=recorder, daemon=True).start()
 

transcriber()
