# Instalacion y uso de los scripts para  configurar faster whysper y piper


##Tabla de contenidos
- [Instalando Faster-Whisper](InstalandoFasterWhsiper)




## Instalando Faster-Whisper

El hub del proyecto nos da instrucciones y los requisitos para usar faster Whisper https://github.com/SYSTRAN/faster-whisper. En resumen necesitamos:

1. Python arriba de 3.9 (en mi caso use 3.11.7 especificamente)
2. cuBLAS for CUDA 12
3. cuDNN 9 for CUDA 12

Los enlaces los provee el mismo proyecto y son los mismos que se usaron.

Si se sigue la ruta de usar el GPU, solo hay que prestar atencion en las librerias necesarias para CUDA, la que mas puede causar error es la referente a Pytorch.
en caso de que marque error con la version que incluyen las librerias de arriba, puedo especificar que la me funciono es la 121

un comando puede solucionar el problema que creo que puede ser el mas comun:

```
-m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Despues de Instalar todo lo que nos pide Faster-Whisper incluí un script para probar que todo haya quedado en orden. lo podemos encontrar como _pruebagpu.py_

Si todos los mensajes  son positivos despues de hacer nuestra instalacion, es 100% seguro que cualquier progragrama o script que utilice Faster-Whisper va a funcionar.

Para este momnento y si asi lo deseamos, podemos correr el programa que contiene el transcriptor de audio a texto _transcriptor_faster_whisper.py_ y hacer pruebas.
De otra manera podemos proceder a instalar Piper.

##ins
