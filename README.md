# Instalacion y uso de los scripts para  configurar Faster-Whisper y Piper


## Tabla de contenidos
- [Instalando Faster-Whisper](InstalandoFasterWhsiper)
- [Instalando Piper](InstalandoPiper)
- [El Controlador para Webots](ElControladorParaWebots)
- [Posibles Errores](PosiblesErrores)






## Instalando Faster-Whisper

El hub del proyecto nos da instrucciones y los requisitos para usar faster Whisper https://github.com/SYSTRAN/faster-whisper. En resumen necesitamos:

1. Python arriba de 3.9 (en mi caso use 3.11.7 especificamente)
2. cuBLAS for CUDA 12
3. cuDNN 9 for CUDA 12

Los enlaces los provee el mismo proyecto y son los mismos que se usaron.

Si se sigue la ruta de usar el GPU, solo hay que prestar atencion en las librerias necesarias para CUDA, la que mas puede causar error es la referente a Pytorch.
en caso de que marque error con la version que incluyen las librerias de arriba, puedo especificar que la me funciono es la 124

un comando puede solucionar el problema que creo que puede ser el mas comun:

```
-m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Despues de Instalar todo lo que nos pide Faster-Whisper incluí un script para probar que todo haya quedado en orden. lo podemos encontrar como _pruebagpu.py_

Si todos los mensajes  son positivos despues de hacer nuestra instalacion, es 100% seguro que cualquier progragrama o script que utilice Faster-Whisper va a funcionar.

Para este momnento y si asi lo deseamos, podemos correr el programa que contiene el transcriptor de audio a texto _transcriptor_faster_whisper.py_ y hacer pruebas.
De otra manera podemos proceder a instalar Piper.

## Instalando Piper

La instalacion de piper no supone mayor complicacion, ya que lo que nos interesa es usar su API de Python https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md 

***Tambien se proveen las instrucciones para usar aceleracion CUDA. Para estas pruebas y el controlador, se opto por no usarlo por un posible conclicto entre los requisitos de las versiones de cuda que pueda necesitar cada motor(tanto faster-whisper como piper). Aunque de nuevo y aclaro, esta opcion nunca se exploró.***

Lo unico a tener a consideracion es la voz, en el progrma _prueba_piper_frase_fija.py_ incluí una funcion para comprabar que siempre descargue y cargue correctamente. La primera vez tarda un poco, despues es inmediato.

En el caso de la voz predeterminada incluye la de _en_US-lessac-medium_ en. En este caso se uso la de _es_MX-claude-high_
asi que el comando para cargar y usar la voz seria:
```
python3 -m piper.download_voices es_MX-claude-high
```

El programa que se tiene para hacer pruebas _prueba_piper_frase_fija.py_, no guarda un archivo de audio, y la frase pretedeterminada puede ser modificada a conveniencia.

## El Controlador para Webots

En el caso de webots, si ambos programas se pueden ejecutar por separado, el controlador _controlador_whisper_piper_ funcionara sin duda alguna.


## Posibles errores

De los problemas mas comunes fueron con Whisper. Con Piper casi no encontre ninguno. 

Si despues de correr el script de _pruebagpu_ aparece algun error, he aqui algunas explicaciones:

| Error | Descripcion |
| --- | --- |
| `Cuda Disponible = False` |Cuando Cuda no esta disponible, es posible que el gpu no soporte CUDA*|
| `Versión CUDA de torch` | la mas importante, si obtenemos un error o no muestra la version, es probable que la version de pytorch no sea compatible y tengamos que instalar e incluso degradar a otra version, para este proyecto se uso la 12.4 |

*segun esta lista de https://es.wikipedia.org/wiki/CUDA hay muchas gpus soportadas para CUDA, incluso las mas antiguas (ya mas de 15 años); Sin embargó, esto es algo que no se probó. En el desarrollo se uso una GPU de la serie RTX 3000 (3060).


