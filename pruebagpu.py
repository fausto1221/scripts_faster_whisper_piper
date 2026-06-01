#prueba de instalacion exitosa#

import torch
print("CUDA disponible:", torch.cuda.is_available())
print("Versión CUDA de torch:", torch.version.cuda)
print("Nombre GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No detectada")
print("Número de dispositivos:", torch.cuda.device_count())