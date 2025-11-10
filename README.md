# BoardScanner

BoardScanner es un proyecto de procesamiento digital de imágenes desarrollado como parte de la asignatura de Imagen Digital dentro del Grado en Ingeniería Informática en Ingeniería del Software de la Universidad de Extremadura.

El sistema utiliza OpenCV y TensorFlow, entre otras bibliotecas, para realizar el análisis y procesado de imágenes, funcionando como una herramienta que escanea pizarras y realiza distintos tipos de transformaciones y reconocimientos en las capturas obtenidas.

---

## Tabla de Contenidos

1. [Características](#características)
2. [Requisitos Previos](#requisitos-previos)
3. [Instalación](#instalación)
4. [Uso](#uso)
5. [Estructura del Proyecto](#estructura-del-proyecto)

---

## Características

- Escaneo de imágenes de pizarras capturadas por cámaras.
- Variedad de transformaciones como detección de bordes, máscaras en HSV, y umbrales adaptativos.
- Implementación de modelos de reconocimiento de caracteres, incluidos KNN, Perceptrón Multi-Capa (MLP) y Google Vision API.
- Servidor web con Flask para visualizar en tiempo real el procesamiento de las imágenes en diferentes etapas.

---

## Requisitos Previos

- Python 3.10 o superior.
- Internet para el uso de Google Cloud Vision API.

---

## Instalación

1. Clona el repositorio:
   ```bash
   git clone https://github.com/rdereparadores/BoardScanner.git
   cd BoardScanner
   ```

2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

3. Asegúrate de tener configuradas tus credenciales de Google Vision API si utilizarás esta funcionalidad.

---

## Uso

Ejecuta la aplicación principal desde `main.py` para iniciar el procesamiento y el servidor web:
```bash
python main.py
```

La aplicación estará disponible localmente en el puerto **8080**. Navega a `http://localhost:8080` en tu navegador para interactuar con la interfaz provista.

---

## Estructura del Proyecto

```
BoardScanner/
├── boardscanner/               # Módulos principales del procesamiento
│   ├── Config.py               # Configuración general
│   ├── Globals.py              # Variables y estado global
│   ├── Utils.py                # Funciones utilitarias
│   ├── callbacks.py            # Callbacks para procesos
│   ├── main_loop.py            # Loop principal de procesamiento
│   ├── train_mlp.py            # Entrenamiento de modelo MLP
│   ├── emnist_sample.keras     # Modelo preentrenado
├── img/                        # Ejemplos de imágenes procesadas
│   ├── pizarraia_1.png
│   ├── pizarraia_2.png
│   ├── ...
├── templates/                  # Plantillas HTML
│   ├── index.html
├── main.py                     # Script principal
├── requirements.txt            # Dependencias del proyecto
```
