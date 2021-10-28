# Music Videos Visual Feature Extraction

This repository contains utilities for extracting visual features from Music Videos.

## ☑️ Prerequisites

In order to run the application, you will need to install the Python libraries listed below:

- Python >= 3.7
- NumPy >= 1.19
- SciPy >= 1.6
- PyInquirer >= 1.0.3
- OpenCV-Python >= 4.1.1
- Tensorflow >= 2.6.0
- Keras >= 2.6.0
- CUDA >= 11.4

For a simple solution, you can simply run the below command in the root directory:

```python
pip install -r requirements.txt
```

Note that you should also install NVIDIA® CUDA® Deep Neural Network library™ (cuDNN) for high-performance GPU acceleration. This is used when training the DNN models in the feature extraction stage.

## 🚀 Launch the Application

The first step to run the engine of the application is to provide a proper configuration file. The `main.py` file, which is the start point of the application, needs such file to provide customized configurations for its differernt modules and then run the application:

#### I. Make a Configuration File

You can find a **config.example.py** in the root directory. What you need to do is to make a copy of this file and rename it to **config.py**. There, you can apply your customized settings. Please note that the **config.py** is placed in `.gitignore` file due to the customized settings.

#### II. Run the Application

After providing the configurations, you can easily run the app using the command below in the root directory:

```python
python ./main.py
```

Please note that we assumed videos are already FPS=1 and ignored the process capturing one frame per second.
