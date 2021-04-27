# cwt-layer-keras

<img alt="Keras" src="https://img.shields.io/badge/Keras-%23D00000.svg?&style=for-the-badge&logo=Keras&logoColor=white"/> <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white" /> [![Generic badge](https://img.shields.io/badge/python-v3.6+-<COLOR>.svg)](https://www.python.org/download/releases/3.0/) [![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0) 

This project main goal is to produce a simple way to compute **CWT** (Continuous Wavelet Transformation) on signals :satellite: with keras functional API.

## Installation
    pip install cwtLayerKeras

## Available wavelets:
* "morl" (morlet)
* "mex_hat" or "rick" (mexican hat / ricker)

## Additional features
class paramenter | type | description | why
--- | --- | --- | ---
depth_pad | int | It allows to specify the number of padding channel that the layer need to add | some standard keras model (like efficientnet) need 3 channel depth to execute (instead of just the single one of the scalogram)
trainable_kernels | bool | It make the wavelets that produce the scalogram trainable | interesting to see how/if the net optimizers change the wavelet kernel (research purpose)

## Disclaimer:
 :exclamation: This project was built for research purpose :exclamation:
 
 Feel free to open some issue if you find some errors.

