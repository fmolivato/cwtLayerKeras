# cwt-layer-keras

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

This project main goal is to produce a simple way to compute **CWT** (Continuous Wavelet Transformation) on signals in keras models.

## Available wavelet:
* morlet
* mexican hat / ricker

## Useful features
class paramenter | type | description | why
--- | --- | --- | ---
depth_pad | int | It allows to specify the number of padding channel that the layer need to add | some standard keras model (like efficientnet) need 3 channel depth to execute (instead of just the single one of the scalogram)
trainable_kernels | bool | It make the wavelets that produce the scalogram trainable | interesting to see how/if the net optimizers change the wavelet kernel (research purpose)

## DISCLAIMER
This project was built for research purpose, so feel free to open some issue if you find some errors!