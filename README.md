# cwtLayerKeras

<img alt="Keras" src="https://img.shields.io/badge/Keras-%23D00000.svg?&style=for-the-badge&logo=Keras&logoColor=white"/> <img alt="TensorFlow" src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?&style=for-the-badge&logo=TensorFlow&logoColor=white" /> [![Generic badge](https://img.shields.io/badge/python-v3.6+-<COLOR>.svg)](https://www.python.org/download/releases/3.0/) [![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0) 

This project main goal is to produce a simple way to compute **CWT** (Continuous Wavelet Transformation) scalogram on signals :satellite: with keras functional API.

## Installation
    pip install cwtLayerKeras

## Available wavelets:
* "morl" (morlet)
* "mex_hat" or "rick" (mexican hat / ricker)

## Additional features
class paramenter | type | description | why
--- | --- | --- | ---
depth_pad | int | It allows to specify the number of padding channel that the layer need to add | some standard keras model (like efficientnet) need 3 channel depth to execute, instead of just the single one of the scalogram
trainable_kernels | bool | It make the wavelets that produce the scalogram trainable | interesting to see how/if the net optimizers change the wavelet kernel (research purpose)

## Example

```python
from cwtLayerKeras import Cwt as cwtt

# some code

def build_model():
    input_data = layers.Input(shape=(x_val.shape[-1],))
    cwt = cwtt(
        sample_count=x_val.shape[-1],
        scales_step=10,
        min_scale = 4,
        max_scale=224,
        output_size=(224, 224),
        depth_pad=2,
    )
    scalogram = cwt(input_data)
    base_model = EfficientNetB0(include_top=False, weights=None)(scalogram)

    x = layers.GlobalAveragePooling2D()(base_model)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    predictions = layers.Dense(OUTPUT_SIZE, activation="softmax")(x)

    model = Model(inputs=input_data, outputs=predictions)

    model.summary()
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
```

## Disclaimer:
 :exclamation: This project was built for research purpose, so there could be some errors:exclamation:
 
 Feel free to open open issues on those.

