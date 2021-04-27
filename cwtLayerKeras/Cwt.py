import tensorflow as tf
import numpy as np


class Cwt(tf.keras.layers.Layer):
    ''' Tensorflow2.4 implementation of Keras layer for doing optimized cwt on batches of signals. '''

    def __init__(
        self,
        sample_count: int,
        wavelet: str = "morl",
        min_scale: int = 4,
        max_scale: int = 104,
        scales_step: float = 1,
        output_size: tuple = (200, 200),
        depth_pad: int = None,
        name: str = "cwt_layer",
        trainable_kernels: bool = False,
        **kwargs
    ):
        '''
        Class initializer

            Parameters:
                sample_count (int): 
                    Tensor's fist dimension size, that need to be shared 
                    between each wavelet kernel in order to stack them in a tf.Tensor
                wavelet (str): 
                    A string name of a supported wavelet.

                    The supported names are: 
                        "morl" -> morlet wavlet, 
                        "rick"/"mex_hat" -> mexican hat wavelet also known as ricker
                min_scale (int): 
                    Bottom boundary of scales range for the batch
                max_scale (int): 
                    Decimal integer, it is the top boundary of scales range for the batch
                scales_step (float): 
                    Range step of the wavelet scales
                output_size (tuple): 
                    Tuple with the (x,y) size of the returned scalograms during training 
                depth_pad (int): 
                    Decimal of how much channel of pad is needed
                name (str): 
                    Name of the layer in the keras summary of the model
                trainable_kernels (bool): 
                    Boolean that is used to enable gradient optimization on the wavelet kernels
                    (mainly for research pourpose).

        '''
        super(Cwt, self).__init__(name=name, **kwargs)
        self.max_scale = max_scale
        self.output_size = output_size
        self.depth_pad = depth_pad
        self.wavelet = wavelet
        self.sample_count = sample_count
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.scales_step = scales_step
        self.trainable_kernels = trainable_kernels

    def build(self, input_shape):
        self.kernels = tf.Variable(
            initial_value=self.create_kernels(
                self.wavelet,
                sample_count=self.sample_count,
                lowest=self.min_scale,
                highest=self.max_scale,
                step=self.scales_step,
            ),
            trainable=self.trainable_kernels,
        )

    def call(self, inputs):
        cwts = self.vectorized_cwts(inputs, self.kernels)
        cwts = tf.image.resize(cwts, self.output_size)

        # take care of the depth_pad that make it runnable on nets like
        # efficientnet that requires 3 channel depth (instead of just one).
        if self.depth_pad:
            pad = tf.ones((tf.shape(cwts)[0], *self.output_size, 1))
            cwts = tf.concat([cwts, pad, pad], axis=3)

        return cwts

    # useful when 'trainable_kernels' are True so it is possible to see
    # how the net gradient optimize the wavelet kernel
    def get_kernels(self):
        '''Return the batch of wavelets scales'''
        return self.kernels

    # @tf.function
    def vectorized_cwts(self, spectra, scaled_kernels):
        '''
        Vectorize cwt across a batch of samples

            Parameters:
                spectra (tf.Tensor): 
                    Batch of samples 
                batch_kernels (tf.Tensor): 
                    Batch of wavelets with same kernel (eg. morlet) and different scales

            Returns:
                (tf.Tensor): Batch of scalograms
        '''
        return tf.vectorized_map(
            lambda spectrum: self.cwt(
                scaled_kernels, spectrum), spectra
        )

    # necessary to silence some tensorflow 2.4 warnings
    @tf.autograph.experimental.do_not_convert
    def cwt(self, batch_kernels, sample):
        '''
        Returns a scalogram of the sample

            Parameters:
                batch_kernels (tf.Tensor): 
                    Batch of wavelets with same kernel (eg. morlet) and different scales
                sample (tf.Tensor): 
                    Spectrum sample

            Returns:
                res_norm (tf.Tensor): 
                    Scalogram normalized in [0, 1]
        '''
        h, w = batch_kernels.shape
        batch_kernels = tf.reshape(batch_kernels, tf.constant([h, w, 1]))

        # Convolution theory says that it is necessary to reverse the signal
        # that will be convoluted on another one
        sample = tf.reverse(sample, tf.constant([0]))
        sample = tf.reshape(sample, tf.constant([sample.shape[0], 1, 1]))

        res = tf.nn.conv1d(batch_kernels, filters=sample,
                           stride=1, padding="SAME")
        res_norm = (res-tf.reduce_min(res)) / \
            (tf.reduce_max(res)-tf.reduce_min(res))

        return res_norm

    # batch of wavelet scales used in vectorized convolution with single sample
    def create_kernels(self, wavelet_name: str, highest: int, sample_count: int, lowest: int = 1, step: float = 1):
        '''
        Returns batch of wavelet with different scale (from 'lowest' to 'highest' with 'step')
        with tensor first dimension of 'sample_count'.

            Parameters:
                wavelet (str): 
                    A string name of a supported wavelet.

                    The supported names are: 
                        "morl" -> morlet wavlet, 
                        "rick"/"mex_hat" -> mexican hat wavelet also known as ricker
                highest (int): 
                    Decimal integer, it is the top boundary of scales range for the batch
                sample_count (int): 
                    Tensor's fist dimension size, that need to be shared 
                    between each wavelet kernel in order to stack them in a tf.Tensor
                lowest (int): 
                    Bottom boundary of scales range for the batch
                step (float): 
                    Range step of the wavelet scales

            Returns:
                batch_kernels (tf.Tensor): 
                    Tensor of wavelet kernels of size (range(lowest, highest, step), sample_count)
        '''
        # since ricker and mexican hat are the same
        wavelet_name = 'rick' if wavelet_name == 'mex_hat' else wavelet_name

        scales = tf.range(lowest, highest + 1, step, dtype=tf.float32)

        switcher = {
            "morl": tf.vectorized_map(
                lambda scale: self.morlet_wavelet(
                    tf.float32, scale, sample_count),
                scales,
            ),
            "rick": tf.vectorized_map(
                lambda scale: self.ricker_wavelet(
                    tf.float32, scale, sample_count),
                scales,
            ),
        }

        if wavelet_name not in switcher.keys():
            raise ValueError(
                f"This wavelet in not present in the {__file__} library")

        batch_kernels = switcher.get(wavelet_name)

        return batch_kernels

    def ricker_wavelet(self, tf_type, scale: float, sample_count: int):
        '''
        Returns a ricker (or mexican hat) wavelet with defined 'scale' and size of 'sample_count'
        Equation defined here:
        https://it.mathworks.com/help/wavelet/ref/mexihat.html

            Parameters:
                tf_type (tf.Type): 
                    A tensorflow 2.4 type (eg. tf.float32)
                scale (float): 
                    Scale of the wavelet, or the longitudinal compression/extension of wavelet kernel
                sample_count (int): 
                    Desired wavelet length

            Returns:
                wav (tf.Tensor): A wavelet with ricker kernel tensor of 'tf_type' type 
        '''
        def wave_equation(time):
            time = tf.cast(time, tf_type)

            tSquare = time ** 2.
            sigma = 1.
            sSquare = sigma ** 2.

            # _1 = 2 / ((3 * a) ** .5 * np.pi ** .25)
            _1a = (3. * sigma) ** .5
            _1b = np.pi ** .25
            _1 = 2. / (_1a * _1b)

            # _2 = 1 - t**2 / a**2
            _2 = 1. - tSquare / sSquare

            # _3 = np.exp(-(t**2) / (2 * a ** 2))
            _3a = -1. * tSquare
            _3b = 2. * sSquare
            _3 = tf.exp(_3a / _3b)

            return _1 * _2 * _3

        return self.wavelet_helper(tf_type, scale, sample_count, wave_equation)

    def morlet_wavelet(self, tf_type, scale: float, sample_count: int):
        '''
        Returns a morlet wavelet with defined 'scale' and size of 'sample_count'
        Equation defined here:
        https://www.mathworks.com/help/wavelet/ref/morlet.html

            Parameters:
                tf_type (tf.Type): 
                    A tensorflow 2.4 type (eg. tf.float32)
                scale (float): 
                    Scale of the wavelet, or the longitudinal compression/extension of wavelet kernel
                sample_count (int): 
                    Desired wavelet length

            Returns:
                wav (tf.Tensor): A wavelet with morlet kernel tensor of 'tf_type' type 
        '''
        def wave_equation(time):
            return tf.exp(-1. * (time ** 2.) / 2.) * tf.cos(5. * time)

        wav = self.wavelet_helper(tf_type, scale, sample_count, wave_equation)

        return wav

    def wavelet_helper(self, tf_type, scale, sample_count, wave_equation):
        '''
        Takes care of all the data casting and time calculation for a correct wavelet calculation,
        and then produce the wavelet of kernel 'wave_equation'

            Parameters:
                tf_type (tf.Type): 
                    A tensorflow 2.4 type (eg. tf.float32)
                scale (float): 
                    Scale of the wavelet, or the longitudinal compression/extension of wavelet kernel
                sample_count (int): 
                    Desired wavelet length
                wave_equation (func):
                    Function that describe the actual mathematical rapresentation of the wavelet

            Returns:
                wav (tf.Tensor): A wavelet with 'wave_equation' kernel tensor of 'tf_type' type 
        '''
        scale = tf.cast(scale, tf_type)
        sample_count = tf.cast(sample_count, tf_type)
        unscaledTimes = tf.cast(
            tf.range(tf.cast(sample_count, tf.int32)), tf_type) - (sample_count - 1.) / 2.
        times = unscaledTimes / scale
        wav = wave_equation(times)
        #wav           = wav * scale ** -.5
        return wav
