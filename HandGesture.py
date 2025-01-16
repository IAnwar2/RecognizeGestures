import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.saving import register_keras_serializable

num_output_classes = 3
num_neurons_layer_1 = 128
num_neurons_layer_2 = 64

@register_keras_serializable()
class FCNN(Model):
    def __init__(self, num_neurons_layer_1, num_neurons_layer_2, num_output_classes, **kwargs):
        super(FCNN, self).__init__(**kwargs)
        self.num_neurons_layer_1 = num_neurons_layer_1
        self.num_neurons_layer_2 = num_neurons_layer_2
        self.num_output_classes = num_output_classes
        self.l1 = Dense(num_neurons_layer_1, activation='relu')
        self.l2 = Dense(num_neurons_layer_2, activation='relu')
        self.out = Dense(num_output_classes)

    def call(self, x, training=False):
        x = self.l1(x)
        x = self.l2(x)
        x = self.out(x)
        if not training:
            x = tf.nn.softmax(x)
        return x

    def get_config(self):
        config = super(FCNN, self).get_config()
        config.update({
            "num_neurons_layer_1": self.num_neurons_layer_1,
            "num_neurons_layer_2": self.num_neurons_layer_2,
            "num_output_classes": self.num_output_classes,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)