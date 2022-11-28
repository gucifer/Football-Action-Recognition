
import tensorflow as tf
from tensorflow import keras
from keras import layers


class Permute(layers.Layer):

    def __init__(self) -> None:
        super().__init__()

    def call(self, x):
        return tf.transpose(x, perm=[0, 2, 1])


class MaxOverTime(layers.Layer):

    def __init__(self) -> None:
        super().__init__()

    def call(self, x):
        return torch.max(x, 2).values


class AlphaModel(layers.Layer):
    def __init__(self) -> None:
        super().__init__()
        
    def build(self, input_shape):
        return super().build(input_shape)

        self.model = tf.keras.Sequential(
            nn.Linear(in_features=512, out_features=256),
            Permute(),
            nn.Conv1d(in_channels=256, out_channels=256, kernel_size=9, stride=1),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=9, stride=1),
            nn.Dropout(p=0.1),
            MaxOverTime(),
            nn.Linear(in_channels=128, out_channels=64)
        )

    def call(self, x):
        return self.model(x)


class AlphaClassificationModel(layers.Layer):

    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.model = nn.Sequential(
            AlphaModel(),
            nn.Linear(in_channels=64, out_channels=self.num_classes)
        )

    def call(self, x):
        return self.model(x)


class AlphaRegressionModel(layers.Layer):

    def __init__(self) -> None:
        super().__init__()

        self.model = nn.Sequential(
            AlphaModel(),
            nn.Linear(in_channels=64, out_channels=1)
        )

    def call(self, x):
        return self.model(x)