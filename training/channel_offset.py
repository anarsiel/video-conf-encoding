import tensorflow as tf
import numpy as np


# Собственный слой для случайного изменения яркости каждого из каналов в отдельности
class RandomChannelOffset(tf.keras.layers.Layer):
    def __init__(self, offset_min_max=0.1, **kwargs):
        super(RandomChannelOffset, self).__init__(**kwargs)
        if type(offset_min_max) in [int, float]:
            offset_min_max = np.repeat([[-offset_min_max, offset_min_max]], 3, axis=0)
        self.offset_min_max = offset_min_max

    def call(self, images, **kwargs):
        get_delta = lambda offset: np.random.uniform(offset[0], offset[1])
        offsets = list(map(get_delta, self.offset_min_max))
        images = tf.image.adjust_brightness(images, offsets)
        images = tf.clip_by_value(images, 0, 255)

        return images


# noinspection PyTypeChecker
tr = RandomChannelOffset([[-30, -5],
                          [-10, 10],
                          [5, 30]])
