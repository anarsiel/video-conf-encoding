from matplotlib import pyplot as plt


def save_image(image, name="tmp.jpg"):
    ax = plt.subplot(3, 3, 1)
    plt.imshow(image.astype("uint8"))
    plt.axis("off")
    plt.savefig(name)


def save_image(image, name="tmp.jpg"):
    ax = plt.subplot(3, 3, 1)
    plt.imshow(image.astype("uint8"))
    plt.axis("off")
    plt.savefig(name)


def read_image(source, image_shape=(50, 60)):
    image = tf.io.read_file(source)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.reshape(image, (1, image_shape[0] * image_shape[1], 3))  # (1, 3000, 3)
    image = tf.cast(image, tf.float32)
    image = tf.cast(image / 255., tf.float32)
    return image
