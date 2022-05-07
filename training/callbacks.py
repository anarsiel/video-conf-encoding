import random
import matplotlib.pyplot as plt


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch = 0

    def on_train_batch_end(self, batch, logs=None):
        # files.download('weights.h5')
        # model.save('batch_w')
        if batch % 200 == 0:
            model.save_weights(f"drive/MyDrive/models/best_model{self.epoch}.hdf5")

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1


class CustomCallback2(tf.keras.callbacks.Callback):
    def __init__(self, val_dataset, show_step=50):
        super().__init__()
        self.dataset = val_dataset
        self.show_step = show_step

    def on_train_batch_end(self, batch, logs=None):
        # if batch % self.show_step == 0 and not batch == 0:
        # show = (random.randint(0, 5) == 0)

        if batch != 0 and batch % 500 == 0 or batch % 501 == 0:  # or batch % 202 == 0 or batch % 203 == 0 or batch % 204 == 0 or batch % 205 == 0:
            show_preds(model, self.dataset)


font = {'weight': 'bold',
        'size': 6}

plt.rc('font', **font)


def show_preds(model, test_dataset):
    k = np.random.choice(len(test_dataset))
    # k = 0
    X, Y = test_dataset[k]
    preds = model(X)
    plt.figure(dpi=150)
    for i, (pred, y) in enumerate(zip(preds[0][:5], Y[0][:5])):
        ax1 = plt.subplot(2, 5, i + 1)
        ax1.imshow(pred)
        ax1.axis('off')

        ax2 = plt.subplot(2, 5, 5 + i + 1)
        ax2.imshow(y)
        ax2.axis('off')
        delta = (y - pred).numpy()
        string = '{:.4f}, {:.4f}'.format(delta.mean(), delta.std())
        ax2.set_title(string)
    plt.show()


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    # filepath=checkpoint_filepath,
    "drive/MyDrive/models/current_best.hdf5",
    monitor="mean_squared_error",
    mode="min",
    save_best_only=True,
    verbose=0,
    save_weights_only=True,
    save_freq=50
)

saving_callback = CustomCallback()
draw_callback = CustomCallback2(test_dataset, show_step=200)
