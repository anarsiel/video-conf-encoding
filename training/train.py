def loss_w(y_true, y_pred):
    nf = 10
    weight_center = 0.85

    y_00_t = y_true[:, :nf, :10, :10, :]
    y_00_p = y_pred[:, :nf, :10, :10, :]

    y_10_t = y_true[:, :nf, 10:30, :10, :]
    y_10_p = y_pred[:, :nf, 10:30, :10, :]

    y_20_t = y_true[:, :nf, 30:, :10, :]
    y_20_p = y_pred[:, :nf, 30:, :10, :]

    y_01_t = y_true[:, :nf, :10, 10:45, :]
    y_01_p = y_pred[:, :nf, :10, 10:45, :]

    y_21_t = y_true[:, :nf, 30:, 10:45, :]
    y_21_p = y_pred[:, :nf, 30:, 10:45, :]

    y_02_t = y_true[:, :nf, :10, 45:, :]
    y_02_p = y_pred[:, :nf, :10, 45:, :]

    y_12_t = y_true[:, :nf, 10:30, 45:, :]
    y_12_p = y_pred[:, :nf, 10:30, 45:, :]

    y_22_t = y_true[:, :nf, 30:, 45:, :]
    y_22_p = y_pred[:, :nf, 30:, 45:, :]

    y_11_t = y_true[:, :nf, 10:30, 10:45, :]
    y_11_p = y_pred[:, :nf, 10:30, 10:45, :]

    mse_edge = (
            mse(y_00_t, y_00_p) + mse(y_10_t, y_10_p) + mse(y_20_t, y_20_p) +
            mse(y_01_t, y_01_p) + mse(y_21_t, y_21_p) +
            mse(y_02_t, y_02_p) + mse(y_12_t, y_12_p) + mse(y_22_t, y_22_p)
    )

    return (1 - weight_center) * mse_edge + weight_center * mse(y_11_t, y_11_p)

#model.compile(optimizer='adam', loss=loss_w)
adam = tf.keras.optimizers.Adam(0.0001)
model = create_model(save_plot=True)
mse = tf.keras.losses.MeanSquaredError(
    name='mean_squared_error'
)
model.compile(
     loss=loss_w,
     optimizer=adam,
     metrics=['mean_squared_error']
)

# model.load_weights("best_model977.hdf5")

checkpoint_filepath = 'checkpoints/'

plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.plot(model.history.history['mean_squared_error'])