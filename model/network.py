import tensorflow as tf


class DNN():
    tf.random.set_seed(42)  # first we set random seed

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1)
    ])
    model.compile(loss=tf.keras.losses.mae,  # mae stands for mean absolute error
                  optimizer=tf.keras.optimizers.SGD(),  # stochastic GD
                  metrics=['mae'])
    model.fit(X_train, y_train, epochs=10)
