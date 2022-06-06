from sklearn.model_selection import train_test_split
import tensorflow as tf

def train(data): 
    target = df_year["target_eret"]
    theory_target = df_year["theory_eret"]
    features = df_year.iloc[:, 2:]

    # split into train, validation and test data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=1)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2

    tf.random.set_seed(42)
    dnn = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1)
    ])
    dnn.compile(loss=tf.keras.losses.mae,
                    optimizer=tf.keras.optimizers.SGD(),
                    metrics=['mae'])
    dnn.fit(X_train, y_train, epochs=100, verbose=0, batch_size=32)

    preds_6 = dnn.predict(X_test)

    # Evaluation
    mae_6 = tf.metrics.mean_absolute_error(y_true=y_test, 
                                        y_pred=preds_6.squeeze()).numpy()
    mse_6 = tf.metrics.mean_squared_error(y_true = y_test,
                                        y_pred=preds_6.squeeze()).numpy()
    
    print(mae_6, mse_6)