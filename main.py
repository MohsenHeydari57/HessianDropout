from HessianDropout import *
from tensorflow import keras


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") / 255.0).reshape(-1, 784)
x_test = (x_test.astype("float32") / 255.0).reshape(-1, 784)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

inputs = keras.Input(shape=(784,))
x = layers.Dense(200, activation="relu", name="dense1")(inputs)
x = HessianDropout(base_rate=0.5, name="hdrop1")(x)
x = layers.Dense(200, activation="relu", name="dense2")(x)
x = HessianDropout(base_rate=0.5, name="hdrop2")(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy", metrics=["accuracy"])

trainer = HessianDropoutTrainer(
    model,
    optimizer=keras.optimizers.Adam(1e-3),
    loss_fn=keras.losses.CategoricalCrossentropy(),
    alpha=1.0, beta=-1.0,
    warmup_epochs=10,
    m_probes=1,
)

trainer.fit(x_train, y_train, epochs=100, batch_size=64,
            validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")            