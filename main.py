import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

def generate_random_data():
    X = np.expand_dims(np.linspace(0, 3, num=200), axis=1)
    Y = X**2 - 2
    noise = np.random.normal(0, 1, size=X.shape)
    Y = Y + noise
    Y = Y.astype("float32")
    return X, Y

def plot_data(X, Y, plot_name):
    plt.scatter(X, Y, color="blue")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("output/" + plot_name, bbox_inches="tight")
    plt.clf()

def plot_results(X, Y, Y_predict, plot_name):
    plt.scatter(X, Y, color="blue")
    plt.plot(X, Y_predict, color="red")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("output/" + plot_name, bbox_inches="tight")
    plt.clf()

def plot_history(history, plot_name):
    plt.plot(history.epoch, np.array(history.history["loss"]), label="Train loss")
    plt.plot(history.epoch, np.array(history.history["val_loss"]), label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Mean squared error)")
    plt.legend()
    plt.savefig("output/" + plot_name, bbox_inches="tight")
    plt.clf()

def create_baseline_model():
    model = Sequential()
    model.add(Dense(1, input_shape=(1,)))
    model.compile(optimizer=SGD(lr=0.001), loss="mean_squared_error")
    return model

def create_nn_model():
    model = Sequential()
    model.add(Dense(10, input_shape=(1,), activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer=SGD(lr=0.001), loss="mean_squared_error")
    return model

# Create file for saving results
results_file = open("output/results.txt", "w")

# Generate data
X_train, Y_train = generate_random_data()
X_val, Y_val = generate_random_data()
plot_data(X_train, Y_train, "data_train.png")
plot_data(X_val, Y_val, "data_val.png")

#~~~~~~~~~~~~~~~~~~~~~~~~~
# Baseline model
#~~~~~~~~~~~~~~~~~~~~~~~~~
model = create_baseline_model()

# Train model
history = model.fit(X_train, Y_train, batch_size=X_train.shape[0], epochs=10000, validation_data=(X_val, Y_val))
plot_history(history, "history_baseline.png")

# Evaluate
final_train_loss = model.evaluate(X_train, Y_train, verbose=0)
final_val_loss = model.evaluate(X_val, Y_val, verbose=0)
results_file.write("Baseline model> final_train_loss: " + str(final_train_loss) + ", final_val_loss: " + str(final_val_loss) + "\n")

# Predict
Y_predict = model.predict(X_train)
plot_results(X_val, Y_val, Y_predict, "results_baseline.png")

#~~~~~~~~~~~~~~~~~~~~~~~~~
# NN model
#~~~~~~~~~~~~~~~~~~~~~~~~~
model = create_nn_model()

# Train model
history = model.fit(X_train, Y_train, batch_size=X_train.shape[0], epochs=10000, validation_data=(X_val, Y_val))
plot_history(history, "history_nn.png")

# Evaluate
final_train_loss = model.evaluate(X_train, Y_train, verbose=0)
final_val_loss = model.evaluate(X_val, Y_val, verbose=0)
results_file.write("NN model> final_train_loss: " + str(final_train_loss) + ", final_val_loss: " + str(final_val_loss) + "\n")

# Predict
Y_predict = model.predict(X_val)
plot_results(X_val, Y_val, Y_predict, "results_nn.png")
