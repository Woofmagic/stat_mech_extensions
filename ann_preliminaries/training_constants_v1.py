"""
The script that was used to visualize the "potential well" that we interpret the 
"predator particle" to move within. In classical physics, the potential well picture
is quite informative, so it helps to see (if you can!) what the particle is moving "in."
"""

# 3rd Party Library | NumPy
import numpy as np

# 3rd Party Library | TensorFlow
import tensorflow as tf

# 3rd Party Library | Matplotlib:
import matplotlib.pyplot as plt

# (X): Define a version number so we don't get confused:
_version_number = "1.1"

# (X): Dynamically set the plot title using the version number:
PLOT_TITLE = f"training_constants_v{_version_number}"

# (X): Fix the plot directory for the LV analysis:
# PLOT_DIRECTORY = "plots"

# (X): We tell rcParams to use LaTeX. Note: this will *crash* your 
# | version of the code if you do not have TeX distribution installed!
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

# (X): rcParams for the x-axis tick direction:
plt.rcParams['xtick.direction'] = 'in'

# (X): rcParams for the "major" (larger) x-axis vertical size:
plt.rcParams['xtick.major.size'] = 5

# (X): rcParams for the "major" (larger) x-axis horizonal width:
plt.rcParams['xtick.major.width'] = 0.5

# (X): rcParams for the "minor" (smaller) x-axis vertical size:
plt.rcParams['xtick.minor.size'] = 2.5

# (X): rcParams for the "minor" (smaller) x-axis horizonal width:
plt.rcParams['xtick.minor.width'] = 0.5

# (X): rcParams for the minor ticks to be *shown* versus invisible:
plt.rcParams['xtick.minor.visible'] = True

# (X): rcParams dictating that we want ticks along the x-axis on *top* (opposite side) of the bounding box:
plt.rcParams['xtick.top'] = True

# (X): rcParams for the y-axis tick direction:
plt.rcParams['ytick.direction'] = 'in'

# (X): rcParams for the "major" (larger) y-axis vertical size:
plt.rcParams['ytick.major.size'] = 5

# (X): rcParams for the "major" (larger) y-axis horizonal width:
plt.rcParams['ytick.major.width'] = 0.5

# (X): rcParams for the "minor" (smaller) y-axis vertical size:
plt.rcParams['ytick.minor.size'] = 2.5

# (X): rcParams for the "minor" (smaller) y-axis horizonal width:
plt.rcParams['ytick.minor.width'] = 0.5

# (X): rcParams for the minor ticks to be *shown* versus invisible:
plt.rcParams['ytick.minor.visible'] = True

# (X): rcParams dictating that we want ticks along the y-axis on the *left* of the bounding box:
plt.rcParams['ytick.right'] = True

# (X): Define the independent variable resolution:
TIME_SLICE = 0.0005

# (X): Define the starting value of the independent variable:
TIME_STARTING_VALUE = -20

# (X): Define the upper valye of the independent variable:
TIME_STOPPING_VALUE = 20

# (X): Define a "time" as the underlying independent variable:
TIME_ARRAY = np.arange(
    TIME_STARTING_VALUE,
    TIME_STOPPING_VALUE,
    TIME_SLICE).reshape(-1, 1)

# (X): Define a "position" as the dependent variable with an underlying functional form:
POSITION_ARRAY = np.sin(TIME_ARRAY)

# (X): [NOTE]: It was guessed that having ONE data point is required for a constant V:
t_single = TIME_ARRAY[500:501]

# (X): See comment above:
x_single = POSITION_ARRAY[500:501]

# (X): Convert the NumPy arrays to TF tensors... ANNOYING!
t_tensor = tf.convert_to_tensor(TIME_ARRAY, dtype = tf.float32)
x_tensor = tf.convert_to_tensor(POSITION_ARRAY, dtype = tf.float32)

class SigmoidLinearModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.w = tf.Variable(tf.random.normal([1]), name = "weight")

        self.b = tf.Variable(tf.zeros([1]), name = "bias")

    def call(self, t):
        return tf.sigmoid(self.w * t + self.b)

# (X): Use the simple model:
model = SigmoidLinearModel()

# (X): Fix a number of epochs:
NUMBER_OF_EPOCHS = 1000

# (X): Fix the learning rate.
# | [NOTE]: This fixed value is important because it sets up the
# | "homogeneous time" interpretation.
LEARNING_RATE = 0.01

# (X): Define the optimizer:
optimizer = tf.keras.optimizers.SGD(learning_rate = LEARNING_RATE)

# (X): Use the MSE loss:
loss_fn = tf.keras.losses.MeanSquaredError()

# (X): Initialize a list to record loss values during training:
history_of_loss_values = []

# (X): Initialize a list to record the weight values during training:
history_of_weight_values = []

# (X): Initialize a list to record the bias values during training:
history_of_bias_values = []

# (X): Initialize a list to record the "constant" V values during training:
history_of_v_values = []

# (X): Begin the gradient descent algorithm:
for epoch in range(NUMBER_OF_EPOCHS):

    # (X): Unravel the gradient tape:
    with tf.GradientTape() as tape:

        # (X): Pass in the training tensor into the model for prediction:
        predicted_x_value = model(t_tensor)

        # (X): Compute the MSE loss:
        current_loss = tf.reduce_mean((predicted_x_value - x_tensor) ** 2)

    # (X): Compute the gradient descent computation:
    computed_gradients = tape.gradient(current_loss, [model.w, model.b])

    # (X): Add the newly-calculated gradient changes to each of the model parameters:
    optimizer.apply_gradients(zip(computed_gradients, [model.w, model.b]))

    # (X): Add the current loss to the "history" array of losses:
    history_of_loss_values.append(current_loss.numpy())

    # (X): Add the current weight value to the history array:
    history_of_weight_values.append(model.w.numpy()[0])

    # (X): Add the current bias value to the history array:
    history_of_bias_values.append(model.b.numpy()[0])

    # (X): Add the current value of the conjectured constant to the history array:
    history_of_v_values.append(
        model.b.numpy()[0] * (len(TIME_ARRAY) * TIME_STARTING_VALUE + 0.5 * len(TIME_ARRAY) * (len(TIME_ARRAY) + 1) * TIME_SLICE)- model.w.numpy()[0])

fig, axs = plt.subplots(2, 2, figsize = (12, 8))

axs[0,0].plot(history_of_loss_values, label = "Loss")
axs[0,0].set_title("Loss vs Epoch")
axs[0,0].set_xlabel("Epoch")
axs[0,0].set_ylabel("Loss")
axs[0,0].legend()

axs[0,1].plot(history_of_weight_values, label = "Weight", color = "blue")
axs[0,1].set_title("Weight vs Epoch")
axs[0,1].set_xlabel("Epoch")
axs[0,1].set_ylabel("Weight")
axs[0,1].legend()

axs[1,0].plot(history_of_bias_values, label = "Bias", color = "green")
axs[1,0].set_title("Bias vs Epoch")
axs[1,0].set_xlabel("Epoch")
axs[1,0].set_ylabel("Bias")
axs[1,0].legend()

axs[1,1].plot(history_of_v_values, label = "Conjectured Training Constant V", color = "red")
axs[1,1].set_title("V vs Epoch")
axs[1,1].set_xlabel("Epoch")
axs[1,1].set_ylabel("V")
axs[1,1].legend()

plt.tight_layout()
plt.show()

t_test = np.linspace(0, 20, 500).reshape(-1, 1)
x_true = np.sin(t_test)
x_pred = tf.sigmoid(model.w * t_test + model.b).numpy()

plt.figure(figsize = (10,5))
plt.plot(t_test, x_true, label = "True Function sin(t)", color = "black")
plt.plot(t_test, x_pred, label = "Learned Model", color = "orange")
plt.title("True vs Learned Function")
plt.xlabel("t (time)")
plt.ylabel("x(t)")
plt.legend()
plt.show()
