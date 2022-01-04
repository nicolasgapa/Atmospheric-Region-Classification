"""
Embry-Riddle Aeronautical University
Space Physics Research Lab (SPRL)
Author: Nicolas Gachancipa
Neural Network to distinguish Auroral Oval and Polar Cap events.
"""

# Imports.
import matplotlib.pyplot as plt
import tensorflow.keras as k
import pandas as pd
import numpy as np

# Input.
split = 0.8  # Percentage of training data.
file_name = 'training_data.csv'
model_type = 'softmax'  # Either softmax or sigmoid activation function.
epochs = 500
plot_loss = False

# Read the file.
df = pd.read_csv(file_name)

# Shuffle all the data instances.
# df = df.sample(frac=1)

# Split training and testing data.
training = df.iloc[:int(df.shape[0] * split), :]
testing = df.iloc[int(df.shape[0] * split):, :]

# Define X (explanatory variables) and y (response variable).
X = training.drop(['REGION'], axis=1)

# 50 HZ signal over 30 seconds, 1500 data points. (300 data instances).
if model_type == 'softmax':
    y = np.array([[0, 1] if l == 'AuroralOval' else [1, 0] for l in training['REGION']])  # 1 for Auroral, 0 for Polar.
else:
    y = np.array([0 if l == 'AuroralOval' else 1 for l in training['REGION']])  # 1 for Auroral, 0 for Polar.

# Define the model.
if model_type == 'softmax':
    model = k.Sequential([k.layers.Dense(units=64, input_shape=[X.shape[1]], activation='sigmoid'),
                          k.layers.Dense(units=32, activation='sigmoid'),
                          k.layers.Dense(units=16, activation='sigmoid'),
                          k.layers.Dense(units=8, activation='sigmoid'),
                          k.layers.Dense(units=4, activation='sigmoid'),
                          k.layers.Dense(units=2, activation='softmax')])
else:
    model = k.Sequential([k.layers.Dense(units=64, input_shape=[X.shape[1]], activation='sigmoid'),
                          k.layers.Dense(units=32, activation='sigmoid'),
                          k.layers.Dense(units=16, activation='sigmoid'),
                          k.layers.Dense(units=8, activation='sigmoid'),
                          k.layers.Dense(units=4, activation='sigmoid'),
                          k.layers.Dense(units=1, activation='sigmoid')])

# Compile the model: Define the loss function and the optimizer.
model.compile(optimizer=k.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['accuracy'])

# Train the model (save the weights in an h5 file every time a better accuracy is obtained).
checkpoint = k.callbacks.ModelCheckpoint('weights.h5', verbose=1, monitor='loss', save_best_only=True,
                                         mode='auto')
# Train.
history = model.fit(X, y, epochs=epochs, verbose=1, callbacks=[checkpoint])

# Plot training history.
losses = history.history['loss']
if plot_loss:
    plt.plot([i for i in range(1, len(losses) + 1)], losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.show()
    model.load_weights('weights.h5')

# Test the model (using the testing data only!).
if model_type == 'softmax':
    y_test_predict = [0 if p[0] > 0.5 else 1 for p in model.predict(testing.drop(['REGION'], axis=1))]
    print('Predicted:', y_test_predict)
else:
    y_test_predict = [p[0] for p in model.predict(testing.drop(['REGION'], axis=1))]
    print('Predicted:', y_test_predict)
    y_test_predict_rounded = [round(p, 0) for p in y_test_predict]
    print('Rounded Predicted:', y_test_predict_rounded)

# Print the real labels (to compare against the predicted values).
y_test_real = [1 if l == 'AuroralOval' else 0 for l in testing['REGION']]
print('Real:     ', y_test_real)

# Find the test accuracy.
misclassified = sum([abs(i - j) for i, j in zip(y_test_predict, y_test_real)])
accurate = len(y_test_real) - misclassified
print('\n\nTesting accuracy: {} out of {} ({}%).'.format(accurate, len(y_test_real),
                                                         round(accurate * 100 / len(y_test_real), 2)))

# Confidence.
auroral_correct = 0
polar_correct = 0
auroral_incorrect = 0
polar_incorrect = 0
for i, j in zip(y_test_predict, y_test_real):

    if i == j:

        if i == 0:
            auroral_correct += 1
        elif i == 1:
            polar_correct += 1

    else:

        if i == 0 and j == 1:
            auroral_incorrect += 1

        if i == 1 and j == 0:
            polar_incorrect += 1
print('\n\nAuroral oval correct: {} out of {} ({}%).'.format(auroral_correct, auroral_correct + auroral_incorrect,
                                                             round(auroral_correct * 100 / (
                                                                     auroral_correct + auroral_incorrect), 1)))
print('Auroral oval incorrect: {} out of {} ({}%).'.format(auroral_incorrect, auroral_correct + auroral_incorrect,
                                                           round(auroral_incorrect * 100 / (
                                                                   auroral_correct + auroral_incorrect), 1)))
print('Polar cap correct: {} out of {} ({}%).'.format(polar_correct, polar_correct + polar_incorrect,
                                                      round(polar_correct * 100 / (
                                                              polar_correct + polar_incorrect), 1)))
print('Polar cap incorrect: {} out of {} ({}%).'.format(polar_incorrect, polar_correct + polar_incorrect,
                                                        round(polar_incorrect * 100 / (
                                                                polar_correct + polar_incorrect), 1)))
