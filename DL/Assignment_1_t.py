# =========================================
# 1. IMPORT LIBRARIES
# =========================================
import numpy as np   # Numerical python library  used for arrays,math calculations and np is short nme for numpy
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler   # sklearn is a library and from that library we use a tool called StandardSacler sklearn = Big toolbox 🧰 preprocessing = One section inside the toolbox StandardScaler = One tool inside that section
from keras.models import Sequential
from keras.layers import Dense


# =========================================
# 2. LOAD DATASET (California instead of Boston)
# =========================================
# from sklearn.datasets import fetch_california_housing
#
# # if CSV file is not given
# # housing = fetch_california_housing()
# #
# # data = pd.DataFrame(housing.data, columns=housing.feature_names)
# # data['PRICE'] = housing.target
# #
# # print("First 5 rows:\n", data.head())

#if CSV file is given
# import pandas as pd
data = pd.read_csv("boston_housing.csv")
print(data.head())

# =========================================
# 3. FEATURE SELECTION
# =========================================
X = data.iloc[:, :-1]
y = data['MEDV']

# =========================================
# 4. TRAIN-TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42  # ensures same data split every time and 42 is random number
)

# =========================================
# 5. FEATURE SCALING
# =========================================

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# =========================================
# 6. BUILD DNN MODEL
# =========================================

model = Sequential()

model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))  # Dense means Every neuron is connected to every neuron in the next layer
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu')) # A neoron actually does this output = activation( (x1*w1 + x2*w2 + ... + xn*wn) + bias )
model.add(Dense(16, activation='relu'))
model.add(Dense(1))  # Output layer

# =========================================
# 7. COMPILE MODEL
# =========================================
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])   #optimizer decides How to change weights to reduce error

# =========================================
# 8. TRAIN MODEL
# =========================================
history = model.fit(
    X_train, y_train,     # training data
    epochs=100,           # How much time it sees data
    validation_split=0.1, # 10% of training data is used for validation
    verbose=1             # shows epochs on screen
)

# =========================================
# 9. EVALUATE MODEL
# =========================================
loss, mae = model.evaluate(X_test, y_test)

print("\n--- Deep Neural Network Result ---")
print("MSE:", loss)
print("MAE:", mae)

# =========================================
# 10. PLOT GRAPH
# =========================================
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(['Train', 'Validation'])
plt.show()

# =========================================
# 11. PREDICTION ON NEW DATA
# =========================================
new_data = np.array([[0.1, 10.0, 5.0, 0, 0.4, 6.0, 50, 6.0, 1, 400, 20, 300, 10]])
new_data = sc.transform(new_data)

prediction = model.predict(new_data)

print("\nPredicted House Price:", prediction)