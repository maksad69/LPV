# =====================================================
# 1. IMPORT LIBRARIES
# =====================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical


# =====================================================
# 2. LOAD DATASET (LOCAL FILE)
# =====================================================
cols = ['lettr','x-box','y-box','width','height','onpix','x-bar','y-bar',
        'x2bar','y2bar','xybar','x2ybr','xy2br','x-ege','xegvy','y-ege','yegvx']

df = pd.read_csv("letter-recognition.data", header=None, names=cols)


# =====================================================
# 3. PREPROCESSING
# =====================================================
X = df.drop('lettr', axis=1).values # .values Converts dataframe into numpy array
y = df['lettr'].values

# Encode labels (A–Z → 0–25)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# One-hot encoding
y_cat = to_categorical(y_encoded, num_classes=26)  # Converts numbers into binary vectors to remove number order problem

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =====================================================
# 4. BUILD MODEL
# =====================================================
model = Sequential()

model.add(Dense(128, activation='relu', input_dim=16))
model.add(Dense(64, activation='relu'))
model.add(Dense(26, activation='softmax'))   # Softmax It gives probability for each class and which class has high probability this is output predicted


# =====================================================
# 5. COMPILE MODEL
# =====================================================
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# =====================================================
# 6. TRAIN MODEL
# =====================================================
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,   # data is divided into small groups (batches) and Each batch has 64 samples
    validation_split=0.1,
    verbose=1
)


# =====================================================
# 7. VISUALIZATION (Accuracy & Loss)
# =====================================================

# Accuracy Graph
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Loss Graph
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# =====================================================
# 8. EVALUATE MODEL
# =====================================================
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)


# =====================================================
# 9. PREDICT SAMPLE
# =====================================================
sample = np.array([[2,8,3,5,1,8,13,0,6,6,10,8,0,8,0,8]])
sample = scaler.transform(sample)

pred = model.predict(sample)
pred_class = np.argmax(pred) # Finds index of highest value

print("Predicted Letter:", le.classes_[pred_class])