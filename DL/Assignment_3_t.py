# =====================================================
# 1. IMPORT LIBRARIES
# =====================================================
import seaborn as sns
import matplotlib.pyplot as plt

from keras.datasets import imdb # here imdb is a dataset we imported here to use from the datasets module of Kears API and Dataset is already in number format means Review column words are already converted into numbers and output column ha snumber value output
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense

from sklearn.metrics import confusion_matrix



# =====================================================
# 2. LOAD DATASET
# =====================================================
NUM_WORDS = 10000  # Use only the top 10,000 most frequent words in the dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=NUM_WORDS)


# =====================================================
# 3. PADDING
# =====================================================
MAX_LEN = 500

X_train = pad_sequences(X_train, maxlen=MAX_LEN) # ❌ Neural networks need same size input
X_test = pad_sequences(X_test, maxlen=MAX_LEN) # ✔️ So we make all reviews equal length
                                                # Short reviews added 0 to make lenth sameand if more than lenth values in review then last 500 values are taken using padding

# =====================================================
# 4. BUILD MODEL
# =====================================================
model = Sequential()

model.add(Embedding(input_dim=NUM_WORDS, output_dim=128)) # This layer converts each word index into a 128-dimensional vector representation.
model.add(GlobalAveragePooling1D()) # 👉 It takes all those vectors of each word 👉 And calculates their average
# Word 1 → [1, 2]
# Word 2 → [3, 4]      => Average = [(1+3+5)/3 , (2+4+6)/3] = [3, 4]
# Word 3 → [5, 6]
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# =====================================================
# 5. COMPILE MODEL
# =====================================================
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Used when output is 2 classes (0)=Positive and (1)=Negative and How wrong the prediction is
              metrics=['accuracy'])


# =====================================================
# 6. TRAIN MODEL
# =====================================================
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=512,
    validation_split=0.2,
    verbose=1
)


# =====================================================
# 7. EVALUATE MODEL
# =====================================================
loss, accuracy = model.evaluate(X_test, y_test)

print("Test Accuracy:", accuracy)


# =====================================================
# 8. CONFUSION MATRIX
# =====================================================
y_pred = (model.predict(X_test) > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()