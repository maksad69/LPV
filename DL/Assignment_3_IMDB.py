# =====================================================
# 1. IMPORT LIBRARIES
# =====================================================
import pandas as pd  # Added for local loading
import seaborn as sns
import matplotlib.pyplot as plt

from keras.datasets import imdb 
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer # Added to process local text
from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split # Added for local splitting

# =====================================================
# 2. LOAD DATASET
# =====================================================
NUM_WORDS = 10000 
MAX_LEN = 500

# --- OPTION A: LOADING FROM KERAS (ACTIVE) ---
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=NUM_WORDS)

# --- OPTION B: LOADING LOCALLY (COMMENTED) ---
"""
# Step 1: Load CSV
df = pd.read_csv('your_dataset.csv') # Should have 'review' and 'sentiment' columns

# Step 2: Tokenize text (Converting words to numbers)
tokenizer = Tokenizer(num_words=NUM_WORDS)
tokenizer.fit_on_texts(df['review'])
sequences = tokenizer.texts_to_sequences(df['review'])

# Step 3: Pad sequences
data = pad_sequences(sequences, maxlen=MAX_LEN)
labels = df['sentiment'].values # Ensure sentiment is 0 or 1

# Step 4: Split into train/test
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
"""

# =====================================================
# 3. PADDING
# =====================================================
# Note: If using Option A, we still need to pad. 
# If using Option B, padding is already handled in the commented block above.
X_train = pad_sequences(X_train, maxlen=MAX_LEN) 
X_test = pad_sequences(X_test, maxlen=MAX_LEN) 

# =====================================================
# 4. BUILD MODEL
# =====================================================
model = Sequential()
model.add(Embedding(input_dim=NUM_WORDS, output_dim=128)) 
model.add(GlobalAveragePooling1D()) 
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# =====================================================
# 5. COMPILE MODEL
# =====================================================
model.compile(optimizer='adam',
              loss='binary_crossentropy',
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
print(f"Test Accuracy: {accuracy:.4f}")

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
