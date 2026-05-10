# =====================================================
# 1. IMPORT LIBRARIES
# =====================================================
import pandas as pd  
import seaborn as sns
import matplotlib.pyplot as plt

# Update these two lines to the new Keras structure:
from keras.datasets import imdb 
from keras.utils import pad_sequences           # Updated import
from keras.preprocessing.text import Tokenizer   # This usually works, but see below if it fails
# Alternative if the above still fails: from tensorflow.keras.preprocessing.text import Tokenizer

from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split 

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
df = pd.read_csv('your_dataset.csv') 

# Step 2: Tokenize text 
tokenizer = Tokenizer(num_words=NUM_WORDS)
tokenizer.fit_on_texts(df['review'].astype(str)) # Added astype(str) to prevent errors with empty rows
sequences = tokenizer.texts_to_sequences(df['review'].astype(str))

# Step 3: Pad sequences
data = pad_sequences(sequences, maxlen=MAX_LEN)
labels = df['sentiment'].values 

# Step 4: Split into train/test
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
"""

# =====================================================
# 3. PADDING
# =====================================================
# We use the updated pad_sequences from keras.utils
X_train = pad_sequences(X_train, maxlen=MAX_LEN) 
X_test = pad_sequences(X_test, maxlen=MAX_LEN) 

# =====================================================
# 4. BUILD MODEL (Rest of the code remains the same)
# =====================================================
model = Sequential([
    Embedding(input_dim=NUM_WORDS, output_dim=128),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=10, batch_size=512, validation_split=0.2, verbose=1)

# Evaluate and Plot
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

y_pred = (model.predict(X_test) > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.show()
