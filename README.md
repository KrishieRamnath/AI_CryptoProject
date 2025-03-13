import random
import numpy as np
import string
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from math import gcd


# ------------------- CIPHER IMPLEMENTATIONS -------------------

def Caesar(plaintext, key):
    alphabet = string.ascii_uppercase
    return ''.join(alphabet[(alphabet.index(c) + key) % 26] if c in alphabet else c for c in plaintext)


def Affine(plaintext, a, b):  # Add a and b as separate parameters
    alphabet = string.ascii_uppercase
    return ''.join(alphabet[(a * alphabet.index(c) + b) % 26] if c in alphabet else c for c in plaintext)


def Vignere(plaintext, key):
    plaintext = plaintext.upper()  # Convert to uppercase
    alphabet = string.ascii_uppercase
    key = (key * (len(plaintext) // len(key) + 1))[:len(plaintext)]
    return ''.join(alphabet[(alphabet.index(p) + alphabet.index(k)) % 26] if p in alphabet else p for p, k in zip(plaintext, key))


def Transposition(plaintext, key):
    key = max(2, key % len(plaintext))  # Ensure key is valid
    num_cols = (len(plaintext) + key - 1) // key  # Determine the number of columns
    grid = [''] * num_cols  # Create empty grid

    for i, char in enumerate(plaintext):
        grid[i % num_cols] += char  # Fill columns with characters

    return ''.join(grid)  # Read column-wise


# ------------------- DATA GENERATION -------------------

def generate_random_text(length=20):
    return ''.join(random.choices(string.ascii_uppercase, k=length))

# Randomly pick a cipher from the available ciphers
def generate_random_caesar_shift():
    """Generate a random shift for the Caesar cipher."""
    return random.randint(1, 25)  # Avoid shift=0 as it's not encryption

def generate_random_affine_keys():
    """Generate a random (a, b) pair where a is coprime with 26."""
    possible_a = [num for num in range(1, 26) if gcd(num, 26) == 1]  # Only values coprime with 26
    a = random.choice(possible_a)
    b = random.randint(0, 25)  # b can be any value 0-25
    return a, b

def generate_random_vigenere_key(length=6):
    """Generate a random Vigenère cipher key of given length."""
    return ''.join(random.choice(string.ascii_uppercase) for _ in range(length))

def generate_random_transposition_key(text_length):
    """Generate a random transposition key between 2 and half the text length."""
    return random.randint(2, max(2, text_length // 2))

ciphers = {
    "Caesar": lambda text: Caesar(text, generate_random_caesar_shift()),
    "Affine": lambda text: Affine(text, *generate_random_affine_keys()),  # * Unpacks (a, b)
    "Vigenère": lambda text: Vignere(text, generate_random_vigenere_key()),
    "Transposition": lambda text: Transposition(text, generate_random_transposition_key(len(text)))
}

def generate_training_data(num_samples=10000):
    plaintexts = []
    ciphertexts = []
    labels = []

    for _ in range(num_samples):
        plaintext = generate_random_text(20)  # Generate random text
        cipher_name, cipher_function = random.choice(list(ciphers.items()))  # Choose a cipher

        ciphertext = cipher_function(plaintext)  # Encrypt text
        plaintexts.append(plaintext)
        ciphertexts.append(ciphertext)
        labels.append(cipher_name)  # Store cipher name as label

    return plaintexts, ciphertexts, labels  # Ensure 3 values are returned

# ------------------- PREPROCESSING -------------------

def text_to_numbers(text, max_length=20):
    return np.array([ord(c) - 65 for c in text.ljust(max_length, 'A')[:max_length]])


def preprocess_data(ciphertexts, labels):
    # Convert ciphertexts into numerical format
    X = np.array([text_to_numbers(text) for text in ciphertexts])  # X = vectorized_texts

    # Convert labels to categorical values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)  # Convert labels to numbers
    y = to_categorical(y)  # One-hot encode the labels

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val, label_encoder


# ------------------- MODEL -------------------

def build_model(input_shape, num_classes):
    model = Sequential([
        Embedding(input_dim=26, output_dim=128),  # Define input_length explicitly
        Bidirectional(LSTM(256, return_sequences=True)),
        Dropout(0.3),
        Bidirectional(LSTM(128)),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=0.0005)  # 🔽 Reduced from 0.01 to 0.0005
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


# ------------------- TRAINING -------------------

plaintext, ciphertexts, labels = generate_training_data(num_samples=10000)
X_train, X_val, y_train, y_val, label_encoder = preprocess_data(ciphertexts, labels)  # Ensure X is processed inside this function

# Split into training and validation sets
from sklearn.model_selection import train_test_split


model = build_model(X_train.shape, len(set(labels)))

early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_val, y_val))


# ------------------- PREDICTION -------------------

def predict_cipher(ciphertext):
    vectorized = np.array(text_to_numbers(ciphertext)).reshape(1, -1)
    prediction = model.predict(vectorized)[0]
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]

    print(f"Ciphertext: {ciphertext}")
    print(f"Predicted Cipher: {predicted_label}")


# ------------------- TESTING -------------------

test_text = generate_random_text(20)
# Generate a random plaintext
test_plaintext = generate_random_text(length=10)

# Select a Random Cipher
chosen_cipher_name = random.choice(["Caesar", "Affine", "Vigenère", "Transposition"])

if chosen_cipher_name == "Caesar":
    caesar_shift = generate_random_caesar_shift()
    cipher_function = lambda text: Caesar(text, caesar_shift)
    print(f"Caesar Cipher Chosen - Shift: {caesar_shift}")

elif chosen_cipher_name == "Affine":
    a, b = generate_random_affine_keys()
    cipher_function = lambda text: Affine(text, a, b)
    print(f"Affine Cipher Chosen - a: {a}, b: {b}")

elif chosen_cipher_name == "Vigenère":
    vigenere_key = generate_random_vigenere_key()
    cipher_function = lambda text: Vignere(text, vigenere_key)
    print(f"Vigenère Cipher Chosen - Key: {vigenere_key}")

elif chosen_cipher_name == "Transposition":
    transposition_key = generate_random_transposition_key(len(test_plaintext))
    cipher_function = lambda text: Transposition(text, transposition_key)
    print(f"Transposition Cipher Chosen - Key: {transposition_key}")

# Encrypt the test plaintext
test_ciphertext = cipher_function(test_plaintext)

# Run the prediction
print(f"\n**Random Test Case**")
print(f"Plaintext: {test_plaintext}")
print(f"Actual Cipher: {chosen_cipher_name}")
print(f"Ciphertext: {test_ciphertext}\n")

predict_cipher(test_ciphertext)
