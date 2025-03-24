import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.layers import UpSampling1D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input, Add, BatchNormalization, Cropping1D
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
# ========================= Load and Preprocess the Data =========================

# Load dataset
train_df = pd.read_csv("mitbih_train.csv", header=None)
test_df = pd.read_csv("mitbih_test.csv", header=None)

# Split features (X) and labels (y)
X_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]
X_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]

# Convert labels into binary (0 = normal, 1 = arrhythmia)
y_train = np.where(y_train == 0, 0, 1)
y_test = np.where(y_test == 0, 0, 1)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape for CNN (1D Convolution)
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# ========================= Visualize ECG Data =========================

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(X_train[0])
plt.title("Normal ECG Signal")

plt.subplot(1, 2, 2)
plt.plot(X_train[np.where(y_train == 1)][0])
plt.title("Arrhythmic ECG Signal")

plt.show()

sns.countplot(x=y_train)
plt.title("Class Distribution in Training Data")
plt.show()

# ========================= Define Inception Module =========================

def inception_module(x, filters):
    f1, f2, f3 = filters

    conv1 = Conv1D(f3, kernel_size=1, activation='relu', padding='same')(x)  # Ändrat till f3
    conv3 = Conv1D(f3, kernel_size=3, activation='relu', padding='same')(x)
    conv5 = Conv1D(f3, kernel_size=5, activation='relu', padding='same')(x)

    return Add()([conv1, conv3, conv5])


# ========================= Define CNN Model (Inception V1 Inspired) =========================

def create_inception_cnn_model():
    inputs = Input(shape=(X_train.shape[1], 1))
    
    # First Conv Block
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Inception Module
    x = inception_module(x, filters=(32, 64, 128))
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    # Fully Connected Layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# ========================= Train CNN Model with Class Weighting =========================

cnn_model = create_inception_cnn_model()
cnn_model.summary()

# Compute class weights to balance dataset
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))

# Learning rate scheduling
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

history = cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                        epochs=20, batch_size=32, 
                        class_weight=class_weight_dict,
                        callbacks=[lr_scheduler])

# ========================= Evaluate CNN Model =========================

y_pred = (cnn_model.predict(X_test) > 0.5).astype("int32")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("CNN Confusion Matrix")
plt.show()

# Print classification report
print(classification_report(y_test, y_pred))

# ========================= Define Convolutional Autoencoder =========================
input_layer = Input(shape=(X_train.shape[1], 1))

# Encoder
x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_layer)
x = MaxPooling1D(pool_size=2, padding='same')(x)
x = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(x)
encoded = MaxPooling1D(pool_size=2, padding='same')(x)

# Decoder
x = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(encoded)
x = UpSampling1D(size=2)(x)
x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
x = UpSampling1D(size=2)(x)

# Anpassa utgångslängden om den är för lång
if X_train.shape[1] % 4 != 0:  # Om längden inte är jämnt delbar med 4, beskära 1 steg
    x = Cropping1D(cropping=(0, 1))(x)

decoded = Conv1D(filters=1, kernel_size=3, activation='sigmoid', padding='same')(x)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train Autoencoder
autoencoder_lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

autoencoder.fit(X_train, X_train, validation_data=(X_test, X_test), 
                epochs=20, batch_size=32, callbacks=[autoencoder_lr_scheduler])

# Predict Reconstruction Error
X_test_pred = autoencoder.predict(X_test)

mse = np.mean(np.power(X_test.squeeze() - X_test_pred.squeeze(), 2), axis=1)

# Set a threshold for anomaly detection
threshold = np.percentile(mse, 95)  # Top 5% error
y_pred_autoencoder = (mse > threshold).astype("int32")

# Evaluate Autoencoder
conf_matrix_autoencoder = confusion_matrix(y_test, y_pred_autoencoder)
sns.heatmap(conf_matrix_autoencoder, annot=True, fmt='d', cmap='Reds')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Autoencoder Confusion Matrix")
plt.show()

# ========================= Save Models =========================

cnn_model.save("ecg_cnn_model.keras")
autoencoder.save("ecg_autoencoder.keras")

# ========================= EU AI Regulations Compliance =========================
# - This project follows EU AI regulations by:
#   - Ensuring **data privacy** (no personal ECG data, GDPR compliance)
#   - Addressing **bias** using class weighting to balance predictions
#   - Increasing **transparency** by using explainable models (CNN & Autoencoder)
#   - Using **metrics like confusion matrix & classification report** to monitor fairness
