import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Step 1: Dataset Preparation
# Define base directories for each category
base_data_dir = "C:/Users/Desktop/Desktop/Thesis/Data/Dataset"
train_dirs = {
    "healthy": os.path.join(base_data_dir, "train", "healthy"),
    "black_pod_rot": os.path.join(base_data_dir, "train", "black_pod_rot"),
    "pod_borer": os.path.join(base_data_dir, "train", "pod_borer"),
}
val_dirs = {
    "healthy": os.path.join(base_data_dir, "val", "healthy"),
    "black_pod_rot": os.path.join(base_data_dir, "val", "black_pod_rot"),
    "pod_borer": os.path.join(base_data_dir, "val", "pod_borer"),
}
test_dirs = {
    "healthy": os.path.join(base_data_dir, "test", "healthy"),
    "black_pod_rot": os.path.join(base_data_dir, "test", "black_pod_rot"),
    "pod_borer": os.path.join(base_data_dir, "test", "pod_borer"),
}

# Set up data generators
datagen = ImageDataGenerator(rescale=1.0 / 255)


def load_data_from_dirs(directories, target_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(rescale=1.0 / 255)
    data = []
    labels = []

    class_map = {label: idx for idx, label in enumerate(directories.keys())}
    for label, path in directories.items():
        for fname in os.listdir(path):
            fpath = os.path.join(path, fname)
            if os.path.isfile(fpath):
                img = plt.imread(fpath)
                img_resized = np.resize(img, target_size + (3,)) / 255.0
                data.append(img_resized)
                labels.append(class_map[label])

    data = np.array(data)
    labels = np.array(labels)
    return data, labels


train_data, train_labels = load_data_from_dirs(train_dirs)
val_data, val_labels = load_data_from_dirs(val_dirs)
test_data, test_labels = load_data_from_dirs(test_dirs)

print("Train shape:", train_data.shape)
print("Validation shape:", val_data.shape)
print("Test shape:", test_data.shape)


# Step 2: Pre-trained Models for Transfer Learning
def build_model(base_model, num_classes):
    # Adapt architecture to match the number of classes
    x = base_model.output
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    predictions = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


# Load pre-trained models
vgg16_base = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
vgg19_base = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

num_classes = len(train_dirs)  # Number of classes
model1 = build_model(vgg16_base, num_classes)
model2 = build_model(vgg19_base, num_classes)

# Step 3: Feature Extraction
for layer in vgg16_base.layers:
    layer.trainable = False
for layer in vgg19_base.layers:
    layer.trainable = False

# Step 4: Train Multiple Models
# Compile models
model1.compile(
    optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
model2.compile(
    optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

history1 = model1.fit(
    train_data,
    train_labels,
    validation_data=(val_data, val_labels),
    epochs=5,
    verbose=1,
)
history2 = model2.fit(
    train_data,
    train_labels,
    validation_data=(val_data, val_labels),
    epochs=5,
    verbose=1,
)


# Training History Visualization
def plot_training_history(history, model_name):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{model_name} Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{model_name} Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()


plot_training_history(history1, "VGG16")
plot_training_history(history2, "VGG19")

# Step 5: Evaluate Individual Models
pred1 = model1.predict(test_data)
pred1_classes = np.argmax(pred1, axis=1)

pred2 = model2.predict(test_data)
pred2_classes = np.argmax(pred2, axis=1)


def evaluate_model(true_labels, predicted_labels, model_name):
    print(f"Metrics for {model_name}:")
    print(f"Accuracy: {accuracy_score(true_labels, predicted_labels)}")
    print(
        f"Classification Report:\n{classification_report(true_labels, predicted_labels)}"
    )
    print(f"Confusion Matrix:\n{confusion_matrix(true_labels, predicted_labels)}")


evaluate_model(test_labels, pred1_classes, "VGG16")
evaluate_model(test_labels, pred2_classes, "VGG19")

# Step 6: Ensemble Learning and Metrics
final_predictions = []
for i in range(len(test_data)):
    votes = [pred1_classes[i], pred2_classes[i]]
    final_predictions.append(np.bincount(votes).argmax())

evaluate_model(test_labels, final_predictions, "Ensemble (Majority Voting)")


# Step 7: Visualization Functions
def plot_confusion_matrix(true_labels, pred_labels, model_name, class_names):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def plot_metrics(true_labels, pred_labels, model_name, class_names):
    precision = precision_score(true_labels, pred_labels, average=None)
    recall = recall_score(true_labels, pred_labels, average=None)
    f1 = f1_score(true_labels, pred_labels, average=None)

    x = range(len(class_names))

    plt.figure(figsize=(12, 6))
    plt.bar(x, precision, width=0.2, label="Precision", align="center")
    plt.bar([p + 0.2 for p in x], recall, width=0.2, label="Recall", align="center")
    plt.bar([p + 0.4 for p in x], f1, width=0.2, label="F1 Score", align="center")

    plt.xticks([p + 0.2 for p in x], class_names)
    plt.title(f"{model_name} Precision, Recall, and F1 Score")
    plt.xlabel("Classes")
    plt.ylabel("Score")
    plt.legend()
    plt.show()


# Step 8: Visualization for Models and Ensemble
class_names = list(train_dirs.keys())

# Confusion matrix
plot_confusion_matrix(test_labels, pred1_classes, "VGG16", class_names)
plot_confusion_matrix(test_labels, pred2_classes, "VGG19", class_names)
plot_confusion_matrix(
    test_labels, final_predictions, "Ensemble (Majority Voting)", class_names
)

# Metrics
plot_metrics(test_labels, pred1_classes, "VGG16", class_names)
plot_metrics(test_labels, pred2_classes, "VGG19", class_names)
plot_metrics(test_labels, final_predictions, "Ensemble (Majority Voting)", class_names)
