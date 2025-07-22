# ✅ 1. Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ✅ 2. Paths & Class Definitions
sample_dir = r"C:\Users\Antino\OneDrive\Desktop\oct_eye\main\DRUSEN.jpeg"
img_size = (224, 224)
print("Starting.........\n")
vgg8_classes = ['AMD', 'CNV', 'CSR', 'DME', 'DR', 'DRUSEN', 'MH', 'NORMAL']
eff4_classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
eff_to_vgg_indices = [vgg8_classes.index(cls) for cls in eff4_classes]
class_to_idx = {cls: idx for idx, cls in enumerate(vgg8_classes)}
print("Loading Model Combined......\n")
# ✅ 3. Load Models
vgg_model = load_model(r"C:\Users\Antino\OneDrive\Desktop\oct_eye\model\3model_combined.h5", compile=False)
print("Loading OCT Model Combined.....")
eff_model = load_model(r"C:\Users\Antino\OneDrive\Desktop\oct_eye\model\oct_eye_model_combined.h5", compile=False)
print("\nLoading Images...........")
# ✅ 4. Load & Preprocess All Test Images
X = []
y_true = []
image_info = []  # (image_path, true_label)

print("📥 Loading and preprocessing test images...")
for cls in os.listdir(sample_dir):
    cls_dir = os.path.join(sample_dir, cls)
    if not os.path.isdir(cls_dir) or cls not in vgg8_classes:
        continue
    for fname in os.listdir(cls_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(cls_dir, fname)
            img = load_img(img_path, target_size=img_size)
            img = img_to_array(img) / 255.0
            X.append(img)
            y_true.append(class_to_idx[cls])
            image_info.append((img_path, cls))

X = np.array(X)
y_true = np.array(y_true)

# ✅ 5. Batch Predictions
print("🔍 Running batch predictions...")
vgg_preds = vgg_model.predict(X, verbose=1)  # Shape: (N, 8)
eff_preds = eff_model.predict(X, verbose=1)  # Shape: (N, 4)

# ✅ 6. Expand EfficientNet Predictions to 8-class Space
eff_expanded = np.zeros_like(vgg_preds)  # Shape: (N, 8)
for i, eff_idx in enumerate(eff_to_vgg_indices):
    eff_expanded[:, eff_idx] = eff_preds[:, i]

# ✅ 7. Ensemble Average and Final Prediction
avg_preds = (vgg_preds + eff_expanded) / 2.0
y_pred = np.argmax(avg_preds, axis=1)

# ✅ 8. Evaluation Metrics
correct = np.sum(y_pred == y_true)
total = len(y_true)
accuracy = correct / total * 100

print(f"\n✅ Total Images Tested: {total}")
print(f"🎯 Correct Predictions: {correct}")
print(f"❌ Incorrect Predictions: {total - correct}")
print(f"📈 Accuracy: {accuracy:.2f}%")

# ✅ 9. Classification Report
print("\n📊 Ensemble Classification Report:")
print(classification_report(y_true, y_pred, target_names=vgg8_classes))

# ✅ 10. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=vgg8_classes, yticklabels=vgg8_classes)
plt.title('Confusion Matrix - Ensemble (EfficientNetB0 + VGG16)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ✅ 11. Optional: Show a Few Predictions
import random
print("\n🖼 Showing some sample predictions:")

for i in random.sample(range(total), min(6, total)):
    img_path, true_label = image_info[i]
    pred_label = vgg8_classes[y_pred[i]]
    confidence = avg_preds[i][y_pred[i]]

    img = load_img(img_path, target_size=img_size)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"True: {true_label} | Pred: {pred_label}\nConf: {confidence:.2f}")
    plt.show()
