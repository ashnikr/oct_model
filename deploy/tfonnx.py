import tensorflow as tf
import tf2onnx
from tensorflow.keras import Input, Model

# Load models
vgg_model = tf.keras.models.load_model(
    r"C:\Users\Antino\OneDrive\Desktop\oct_eye\model\3model_combined.h5", compile=False
)
eff_model = tf.keras.models.load_model(
    r"C:\Users\Antino\OneDrive\Desktop\oct_eye\model\oct_eye_model_combined.h5", compile=False
)

# Rename layers to avoid conflicts
for layer in vgg_model.layers:
    layer._name = "vgg_" + layer.name
for layer in eff_model.layers:
    layer._name = "eff_" + layer.name

# Re-wrap the models with unique names to avoid conflicts during ONNX export
vgg_model = Model(inputs=vgg_model.input, outputs=vgg_model.output, name="vgg_model")
eff_model = Model(inputs=eff_model.input, outputs=eff_model.output, name="eff_model")

# Define input tensor
input_tensor = Input(shape=(224, 224, 3), name="input")

# Get outputs from each model
vgg_output = vgg_model(input_tensor)
eff_output = eff_model(input_tensor)

# Expand EfficientNet output from 4 to 8 classes using a Lambda layer
def expand_effnet(x):
    zeros = tf.zeros_like(x[:, :1])  # shape: (None, 1)
    return tf.concat([
        zeros,          # AMD
        x[:, 0:1],      # CNV
        zeros,          # CSR
        x[:, 1:2],      # DME
        zeros,          # DR
        x[:, 2:3],      # DRUSEN
        zeros,          # MH
        x[:, 3:4]       # NORMAL
    ], axis=1)

eff_expanded = tf.keras.layers.Lambda(expand_effnet, name="expand_effnet")(eff_output)

# Combine predictions by averaging
ensemble_output = tf.keras.layers.Average(name="ensemble_avg")([vgg_output, eff_expanded])

# Define ensemble model
ensemble_model = Model(inputs=input_tensor, outputs=ensemble_output, name="ensemble_model")

# Convert to ONNX
spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
output_path = r"C:\Users\Antino\OneDrive\Desktop\oct_eye\model\ensemble_model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(ensemble_model, input_signature=spec, output_path=output_path)

print("✅ Combined ensemble model exported to:", output_path)
