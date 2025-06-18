import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import wikipedia

# =============== Config ===============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 200  # Adjust this according to the model
IMG_SIZE = 224

# =============== Transforms ===============
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


def get_class_description(label):
    try:
        summary = wikipedia.summary(label, sentences=2)
        return summary
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Too many results. Try narrowing it down: {e.options[:3]}"
    except wikipedia.exceptions.PageError:
        return "No description available for this label."

# =============== Load Class Index to Label Mapping ===============
def load_class_labels(train_dir, words_path):
    class_folders = sorted(os.listdir(train_dir))
    wnid_to_label = {}

    with open(words_path, "r") as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                wnid, label = parts
                wnid_to_label[wnid] = label

    # Map index -> readable label
    class_labels = [wnid_to_label.get(wnid, wnid) for wnid in class_folders]
    return class_labels

# =============== Load Model ===============
def load_model(checkpoint_path):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)

    # Ensure the classifier layers match the trained model
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.classifier[1].in_features, 512),
        nn.ReLU(),
        nn.Linear(512, NUM_CLASSES)
    )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# =============== Predict ===============
def predict_image(model, image_path, class_labels):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
        predicted_idx = pred.item()
        predicted_label = class_labels[predicted_idx]

    return predicted_label

# =============== Streamlit App ===============
# Streamlit UI
st.set_page_config(page_title="Image Classifier", layout="centered")
st.title("🖼️ Image Recognition")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load class labels (Use your paths)
    train_dir = "tiny-imagenet-200\Train"
    words_txt = "tiny-imagenet-200\words.txt"
    class_labels = load_class_labels(train_dir, words_txt)

    # Load model
    model_path = "best_effnetb0_augmented.pth"  # Path to the saved model
    model = load_model(model_path)

    # Predict
    result = predict_image(model, uploaded_file, class_labels)
    st.success(f"Predicted Class: {result}")

    description = get_class_description(result)
    st.info(f"📝 Description: {description}")

