# Meal Mastery: Deep Learning Web-based System for Ingredient Recognition and Recipe Recommendation

This project delivers an **AI-driven multimodal application** designed to revolutionize cooking by seamlessly connecting computer vision (CV) and natural language processing (NLP). The system accepts an image of raw ingredients or a prepared dish and instantly provides context-aware recipe suggestions and cooking instructions.

## 🚀 Key Features

* **End-to-End Deep Learning Pipeline:** Integrates multiple state-of-the-art models for intuitive food recognition and dynamic recipe generation without requiring manual input.
* **Web-Based Application:** The system is deployed via a **Flask** backend (for model serving) and a **Next.js (React)** frontend for a smooth user experience.

## 🧠 Core Technology Components

The system is built on the **Transformer architecture** and utilizes three specialized deep learning models:

### 1. Ingredient Recognition (Object Detection)
* **Model:** **YOLOv8 (You Only Look Once)**.
* **Function:** Performs real-time object detection to precisely localize and classify **raw food ingredients** from user-uploaded images, supporting multi-ingredient detection.

### 2. Dish Classification (Image Classification)
* **Model:** **Vision Transformer (ViT)**.
* **Function:** Extracts high-level features for the classification of **prepared dishes** (e.g., identifying the type of meal) by leveraging its self-attention mechanism, achieving an accuracy of **95.65%** on the evaluation set.

### 3. Recipe Generation (Natural Language Generation)
* **Model:** Fine-tuned **GPT-2 Transformer**.
* **Function:** Processes the detected ingredients or a user-provided dish name to generate **coherent, step-by-step cooking instructions** and suggest complementary recipes.