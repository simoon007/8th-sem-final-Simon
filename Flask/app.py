from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from PIL import Image
import torch
import json
import difflib
import re
import cv2
import numpy as np
from ultralytics import YOLO
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    AutoImageProcessor,
    AutoModelForImageClassification,
)
import threading
from fuzzywuzzy import fuzz, process

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# ========= MODEL LOADING =========
yolo_model = YOLO("/Users/utsav/Desktop/untitled folder 2/finetune.best.pt")

model_path = "simonneupane/gpt-finetuned-recipes"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
model.eval()

image_model_path = "Utsav201247/food_recognition"
processor = AutoImageProcessor.from_pretrained(image_model_path)
image_model = AutoModelForImageClassification.from_pretrained(image_model_path)
image_model.eval()

# ========= LOAD LOCAL JSON =========
with open("file.json", "r") as f:
    recipes_list = json.load(f)

local_recipes = {
    (r.get("dishName") or r.get("name", "")).lower(): r
    for r in recipes_list
    if r.get("dishName") or r.get("name")
}

# ========= UTILS =========
def parse_recipe_text(text):
    def extract_field(field_name):
        pattern = re.compile(rf"{field_name}:\s*(.*)", re.IGNORECASE)
        match = pattern.search(text)
        return match.group(1).strip() if match else None

    cuisine = extract_field("Cuisine")
    prep_time = extract_field("Prep Time")
    cook_time = extract_field("Cook Time")
    servings_raw = extract_field("Servings")
    servings = int(servings_raw) if servings_raw and servings_raw.isdigit() else servings_raw

    ingredients_match = re.search(r"Ingredients:\n(.*?)\n\n", text, re.DOTALL | re.IGNORECASE)
    ingredients_text = ingredients_match.group(1) if ingredients_match else ""
    ingredients = [line.strip("- ").strip() for line in ingredients_text.strip().splitlines() if line.strip()]

    steps_match = re.search(r"Steps:\n(.*)", text, re.DOTALL | re.IGNORECASE)
    steps_text = steps_match.group(1) if steps_match else ""
    steps = []
    for line in steps_text.strip().splitlines():
        line = line.strip()
        if re.match(r"^\d+\.", line):
            step_text = re.sub(r"^\d+\.\s*", "", line)
            steps.append(step_text)
    if not steps and steps_text:
        steps = [line.strip() for line in steps_text.strip().splitlines() if line.strip()]

    dishname_match = re.search(r"Recipe:\s*(.*)", text, re.IGNORECASE)
    dish_name = dishname_match.group(1).strip() if dishname_match else None

    return {
        "dishName": dish_name,
        "cuisine": cuisine,
        "prep_time": prep_time,
        "cook_time": cook_time,
        "servings": servings,
        "ingredients": ingredients,
        "steps": steps,
    }

def format_recipe_response(json_recipe):
    dish_name = json_recipe.get("dishName") or json_recipe.get("name") or "Unknown"
    recipe_data = json_recipe.get("recipe")

    if isinstance(recipe_data, dict):
        combined = {**json_recipe, **recipe_data}
        combined["dishName"] = dish_name
        combined["source"] = "local"
        return combined
    elif isinstance(recipe_data, str):
        parsed = parse_recipe_text(recipe_data)
        parsed["dishName"] = parsed.get("dishName") or dish_name
        parsed["source"] = "local"
        return parsed
    else:
        result = json_recipe.copy()
        result["dishName"] = dish_name
        result["source"] = "local"
        return result

def format_recipe_as_text(recipe):
    lines = [f"Dish Name: {recipe.get('dishName', 'Unknown')}"]
    if recipe.get("cuisine"): lines.append(f"Cuisine: {recipe['cuisine']}")
    if recipe.get("prep_time"): lines.append(f"Prep Time: {recipe['prep_time']}")
    if recipe.get("cook_time"): lines.append(f"Cook Time: {recipe['cook_time']}")
    if recipe.get("servings"): lines.append(f"Servings: {recipe['servings']}")
    lines.append("")

    ingredients = recipe.get("ingredients", [])
    if ingredients:
        lines.append("Ingredients:")
        for ing in ingredients:
            lines.append(f"- {ing}")
        lines.append("")
    else:
        lines.append("Ingredients: Not available\n")

    steps = recipe.get("steps", [])
    if steps:
        lines.append("Steps:")
        for i, step in enumerate(steps, 1):
            lines.append(f"{i}. {step}")
    else:
        lines.append("Steps: Not available")

    return "\n".join(lines)

def generate_recipe(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_args = {
        "inputs": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_new_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
        "repetition_penalty": 1.2,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }
    outputs = model.generate(**generation_args)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()

# ========= IMPROVED RECIPE MATCHING =========
def find_recipe_from_json(query, ingredients=None):
    query = query.lower().strip()
    ingredients = [ing.lower() for ing in ingredients] if ingredients else []

    # 1. Match detected ingredients to dish names
    if ingredients:
        # Check for exact match of any ingredient in dish names
        for ing in ingredients:
            if ing in local_recipes:
                return local_recipes[ing]
        
        # Check for substring match of any ingredient in dish names
        for ing in ingredients:
            for dish_name in local_recipes.keys():
                if ing in dish_name:
                    return local_recipes[dish_name]

    # 2. Exact match of query
    if query in local_recipes:
        return local_recipes[query]

    # 3. Fuzzy match
    best_match, score = process.extractOne(query, local_recipes.keys(), scorer=fuzz.partial_ratio)
    if score > 80:
        return local_recipes[best_match]

    # 4. Substring match
    for name in local_recipes.keys():
        if query in name:
            return local_recipes[name]

    # 5. difflib close match
    match = difflib.get_close_matches(query, local_recipes.keys(), n=1, cutoff=0.75)
    if match:
        return local_recipes[match[0]]

    # 6. Ingredient-based match
    if ingredients:
        ingredients_set = set(ingredients)
        best_match = None
        best_score = 0
        for name, recipe in local_recipes.items():
            recipe_ings = recipe.get("ingredients", [])
            if not recipe_ings: continue
            recipe_ings_set = set(ing.lower() for ing in recipe_ings)
            match_count = len(ingredients_set & recipe_ings_set)
            match_percent = (match_count / len(recipe_ings_set)) * 100
            if match_percent > best_score:
                best_score = match_percent
                best_match = recipe
        if best_match and best_score > 30:
            return best_match

    return None

# ========= ROUTES =========
@app.route('/chat', methods=['POST', 'OPTIONS'])
def chat():
    if request.method == 'OPTIONS':
        return Response(status=200)

    user_input = request.json.get("message", "").strip()
    if not user_input:
        return Response("Please enter a message.", mimetype='text/plain')

    # Try matching by name or ingredients (comma-separated)
    ingredients_text = [x.strip() for x in user_input.lower().split(",") if x.strip()]
    json_recipe = find_recipe_from_json(user_input, ingredients=ingredients_text)
    if json_recipe:
        clean = format_recipe_response(json_recipe)
        return Response(format_recipe_as_text(clean), mimetype="text/plain")

    # GPT fallback
    prompt = f"You are a cooking assistant.\nUser: {user_input}\nAssistant:"
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    def generate():
        model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
            temperature=0.9,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            streamer=streamer,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    threading.Thread(target=generate).start()
    return Response(streamer, mimetype="text/plain")

@app.route('/classify', methods=['POST', 'OPTIONS'])
def classify():
    if request.method == 'OPTIONS':
        return Response(status=200)

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        image = Image.open(request.files['image']).convert('RGB')
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = image_model(**inputs)

        predicted_index = torch.argmax(outputs.logits, dim=-1).item()
        confidence = torch.softmax(outputs.logits, dim=-1)[0][predicted_index].item()
        class_labels = image_model.config.id2label
        predicted_label = class_labels.get(predicted_index, "Unknown Dish")

        if confidence < 0.1:
            return jsonify({"error": "Low confidence in classification"}), 400

        json_recipe = find_recipe_from_json(predicted_label)
        if json_recipe:
            clean = format_recipe_response(json_recipe)
            return jsonify({
                "recipe": format_recipe_as_text(clean),
                "source": "local",
                "confidence": confidence,
                "dishName": predicted_label
            })

        # GPT fallback
        prompt = f"You are a cooking assistant. Give a recipe for {predicted_label} with all details."
        generated = generate_recipe(prompt)
        return jsonify({
            "dishName": predicted_label,
            "confidence": confidence,
            "source": "gpt",
            "recipe": generated
        })

    except Exception as e:
        return jsonify({"error": f"Image classification failed: {str(e)}"}), 500

@app.route('/detect_ingredients', methods=['POST', 'OPTIONS'])
def detect_ingredients():
    if request.method == 'OPTIONS':
        return Response(status=200)

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    try:
        file_bytes = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        results = yolo_model(image)
        boxes = results[0].boxes
        class_names = results[0].names if hasattr(results[0], 'names') else yolo_model.names
        detected = [class_names[int(box.cls[0])] for box in boxes]
        detected_lower = [x.lower() for x in detected]

        # Remove duplicate ingredients while preserving order
        unique_detected = []
        seen = set()
        for item in detected:
            if item.lower() not in seen:
                seen.add(item.lower())
                unique_detected.append(item)
        detected = unique_detected
        detected_lower = [x.lower() for x in detected]

        # Form query from detected ingredients
        query = ", ".join(detected)
        json_recipe = find_recipe_from_json(query, ingredients=detected_lower)

        if json_recipe:
            # Format the recipe only once
            formatted_recipe = format_recipe_response(json_recipe)
            recipe_text = format_recipe_as_text(formatted_recipe)
            
            # Build the response with detection info
            response = {
                "message": f"By using YOLO, I found the following ingredient(s): {', '.join(detected)}",
                "recipe": recipe_text,
                "ingredients_detected": detected,
                "source": "local"
            }
            return jsonify(response)

        # GPT fallback
        prompt = f"You are a cooking assistant. I have: {', '.join(detected)}. Suggest a full recipe."
        generated = generate_recipe(prompt)
        return jsonify({
            "message": f"By using YOLO, I found the following ingredient(s): {', '.join(detected)}",
            "recipe": generated,
            "ingredients_detected": detected,
            "source": "gpt"
        })

    except Exception as e:
        return jsonify({'error': f'Ingredient detection failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5050, debug=True)