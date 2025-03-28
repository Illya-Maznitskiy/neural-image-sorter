from flask import Flask, request, render_template, jsonify
from classify import classify_image
from PIL import Image
import io
from waitress import serve
import os


app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static",
)


# Print working directory
print("🚀 Starting Flask App...")
print("Current Directory:", os.getcwd())

# Check main directory contents
print("📂 Files in Root Directory:", os.listdir("."))

# Ensure model directory exists
model_dir = os.path.abspath("/app/backend/model/model.h5")
print(f"📌 Model directory absolute path: {model_dir}")

try:
    print("📂 Files in Model Directory:", os.listdir(model_dir))
except Exception as e:
    print(f"⚠️ Error accessing model directory: {e}")


@app.route("/", methods=["GET"])
def home():
    print("🔗 Home page accessed")
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            print("❌ Error: No image file uploaded.")
            return jsonify({"error": "No image uploaded"}), 400

        image_file = request.files["image"]

        # Debugging file info
        print(f"📸 Uploaded Image: {image_file.filename}")
        print(f"🔍 File Type: {image_file.content_type}")

        image = Image.open(io.BytesIO(image_file.read()))

        # Print Image Details
        print(
            f"🖼️ Image Format: {image.format}, Mode: {image.mode}, Size: {image.size}"
        )

        # Ensure RGB mode (handling PNG transparency issues)
        if image.mode != "RGB":
            print("🎨 Converting Image to RGB mode")
            image = image.convert("RGB")

        # Resize image
        print("📏 Resizing Image to (128,128)")
        image = image.resize((128, 128))

        # Run classification
        print("🧠 Running classification...")
        prediction, confidence = classify_image(image)

        confidence = round(float(confidence) * 100, 2)
        print(f"✅ Prediction: {prediction}, Confidence: {confidence}%")

        return jsonify({"prediction": prediction, "confidence": confidence})

    except Exception as e:
        print(f"❌ Error in predict route: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/result")
def result():
    prediction = request.args.get("prediction")
    confidence = request.args.get("confidence")
    if prediction == "cats":
        emoji = "😸"
    else:
        emoji = "🐶"

    print(
        f"📊 Result Page - Prediction: {prediction}, Confidence: {confidence}%"
    )

    return render_template(
        "result.html",
        prediction=prediction + " " + emoji,
        confidence=confidence,
    )


if __name__ == "__main__":
    print("Starting the app with Waitress...")
    app.debug = True  # Enable debug mode to capture more logs
    serve(app, host="0.0.0.0", port=5000)
