from flask import Flask, request, render_template, jsonify
from classify import classify_image
from PIL import Image
import io
from waitress import serve
import os
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static",
)

logging.info("🚀 Starting Flask App...")
logging.info(f"Current Directory: {os.getcwd()}")
logging.info(f"📂 Files in Root Directory: {os.listdir('.')}")


@app.route("/", methods=["GET"])
def home():
    logging.info("🔗 Home page accessed")
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            logging.error("❌ Error: No image file uploaded.")
            return jsonify({"error": "No image uploaded"}), 400

        image_file = request.files["image"]
        logging.info(f"📸 Uploaded Image: {image_file.filename}")
        logging.info(f"🔍 File Type: {image_file.content_type}")

        image = Image.open(io.BytesIO(image_file.read()))
        logging.info(
            f"🖼️ Image Format: {image.format}, "
            f"Mode: {image.mode}, Size: {image.size}"
        )

        if image.mode != "RGB":
            logging.info("🎨 Converting Image to RGB mode")
            image = image.convert("RGB")

        logging.info("📏 Resizing Image to (128,128)")
        image = image.resize((128, 128))

        from classify import model

        if model is None:
            logging.critical("❌ Model not loaded, exiting application.")
            exit(1)

        logging.info("🧠 Running classification...")
        prediction, confidence = classify_image(image)

        confidence = round(float(confidence) * 100, 2)
        logging.info(f"✅ Prediction: {prediction}, Confidence: {confidence}%")

        return jsonify({"prediction": prediction, "confidence": confidence})

    except Exception as e:
        logging.error(f"❌ Error in predict route: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/result")
def result():
    prediction = request.args.get("prediction")
    confidence = request.args.get("confidence")
    emoji = "😸" if prediction == "cats" else "🐶"

    logging.info(
        f"📊 Result Page - Prediction: {prediction}, Confidence: {confidence}%"
    )

    return render_template(
        "result.html",
        prediction=prediction + " " + emoji,
        confidence=confidence,
    )


if __name__ == "__main__":
    logging.info("Starting the app with Waitress...")
    app.debug = False
    serve(app, host="0.0.0.0", port=5000)
