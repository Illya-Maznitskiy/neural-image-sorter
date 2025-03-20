from flask import Flask, request, render_template, jsonify
from classify import classify_image
from PIL import Image
import io

app = Flask(
    __name__,
    template_folder="../frontend/templates",
    static_folder="../frontend/static",
)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            print("Error: No image file uploaded.")
            return jsonify({"error": "No image uploaded"}), 400

        image_file = request.files["image"]
        image = Image.open(io.BytesIO(image_file.read()))

        # Ensure RGB mode (handling transparent images like PNGs)
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = image.resize((128, 128))
        prediction, confidence = classify_image(image)

        confidence = round(float(confidence) * 100, 2)

        return jsonify({"prediction": prediction, "confidence": confidence})

    except Exception as e:
        print(f"Error in predict route: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/result")
def result():
    prediction = request.args.get("prediction")
    confidence = request.args.get("confidence")
    if prediction == "cats":
        emoji = "😸"
    else:
        emoji = "🐶"

    return render_template(
        "result.html",
        prediction=prediction + " " + emoji,
        confidence=confidence,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
