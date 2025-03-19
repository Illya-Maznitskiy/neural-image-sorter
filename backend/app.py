from flask import Flask, request, jsonify
from classify import classify_image
from PIL import Image
import io

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    print("GET / - Serving the home page with the upload form.")
    return """
    <h1>Hi, it's Neural Image Sorter!</h1>
    <p>Upload an image to classify (cats or dogs):</p>
    <form action="/predict" method="POST" enctype="multipart/form-data">
        <input type="file" name="image" accept="image/*" required>
        <button type="submit">Upload Image</button>
    </form>
    """


@app.route("/predict", methods=["POST"])
def predict():
    print("POST /predict - Received POST request to classify an image.")

    try:
        # Check if the image part exists in the request
        if "image" not in request.files:
            print("Error: No image part in request.")
            return jsonify({"error": "No image part"}), 400

        # Get the image file from the request
        image_file = request.files["image"]
        print(f"Received image: {image_file.filename}")

        # Open and process the image
        image = Image.open(io.BytesIO(image_file.read()))
        print(
            f"Original image size: {image.size}"
        )  # Debug print for original image size

        image = image.resize((128, 128))
        print(
            f"Resized image size: {image.size}"
        )  # Debug print for resized image size

        # Classify the image using the classify_image function
        prediction = classify_image(image)
        print(f"Prediction: {prediction}")

        # Return the result as a simple HTML page (or JSON if you prefer)
        return f"""
        <h1>Prediction: {prediction}</h1>
        <a href="/">Go back</a>
        """

    except Exception as e:
        print(f"Error during image classification: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(host="0.0.0.0", port=5000)
