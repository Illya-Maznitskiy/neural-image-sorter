# Neural Image Sorter


## Description
Neural Image Sorter is an AI-powered web app that classifies images with deep learning, featuring a simple upload and prediction interface. 🖼️🤖


## Technologies Used
- Python
- Kaggle
- Tensorflow
- Flask


## Features
- Classify images (e.g., cats, dogs).
- Upload and get instant predictions.
- Simple and user-friendly interface.


## Setup
To install the project locally on your computer, execute the following commands in a terminal:
```bash
git clone https://github.com/Illya-Maznitskiy/neural-image-sorter.git
cd neural-image-sorter
python -m venv venv
venv\Scripts\activate (on Windows)
source venv/bin/activate (on macOS)
pip install -r requirements.txt
```


## Install a dataset
Use the following commands to install it:
(Average execution time: ~3 minutes)
```bash
cd backend
python download_dataset.py
```


## Run the local server
Open the terminal and use the following commands:
```bash
cd backend
python app.py
```


# Screenshots:

### Scraping Data
![App Home Page](frontend/screenshots/app_home_page.png)

### Data Analysis 
![App Result Page](frontend/screenshots/app_result_page.png)


## License
This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.
