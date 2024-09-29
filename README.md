# Facial Recognition System

## Project Overview
This project implements a robust face recognition system using a Siamese neural network architecture. The model utilizes twin networks with shared weights to compute similarity scores for precise face matching, enabling reliable identification and verification.

### Key Features
* **Siamese Neural Network:** Employs twin neural networks that share weights and learn to encode facial features into a low-dimensional space.
* **Loss Function:** The model uses a contrastive loss function to minimize the distance between faces of the same person and maximize the distance between faces of different people.
* **OpenCV Integration:** Leverages OpenCV for facial detection, utilizing Haar cascades to detect faces in various lighting conditions, expressions, and poses.

This project combines the power of deep learning with traditional computer vision techniques for robust face recognition in varied environments.

## Technologies Used
* **Python:** Core programming language.
* **Keras/TensorFlow:** For building and training the Siamese network.
* **OpenCV:** For face detection and preprocessing.
* **NumPy/Pandas:** Data manipulation and handling.
* **Matplotlib:** Visualization of loss curves and training progress.

## Project Structure
```
.
├── data/                   # Directory containing facial images
│   ├── person1/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── person2/
│   │   └── ...
│   └── ...
├── dataset/                # Generated dataset from facial images
├── models/                 # Trained model weights
├── database/               # Database (Create on your own accord)
│   ├── images/                   # Single image for each person in the database
│   │   ├── person1/
│   │   │   ├── person1.jpg
│   │   ├── person2/
│   │   │   ├── person2.jpg
│   │   └── ...
|   └── database.parquet          # Generated embeddings for each person in the database
├── src/
│   ├── Face Recognition.py       # Script for Facial Recignition
│   ├── Model.ipynb               # Jupyter Notebook for model
│   ├── generate_database.py      # Script to generate database embeddings
│   └── generate_dataset.py       # Script to generate dataset from images
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Installation
1. **Clone Repository**
   ```
   git clone https://github.com/Blaze-DSP/Face-Recognition.git
   cd Face-Recognition
   ```
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
The dataset containing facial images is provided as a data.zip file, which is uploaded on Google Drive. To use the dataset:

1. **Download the Dataset**
   Download the data.zip file from the [Google Drive](https://drive.google.com/file/d/1Yic3_htK-vEAGc4KkFsoTUxWqFk7WuZG/view?usp=drive_link) link.
2. **Extract the Dataset**
   After downloading, extract the zip file into the dataset/ directory
3. **Dataset Structure**
   The extracted dataset should be organized into subfolders where each subfolder corresponds to an individual's images:
   ```
   data/
   ├── person1/
   ├── person2/
   └── ...
   ```
   
This dataset is used for training the Siamese network with Triplet Loss for facial recognition.

## Usage
* **Face Detection:** The system uses OpenCV to detect faces in images. Before training, the facial regions are extracted using Haar cascades.
* **Model Inference:** Once trained, the Siamese network can compare two images and output a similarity score to determine if the two faces belong to the same individual.

## Future Enhancements
* Add support for Triplet Loss for improved face verification accuracy.
* Expand the dataset to include more diverse facial images.
* Integrate real-time face recognition using a camera feed.
