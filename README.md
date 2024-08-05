# Plant Leaf Image Classification with LeNet 🌿

## Description

This project aims to classify images of plant leaves using a Convolutional Neural Network (CNN) based on the LeNet architecture. The goal is to accurately categorize the leaves into one of three predefined classes.

## Installation

### Prerequisites

Make sure you have the following libraries installed in your Python environment:

- Python 3.x
- OpenCV (`cv2`)
- NumPy (`numpy`)
- PIL (Pillow)
- Matplotlib (`matplotlib`)
- Keras (`keras`)

You can install the required dependencies with pip:

```bash
pip install opencv-python numpy pillow matplotlib keras
```

## Usage

### 1. Data Acquisition

Place the images of plant leaves in a directory structured by classes. Ensure that your images are organized as follows:

```
/path_to_data/
    ├── Class_1/
    │   ├── image_1.jpg
    │   ├── image_2.jpg
    │   └── ...
    ├── Class_2/
    │   ├── image_1.jpg
    │   └── ...
    └── Class_3/
        ├── image_1.jpg
        └── ...
```

### 2. Image Reading and Preprocessing

The script reads the images from the specified directory, resizes them to 100x100 pixels, and normalizes them to prepare the data for training.

```python
[X_train, y_train] = read_images("/path_to_data")
```

### 3. Model Training

The LeNet model is built and trained on the preprocessed images. The primary hyperparameters are set as follows:

- Number of epochs: 1
- Batch size: 64
- Optimizer: Adam

```python
history = model.fit(train_features, train_targets, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE)
```

### 4. Model Saving

The trained model is saved to a file named `lenet_groupe_2.h5` for future use.

```python
model.save("lenet_groupe_2.h5")
```

### 5. Image Visualization

The script includes a function to display images from the dataset.

```python
plot_images(X_train, total_images=2, rows=1, cols=2, fsize=(10, 50), title='Training Dataset')
```

## Author

This project was created by ISSA IBRAHIM Moubarak.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
