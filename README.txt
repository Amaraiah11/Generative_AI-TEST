100-image handwritten digit dataset
File: digits_100.csv
Rows: 100 labeled images covering digits 0-9.
Each row contains a label and 64 pixel values for an 8x8 grayscale image.
Pixel values range from 0 to 16. This is the scikit-learn Digits dataset, not MNIST.
For your app, use a train/test split (for example 80/20). For an assignment requiring MNIST, use torchvision.datasets.MNIST instead.
