import matplotlib.pyplot as plt
import numpy as np
import sys

from classifier import Classifier
from keras.datasets import mnist


BRIGHTNESS_THRESHOLD = 50 # determined through trial and error

TRAIN_MAX = 60000
TEST_MAX = 10000


def extract_features(data: np.ndarray) -> np.ndarray:
    """
    Takes the pixel data from the 28x28 image from the MNIST dataset and
    flattens them into a 784x1 array of binary features with a
    1 representing a pixel with a brightness of more than 50.

    :param data: the data from which the features are to be extracted
    :return: an array of the extracted binary features for each
    entity in data
    """
    # # compute the binary features for each pixel the brightness threshold (not inverted this time)
    extracted_features = (data > BRIGHTNESS_THRESHOLD).astype(int)

    # # flatten the pixel features from 28 x 28 for each digit to 784 x 1
    extracted_features = np.reshape(extracted_features, (extracted_features.shape[0], -1))

    return extracted_features

def get_args() -> tuple:
    """
    Parse the command line arguments and search for any specified options.

    :return: train_start, train_end, test_start, test_end, smoothing, verbose, visual
             train_start: the index at which the training data will start at
             or 0 if not specified
             train_size: the number of entities from train_start that will
             be used for training or the maximum if not specified
             test_start: the index at which the testing data will start at
             or 0 if not specified
             test_size: the number of entities from test_start that will
             be used for testing or the maximum if not specified
             smoothing: the value of alpha to be passed to the Classifier class
             constructor for Laplace smoothing
             verbose: true if option is specified, false otherwise
             visual: true if option is specified, false otherwise
    """
    train_start_index = sys.argv.index("-train-start") if "-train-start" in sys.argv else -1
    train_start = int(sys.argv[train_start_index + 1]) if train_start_index != -1 else 0

    if train_start < 0 or train_start > TRAIN_MAX:
        raise ValueError(f"train-start must be between 0 and {TRAIN_MAX} (inclusive)")

    train_size_index = sys.argv.index("-train-size") if "-train-size" in sys.argv else -1
    train_size = int(sys.argv[train_size_index + 1]) if train_size_index != -1 else TRAIN_MAX - train_start

    if (train_size is not None) and (train_size <= 0 or train_size + train_start > TRAIN_MAX):
        raise ValueError(f"train-size must be between 1 and {TRAIN_MAX - train_start} (inclusive) for train-start value {train_start}")

    test_start_index = sys.argv.index("-test-start") if "-test-start" in sys.argv else -1
    test_start = int(sys.argv[test_start_index + 1]) if test_start_index != -1 else 0

    if test_start < 0 or test_start > TEST_MAX:
        raise ValueError(f"test-start must be between 0 and {TEST_MAX} (inclusive)")

    test_size_index = sys.argv.index("-test-size") if "-test-size" in sys.argv else -1
    test_size = int(sys.argv[test_size_index + 1]) if test_size_index != -1 else TEST_MAX - test_start

    if (test_size is not None) and (test_size <= 0 or test_size + test_start > TEST_MAX):
        raise ValueError(f"test-size must be between 1 and {TEST_MAX - test_start} (inclusive) for test-start value {test_start}")

    smoothing_index = sys.argv.index("-smoothing") if "-smoothing" in sys.argv else -1
    smoothing = float(sys.argv[smoothing_index + 1]) if smoothing_index != -1 else 1.0

    if smoothing < 0:
        raise ValueError("value for smoothing must be greater than 0")

    visual = True if "--visual" in sys.argv else False

    verbose = True if "--verbose" in sys.argv else False

    return train_start, train_size, test_start, test_size, smoothing, verbose, visual

def get_data() -> tuple:
    """
    Get the MNIST dataset training and testing data arrays and use the command
    line arguments to produce the sub arrays to be returned.

    :return: train_pixels, train_labels, test_pixels, test_labels
             train_pixels: the 28x28 pixel array of each entity in the
             specified range from the training data portion of the MNIST dataset
             train_labels: the label for each array of pixels in the
             specified range from the training data portion of the MNIST dataset
             test_pixels: the 28x28 pixel array of each entity in the
             specified range from the testing data portion of the MNIST dataset
             test_labels: the label for each array of pixels in the
             specified range from the testing data portion of the MNIST dataset
    """
    train_start, train_size, test_start, test_size, _, _, _ = get_args()

    (train_pixels, train_labels), (test_pixels, test_labels) = mnist.load_data()

    train_end = train_start + train_size if train_size is not None else TRAIN_MAX
    test_end = test_start + test_size if test_size is not None else TEST_MAX

    train_pixels = train_pixels[train_start: train_end]
    train_labels = train_labels[train_start: train_end]

    test_pixels = test_pixels[test_start: test_end]
    test_labels = test_labels[test_start: test_end]

    return train_pixels, train_labels, test_pixels, test_labels

def main():
    _, _, _, _, smoothing, verbose, visual = get_args()

    training_data_pixels, training_data_labels, testing_data_pixels, testing_data_labels = get_data()

    # digits are from 0-9 so use 10 classes and use value of variable "smoothing" for Laplace smoothing
    digit_classifier = Classifier(10, alpha=smoothing)

    training_data = extract_features(training_data_pixels)
    digit_classifier.train(training_data, training_data_labels)

    testing_data = extract_features(testing_data_pixels)
    predictions = digit_classifier.predict(testing_data)

    accuracy = np.count_nonzero(predictions == testing_data_labels) / testing_data_labels.shape[0]
    print(f"Accuracy on test data using new features is: {accuracy}")

    # if the verbose option is set then print the actual label and the classifiers prediction for each prediction
    if verbose:
        print("Actual:\t\tPredicted:")
        for i, prediction in enumerate(predictions):
            print(f"{testing_data_labels[i]}\t\t{prediction}")

    # if the visual option is set then show the plot of each digit along with the actual label
    # and the classifiers prediction for each prediction
    if visual:
        for i, prediction in enumerate(predictions):
            plt.imshow(testing_data_pixels[i], cmap=plt.get_cmap("gray"))
            plt.axis("off")
            plt.title(f"Actual: {testing_data_labels[i]} Predicted: {prediction}")
            plt.show()


if __name__ == "__main__":
    main()
