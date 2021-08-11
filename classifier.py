import numpy as np


class Classifier:
    """
    Classifier is a naive Bayes classifier that uses Laplace smoothing.
    """

    def __init__(self, classes: int, alpha: float=1.0) -> None:
        """
        Constructor for Classifier, creates a new instance of a Classifier class.

        :param classes: the number of classes that data will be classified into
        :param alpha: the value used for Laplace smoothing, default is 1.0
        """
        self.classes = classes
        self.alpha = alpha

        # initialise the lists that will hold the required log estimates for the classifier
        self.__log_class_priors = []
        self.__log_class_conditional_likelihoods = []

    def __estimate_log_class_priors(self, labels: np.ndarray) -> None:
        for i in range(self.classes):
            # for each class, count the number of occurrences of the class in the training labels
            # and calculate the proportion of this class in the training data
            class_proportion = np.count_nonzero(labels == i) / len(labels)

            # calculate the log of this proportion and append it to the list of priors
            if class_proportion != 0:
                self.__log_class_priors.append(np.log(class_proportion))
            else:
                self.__log_class_priors.append(0)

        self.__log_class_priors = np.array(self.__log_class_priors)

    def __estimate_log_class_conditional_likelihoods(self, features: np.ndarray, labels: np.ndarray) -> None:
        total_features = len(features.T) # count the number of features that the data has

        # count the number of features that occur in each instance of each class
        class_feature_counts = [np.count_nonzero(np.logical_and(features.T > 0, labels == i)) for i in range(self.classes)]

        for i in range(self.classes):
            class_theta = [] # list of likelihoods for the class

            # loop through the transposed matrix of features
            for feature in features.T:
                # count the number of times a feature appears in each class
                feature_count = np.count_nonzero(np.logical_and(feature > 0, labels == i))

                # calculate the probability of each feature occurring in this class using laplace smoothing
                theta = (feature_count + self.alpha) / (class_feature_counts[i] + (self.alpha * total_features))

                class_theta.append(np.log(theta))

            # for each class, append the list of likelihoods for each feature
            self.__log_class_conditional_likelihoods.append(class_theta)

        self.__log_class_conditional_likelihoods = np.array(self.__log_class_conditional_likelihoods)

    def train(self, training_data: np.ndarray, training_labels: np.ndarray) -> None:
        """
        Trains the classifier to generate the conditional likelihoods for each
        class from the provided data.

        :param training_data: the binary features of each training data  entity
        :param training_labels: the actual class for each training data entity
        """
        # before training, make sure the prior list and likelihoods list is empty
        self.__log_class_priors = []
        self.__log_class_conditional_likelihoods = []

        if not (training_data is None or training_labels is None):
            self.__estimate_log_class_priors(training_labels)
            self.__estimate_log_class_conditional_likelihoods(training_data, training_labels)

    def predict(self, testing_data: np.ndarray) -> np.ndarray:
        """
        Classifies the data entities from the testing data using the conditional
        likelihoods estimated from training.

        :param testing_data: the binary features of each testing data entity
        :return: an array of the predictions for the classes for each set of
        features in testing data
        """
        class_predictions = []

        for features in testing_data:
            # create a list for storing the predictions for each class,
            # with the initial values being the priors for each class
            posteriori = [self.__log_class_priors[i] for i in range(self.classes)]

            for i in range(self.classes):
                for j in range(len(features)):
                    # for each class, go through and add to the prior estimate, the estimate of the likelihoods
                    # of each feature appearing in an instance of the class, multiplied by the feature itself
                    posteriori[i] += features[j] * self.__log_class_conditional_likelihoods[i][j]

            # append to the final list of predictions, the class that maximised the final probability
            class_predictions.append(np.argmax(posteriori))

        return np.array(class_predictions)
