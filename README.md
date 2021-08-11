# Simple Digit Classifier

This was a coursework assignment I was given during my first year at university. It is a naive Bayes classifier that uses Laplace smoothing to identify digits from the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database).

### Run
---
First clone the repository with:
```
git clone https://github.com/cg-2611/simple-digit-classifier.git
```
Next, open the directory created by the `git clone` command:
```
cd simple-digit-classifier
```
Then to run the program with default options (not advised), run:
```
python main.py
```
For a small demonstration, run the program using:
```
python main.py -train-size 1000  -test-size 25 --verbose
```
> Note: the program requires python 3.9  and so the interpreter used to execute the program must support this version at least. The command `python3` might need to be used instead.


### Options
---
> Note: all the options have defaults within the program and so any combination of options, all options or no options can be supplied to the program.

When executing the program, there are some command line options that I have made available to control how the program is run and the output of the program:
- `-train-start`: the index at which the range for the training data will start, i.e. a value of 100 will use the training data from the 100th digit in the dataset, the default value is 0
- `-train-size`: the number of digits that will be used from the training data from the dataset the default value is the maximum possible value (currently 60000) minus train_start
- `-test-start`: the index at which the range for the testing data will start, i.e. a value of 100 will use the testing data from the 100th digit in the dataset, the default value is 0
- `-test-size`: the number of digits that will be used from the testing data from the dataset the default value is the maximum possible value (currently 10000) minus train_start
- `-smoothing`: the value used for Laplace smoothing
- `--verbose`: if used, a flag is set to true and the program will output each prediction and the actual label for each digit in the testing data
- `--visual`: if used, a flag is set to true and the program will show a window with the 28x28 image of the digit along with the actual digit label and the classifiers prediction for each digit in the testing data

> Note: the dataset is very large and if no options are provided, the classifier will be trained on all 60000 training digits adn tested on all 10000 testing digits, so it is recommended to use the options to use a smaller subset of digits for training and testing.
