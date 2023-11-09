# Multilayer Perceptron

Implementation of a basic multilayer perceptron.

Tested on MacOS Big Sur 11.6.6.

Made on October 25 2023 as part of my education in School 21 :)

# Information

This project:
- The program must implemented using the MVC pattern, and also:
  - there are no business code in the view code
  - there are no interface code in the controller and the model
  - controller is thin
- The program provide the ability to form and train neural network models to classify handwritten Latin letters
- The perceptron can:
  - classify images with handwritten letters of the Latin alphabet
  - have **from 2 to 5** hidden layers
  - use a sigmoid activation function for each hidden layer
  - learn on an open dataset (e.g. EMNIST-letters presented in the datasets directory).
  - show accuracy on a test sample *over 70 percent*
  - trained using the backpropagation method
- The perceptron implemented in *two* ways:
  - in matrix form (all layers are represented as weight matrices)
  - in graph form (adjacency matrix)
- The interface of the program provide the ability to:
  - run the experiment on the test sample or on a part of it, given by a floating point number between 0 and 1 (where 0 is the empty sample - the degenerate situation, and 1 is the whole test sample). After the experiment, there should be an average accuracy, precision, recall, f-measure and total time spent on the experiment displayed on the screen
  - load BMP images (image size can be up to 512x512) with Latin letters and classify them
  - draw two-color square images by hand in a separate window
  - start the real-time training process for a user-defined number of epochs with displaying the error control values for each training epoch. Make a report as a graph of the error change calculated on the test sample for each training epoch
  - run the training process using cross-validation for a given number of groups _k_
  - switch perceptron implementation (matrix or graph)
  - switch the number of perceptron hidden layers (from 2 to 5)
  - save to a file and load weights of perceptron from a file

# Usage

After you build programm with **make** command:
1) If you dont want to train network you can load trained weights from file included in project and proceed to demonstration tab to try letter recognition.
2) If you want train, in **datasets** directory present EMNIST-letters. Unzip it, add train and test files in programm interface, select all the settings you want and then hit **train**. After that wait for training to finish you will see overall statistics.

