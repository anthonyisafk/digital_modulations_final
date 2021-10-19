'''
Blind modulation classification using a Deep Neural Network and a 
secondary SVM helper for differentiating QPSK and 8PSK modulations.
***********************************************************
The main network is trained on BPSK, QPSK, 8PSK, 16QAM, 64QAM modulations,
for which the C40, C42, C21, C41, C20 cumulants.
***********************************************************
The secondary model is used to better classify QPSK and 8PSK modulations,
for which we calculate the C65, C88 cumulants.
Antoniou Antonios - aantonii@ece.auth.gr
Kaimakamidis Anestis - anestisk@ece.auth.gr
2021 Aristotle University Thessaloniki.
'''

import csv
from PyMoments import kstat
from tensorflow.keras import layers, models
import numpy as np
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
from scipy.stats import kurtosis, skew, moment


'''
input: String name of the csv file to convert
Converts all i in a csv file to j
'''
def convert_i_to_j(name):
  text = open(name, "r")
  text = ''.join([i for i in text]).replace("i", "j")
  x = open(name, "w")
  x.writelines(text)
  x.close()


'''
Used for the main DNN.
returns an array of 4 columns (X_train, Y_train, X_test, Y_test)
X_train and X_test contains a list of five feature array
Y_train and Y_test contains a list of the modulation for each array of features
'''
def calculate_main_cumulants():
  X_train = []
  Y_train = []
  X_test = []
  Y_test = []

  # The array of modulations the set is trained to.
  # The index of each modulation in the array is used in the CSV file.
  mods = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"]

  # Use the name of the training set, as if it's stored in the project directory.
  with open("test.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    samples = [[]]

    # Transfer all the samples to a multidimensional array so we can calculate the cumulants.
    for row in csv_reader:
      for i in range(1, len(row)):
        samples[line_count].append(complex(row[i]))

      # For each row create a list with 5 elements [C40, C42, C21, C41, C20].
      # Distinguish the complex and conjugate samples, turn them to two numpy columns,
      # then stack them to calculate cumulants using PyMoments.
      np_samples = np.array(samples[line_count], dtype=np.dtype(np.complex_)).reshape((len(samples[line_count]), 1))
      conjugates = [num.conjugate() for num in samples[line_count]]
      np_conjugates = np.array(conjugates, dtype=np.dtype(np.complex_)).reshape((len(conjugates), 1))

      cumulant_array = np.hstack((np_samples, np_conjugates))
      cumulant_array = shuffle(cumulant_array, random_state=0) # shuffle data.

      # Make up for the empty arrays the kstat function sometimes returns
      # (probably when dealing with complex numbers)
      if len(cumulant_array) > 0:
        c40 = kstat(cumulant_array, (0, 0, 0, 0))
        c42 = kstat(cumulant_array, (0, 0, 1, 1))
        c21 = kstat(cumulant_array, (0, 1))
        c41 = kstat(cumulant_array, (0, 0, 0, 1))
        c20 = kstat(cumulant_array, (0, 0))
        #c61 = kstat(cumulant_array, (0, 0, 0, 0, 0, 1))
        c65 = kstat(cumulant_array, (0, 1, 1, 1, 1, 1))
        #c60 = kstat(cumulant_array, (0, 0, 0, 0, 0, 0))
        #c66 = kstat(cumulant_array, (1, 1, 1, 1, 1, 1))
        #skewn = skew(samples[line_count])
        print(line_count)

      calculated_cumulants = [c42, c21, c41, c20, c65, c40]
      for j in range(len(calculated_cumulants)):
        calculated_cumulants[j] = np.sqrt(calculated_cumulants[j].real ** 2 + calculated_cumulants[j].imag ** 2)

      # Keep the first 10000 samples of each modulation in the training set.
      # Feed the remaining samples to the training data.
      if line_count % 12000 <= 10000:
        X_train.append(calculated_cumulants)
        Y_train.append(int(complex(row[0]).real - 1))  # recognize row[0] delimiter
      else:
        X_test.append(calculated_cumulants)
        Y_test.append(int(complex(row[0]).real - 1))  # recognize row[0] delimiter

      line_count += 1
      samples.append([])

    data = [X_train, Y_train, X_test, Y_test]

  return data


'''
Trains the main neural network.
'''
def train_main_dnn():
  data = calculate_main_cumulants()

  # DNN: input layer 6 nodes,
  # 2 hidden layers 20 and 60 nodes,
  # output layer 6 nodes.
  model = models.Sequential()
  model.add(layers.Dense(60, input_shape=(6,), activation="relu"))
  model.add(layers.Dense(20, activation="relu"))
  model.add(layers.Dense(5, activation="softmax"))
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model.fit(data[0], data[1], epochs=100, batch_size=64, validation_data=(data[2], data[3]))
  loss, accuracy = model.evaluate(data[2], data[3])

  print(f"loss = {loss}")
  print(f"accuracy = {accuracy}")
  return model


'''
Used for the secondary model.
Calculates the C6,6 and C8,8 cumulants for the QPSK and 8-PSK modulations 
(indices 1 and 2 in the mods array).
'''
def calculate_secondary_cumulants():
  X_train = []
  Y_train = []
  X_test = []
  Y_test = []

  mods = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"] # we're using mods[1] and mods[2].

  with open("test.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    samples = [[]]

    # Transfer all the samples to a multidimensional array so we can calculate the cumulants.
    for row in csv_reader:
      if complex(row[0]).real == 2 or complex(row[0]).real == 3:  # only keep the necessary mods.
        for i in range(1, len(row)):
          samples[line_count].append(complex(row[i]))

      # For each row create a list with 2 elements [C61, C80].
      np_samples = np.array(samples[line_count], dtype=np.dtype(np.complex_)).reshape((len(samples[line_count]), 1))
      conjugates = [num.conjugate() for num in samples[line_count]]
      np_conjugates = np.array(conjugates, dtype=np.dtype(np.complex_)).reshape((len(conjugates), 1))

      cumulant_array = np.hstack((np_samples, np_conjugates))
      cumulant_array = shuffle(cumulant_array, random_state=0)

      if complex(row[0]).real == 2 or complex(row[0]).real == 3:
        if len(cumulant_array) > 0:
          c65 = kstat(cumulant_array, (0, 1, 1, 1, 1, 1))
          c88 = kstat(cumulant_array, (1, 1, 1, 1, 1, 1, 1, 1))
          calculated_cumulants = [c65, c88]
          print(line_count)

        for j in range(len(calculated_cumulants)):
          calculated_cumulants[j] = np.sqrt(calculated_cumulants[j].real ** 2 + calculated_cumulants[j].imag ** 2)

        if line_count % 12000 <= 10000:
          X_train.append(calculated_cumulants)
          Y_train.append(int(complex(row[0]).real - 2))  # recognize row[0] delimiter.
        else:
          X_test.append(calculated_cumulants)
          Y_test.append(int(complex(row[0]).real - 2))  # recognize row[0] delimiter.

        samples.append([])
        line_count += 1

    data = [X_train, Y_train, X_test, Y_test]
  return data


'''
Trains the SVM.
'''
def train_svm():
  mods = ['QPSK', '8PSK']
  data = calculate_secondary_cumulants()
  normal = StandardScaler() # normalize data.
  norm_data = normal.fit_transform(data)
  original_data = normal.inverse_transform(norm_data)

  svc = svm.SVC(
    kernel = 'poly',
    degree = 3,
    probability = True,
    decision_function_shape = 'ovo',
    break_ties = True
  )
  svc.fit(norm_data[0], original_data[1])

  y_predictions = svc.predict(original_data[2])
  print(classification_report(original_data[3], y_predictions, target_names = mods))

  return svc


'''
Input: trained model
Tests neural network and generates confusion matrix and classification report.
'''
def test_main_dnn(model):
  x_test = []
  y_test = []
  mods = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"]
  with open("/Users/anestiskaimakamidis/PycharmProjects/paper_ex3/test1.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    samples = [[]]
    cumulants = [[]]
    for row in csv_reader:
      for i in range(1, len(row)):
        samples[line_count].append(complex(row[i]))

      np_samples = np.array(samples[line_count], dtype=np.dtype(np.complex_)).reshape((len(samples[line_count]), 1))
      conjugates = [num.conjugate() for num in samples[line_count]]
      np_conjugates = np.array(conjugates, dtype=np.dtype(np.complex_)).reshape((len(conjugates), 1))

      cumulant_array = np.hstack((np_samples, np_conjugates))

      if len(cumulant_array) > 0:
        c40 = kstat(cumulant_array, (0, 0, 0, 0))
        c42 = kstat(cumulant_array, (0, 0, 1, 1))
        c21 = kstat(cumulant_array, (0, 1))
        c41 = kstat(cumulant_array, (0, 0, 0, 1))
        c20 = kstat(cumulant_array, (0, 0))
        #c61 = kstat(cumulant_array, (0, 0, 0, 0, 0, 1))
        #c60 = kstat(cumulant_array, (0, 0, 0, 0, 0, 0))
        #c66 = kstat(cumulant_array, (1, 1, 1, 1, 1, 1))
        c65 = kstat(cumulant_array, (0, 1, 1, 1, 1, 1))
        #skewn = skew(samples[line_count])

      calculated_cumulants = [c42, c21, c41, c20, c65, c40]

      for j in range(len(calculated_cumulants)):
        calculated_cumulants[j] = np.sqrt(calculated_cumulants[j].real ** 2 + calculated_cumulants[j].imag ** 2)

      x_test.append(calculated_cumulants)
      y_test.append(int(complex(row[0]).real - 1))
      line_count += 1
      samples.append([])

  # Make a prediction.
  ynew = model.predict_classes(x_test)
  yprob = model.predict_proba(x_test)

  # Show the inputs and predicted outputs.
  correct = 0
  for i in range(len(x_test)):
    if y_test[i] == ynew[i]:
      correct += 1
    print(i, "X=%s, Y=%s, Predicted=%s, Prob=%s" % (x_test[i], mods[y_test[i]], mods[ynew[i]], yprob[i]))

  # Output the precision percentage, confusion matrix and classification report.
  print(correct / len(x_test))
  print("confusion matrix")
  print(confusion_matrix(y_test, ynew))
  print("classification report")
  print(classification_report(y_test, ynew, target_names=mods))


'''
Input: trained model
Tests SVM and generates confusion matrix and classification report.
'''
def test_secondary_model(model):
  x_test = []
  y_test = []
  mods = ["QPSK", "8PSK"]
  with open("test1.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    samples = [[]]
    for row in csv_reader:
      if complex(row[0]).real == 2 or complex(row[0]).real == 3:
        for i in range(1, len(row)):
          samples[line_count].append(complex(row[i]))

      np_samples = np.array(samples[line_count], dtype=np.dtype(np.complex_)).reshape((len(samples[line_count]), 1))
      conjugates = [num.conjugate() for num in samples[line_count]]
      np_conjugates = np.array(conjugates, dtype=np.dtype(np.complex_)).reshape((len(conjugates), 1))

      cumulant_array = np.hstack((np_samples, np_conjugates))
      if complex(row[0]).real == 2 or complex(row[0]).real == 3:
        if len(cumulant_array) > 0:
          c65 = kstat(cumulant_array, (0, 1, 1, 1, 1, 1))
          c88 = kstat(cumulant_array, (1, 1, 1, 1, 1, 1, 1, 1))
          calculated_cumulants = [c65, c88]
          print(line_count)

        for j in range(len(calculated_cumulants)):
          calculated_cumulants[j] = np.sqrt(calculated_cumulants[j].real ** 2 + calculated_cumulants[j].imag ** 2)

        x_test.append(calculated_cumulants)
        y_test.append(int(complex(row[0]).real - 2))
        samples.append([])
        line_count += 1

  data = [x_test, y_test]

  # Normalize data.
  normal = StandardScaler() # normalize data.
  norm_data = normal.fit_transform(data)
  original_data = normal.inverse_transform(norm_data)

  # Make a prediction.
  ynew = model.predict(norm_data[0])
  yprob = model.predict_proba(norm_data[0])

  # Show the inputs and predicted outputs.
  correct = 0
  for i in range(len(data[0])):
    if original_data[1][i] == ynew[i]:
      correct += 1
    print(i + 239, "X=%s, Y=%s, Predicted=%s, Prob=%s" % (original_data[0][i], mods[original_data[1][i]], mods[ynew[i]], yprob[i]))

  print(correct / len(x_test))
  print("confusion matrix")
  print(confusion_matrix(y_test, ynew))
  print("classification report")
  print(classification_report(y_test, ynew, target_names=mods))


'''
Input: both the trained models.
We set a threshold for confidence scores. If the DNN scores below it, it is up to 
both the DNN and the SVM to determine the modulation type.
Tests neural network and generates confusion matrix and classification report.
'''
def test_combination(primary_model, secondary_model):
  x_test = []
  y_test = []
  x_test1 = []
  y_test1 = []
  mods = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM"]
  with open("/Users/anestiskaimakamidis/PycharmProjects/paper_ex3/test1.csv") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    samples = [[]]
    cumulants = [[]]
    for row in csv_reader:
      for i in range(1, len(row)):
        samples[line_count].append(complex(row[i]))

      np_samples = np.array(samples[line_count], dtype=np.dtype(np.complex_)).reshape((len(samples[line_count]), 1))
      conjugates = [num.conjugate() for num in samples[line_count]]
      np_conjugates = np.array(conjugates, dtype=np.dtype(np.complex_)).reshape((len(conjugates), 1))

      cumulant_array = np.hstack((np_samples, np_conjugates))
      if len(cumulant_array) > 0:
        c40 = kstat(cumulant_array, (0, 0, 0, 0))
        c42 = kstat(cumulant_array, (0, 0, 1, 1))
        c21 = kstat(cumulant_array, (0, 1))
        c41 = kstat(cumulant_array, (0, 0, 0, 1))
        c20 = kstat(cumulant_array, (0, 0))
        c65 = kstat(cumulant_array, (0, 1, 1, 1, 1, 1))

      calculated_cumulants = [c42, c21, c41, c20, c65, c40]
      print(line_count)
      for j in range(len(calculated_cumulants)):
        calculated_cumulants[j] = np.sqrt(calculated_cumulants[j].real ** 2 + calculated_cumulants[j].imag ** 2)

      x_test.append(calculated_cumulants)
      y_test.append(int(complex(row[0]).real - 1))

      # Calculate the new cumulants for the secondary network only if the samples
      # are of the necessary modulations: QPSK and 8-PSK.
      if complex(row[0]).real == 2 or complex(row[0]).real == 3:
        if len(cumulant_array) > 0:
          c65 = kstat(cumulant_array, (0, 1, 1, 1, 1, 1))
          c88 = kstat(cumulant_array, (1, 1, 1, 1, 1, 1, 1, 1))
          calculated_cumulants1 = [c65, c88]


        for j in range(len(calculated_cumulants1)):
          calculated_cumulants1[j] = np.sqrt(calculated_cumulants1[j].real ** 2 + calculated_cumulants1[j].imag ** 2)
        x_test1.append(calculated_cumulants1)
        y_test1.append(int(complex(row[0]).real - 1))

      line_count += 1
      samples.append([])

  # Keep the original data and extract the normalized values
  normal = StandardScaler()
  x_test1_norm = normal.fit_transform(x_test1)
  x_test1_orig = normal.inverse_transform(x_test1_norm)
  y_test1_norm = normal.fit_transform(y_test1)
  y_test1_orig = normal.inverse_transform(y_test1_norm)

  # Make predictions for all the samples using the main DNN.
  ynew = primary_model.predict_classes(x_test)
  yprob = primary_model.predict_proba(x_test)

  # Give the cumulants of the QPSK and 8-PSK samples to the SVM
  # and keep its predictions.
  ynew1 = secondary_model.predict(x_test1_norm)
  yprob1 = secondary_model.predict_proba(x_test1_norm)

  # Add one to the modulation index, for the two tables to map
  # the modulations to the same indexes.
  for i in range(len(ynew1)):
    ynew1[i] += 1

  correct = 0

  '''
  Only combine the two predictions if and only if:
  1. The prediction is either QPSK or 8-PSK, AND
  2. The confidence score of the first network is below 0.6. 
  '''
  for i in range(len(ynew)):
    if ynew[i] == 2 or ynew[i] == 1:
      if max(yprob[i]) < 0.6:
        """
        # Naive Bayes.
        qpsk_score = yprob[i][1] * yprob1[i - 240][0]
        epsk_score = yprob[i][2] * yprob1[i - 240][1]

        # Law of total probability.
        qpsk_score = 0.879 * yprob[i][1] + 0.945 * yprob1[i - 240][0]
        epsk_score = 0.883 * yprob[i][2] + 0.779 * yprob1[i - 240][1]
        """

        qpsk_score = 0.4*yprob[i][1] + 0.6*yprob1[i - 240][0]
        epsk_score = 0.65*yprob[i][2] + 0.35*yprob1[i - 240][1]
        if qpsk_score > epsk_score:
          if y_test[i] == 1:
            correct += 1
        else:
          if y_test[i] == 2:
            correct += 1
      else:
        if ynew[i] == y_test[i]:
          correct += 1
    else:
      if ynew[i] == y_test[i]:
        correct += 1
    #print(i, "Y=%s, Predicted=%s, Prob=%s" % (mods[y_test[i]], mods[ynew[i]], yprob[i]))
  print("Accuracy = ", correct / len(x_test))



# MAIN ROUTINE. What is commented had to be run only the first time the program was executed.

# Start with training the main network, or load an existing one.
#main_model = train_main_dnn()
#main_model.save('modulation_classification.model')
#main_model = models.load_model('modulation_classification.model')
#test_main_dnn(main_model)

# Train or load the secondary network.
#secondary_model = train_svm()
#secondary_model.save('modulation_classification_second(c88 c65).model')
#secondary_model = models.load_model('modulation_classification_second(c88 c65).model')
#test_secondary_dnn(secondary_model)


primary_model = models.load_model('modulation_classification.model')
secondary_model = models.load_model('modulation_classification_second(c88 c65).model')
test_combination(primary_model, secondary_model)
