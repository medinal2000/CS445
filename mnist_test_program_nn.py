import csv
import numpy
import random
from mnist_nn import MNISTClassifier

def train(training_file, test_file, model, epochs, num_training_ex = 60000,
          learning_rate=0.1, momentum=0.9):
    #record of accuracies for each epoch
    # epoch number, training accuracy, test_accuracy
    all_accuracies = numpy.zeros((epochs+1, 3))
    
    # get initial accuracy
    train_accuracy = test(training_file, model)[0]
    test_accuracy = test(test_file, model)[0]
    all_accuracies[0][0] = 0
    all_accuracies[0][1] = train_accuracy
    all_accuracies[0][2] = test_accuracy
    
    print(train_accuracy, test_accuracy)
    
    with open(training_file, newline='') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        for epoch in range(epochs):
            data = numpy.empty((num_training_ex, 785))
            index = 0
            
            #get all the rows in training file
            for row in file_reader:
                if index >= num_training_ex:
                    break
                data[index,:] = row
                index += 1
            
            # randomize starting point and train
            starting_point = random.randint(0,num_training_ex)
            index = starting_point
            while index < num_training_ex:
                input_vec, label = get_inputs(data[index,:])
                result = model.forward(input_vec)
                if result != label:               # update weights if necessary
                    model.back_prop(input_vec, label, learning_rate, momentum)
                index += 1
            index = 0
            while index < starting_point:
                input_vec, label = get_inputs(data[index,:])
                result = model.forward(input_vec)
                if result != label:               # update weights if necessary
                    model.back_prop(input_vec, label, learning_rate, momentum)
                index += 1
                    
            train_accuracy = test(training_file, model)[0]
            test_accuracy = test(test_file, model)[0]
            all_accuracies[epoch+1,0] = epoch + 1
            all_accuracies[epoch+1,1] = train_accuracy
            all_accuracies[epoch+1,2] = test_accuracy
            print(train_accuracy, test_accuracy)
            
        return all_accuracies
         
            
# gets the input vector            
def get_inputs(row):
    num_elements = 0
    input_vec = numpy.empty(785)
    for element in row:
        if num_elements > 0:
            element = int(element)
            element /= 255  
            input_vec[num_elements] = element
        else:
            label = int(element)
            input_vec[num_elements] = 1
        num_elements += 1

    return input_vec, label
        
        
def test(file_name, model):
    confusion_matrix = numpy.zeros((10, 10), dtype=int)
    correct = 0
    total = 0
    with open(file_name, newline='') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        for row in file_reader:
            input_vec, label = get_inputs(row)
            result = model.forward(input_vec)
            total += 1
            if result == label:
                confusion_matrix[label][label] += 1
                correct += 1
            else:
                confusion_matrix[label][result] += 1
                
    accuracy = correct/total                
    return accuracy, confusion_matrix


training_file ='mnist_train.csv'
test_file = 'mnist_test.csv'
epochs = 50

############################ 20 hidden nodes ##################################
perceptron = MNISTClassifier(20)

all_accuracies = train(training_file, test_file, perceptron, epochs)
accuracy_test, confusion_matrix = test(test_file, perceptron)

file = open("accuaracies_and_conf_mat_20_hidden.txt", "w+")
for row in all_accuracies:
    for element in row:
        file.write(str(element))
        file.write(" ")
    file.write("\n")
    
file.write("\n")

for row in confusion_matrix:
    file.write(str(row))
    file.write("\n")

file.close()

############################ 50 hidden nodes ##################################
perceptron = MNISTClassifier(50)

all_accuracies = train(training_file, test_file, perceptron, epochs)
accuracy_test, confusion_matrix = test(test_file, perceptron)

file = open("accuaracies_and_conf_mat_50_hidden.txt", "w+")
for row in all_accuracies:
    for element in row:
        file.write(str(element))
        file.write(" ")
    file.write("\n")
    
file.write("\n")

for row in confusion_matrix:
    file.write(str(row))
    file.write("\n")
    
file.close()

############################ 100 hidden nodes ##################################
perceptron = MNISTClassifier(100)

all_accuracies = train(training_file, test_file, perceptron, epochs)
accuracy_test, confusion_matrix = test(test_file, perceptron)

file = open("accuaracies_and_conf_mat_100_hidden_0_9_momentum.txt", "w+")
for row in all_accuracies:
    for element in row:
        file.write(str(element))
        file.write(" ")
    file.write("\n")
    
file.write("\n")

for row in confusion_matrix:
    file.write(str(row))
    file.write("\n")

file.close()

############################ 0 momentum ##################################
perceptron = MNISTClassifier(100)

all_accuracies = train(training_file, test_file, perceptron, epochs, momentum=0)
accuracy_test, confusion_matrix = test(test_file, perceptron)

file = open("accuaracies_and_conf_mat_0_momentum.txt", "w+")
for row in all_accuracies:
    for element in row:
        file.write(str(element))
        file.write(" ")
    file.write("\n")
    
file.write("\n")

for row in confusion_matrix:
    file.write(str(row))
    file.write("\n")

file.close()

############################ 0.25 momentum ###############################
perceptron = MNISTClassifier(100)

all_accuracies = train(training_file, test_file, perceptron, epochs, momentum=0.25)
accuracy_test, confusion_matrix = test(test_file, perceptron)

file = open("accuaracies_and_conf_mat_0_25_momentum.txt", "w+")
for row in all_accuracies:
    for element in row:
        file.write(str(element))
        file.write(" ")
    file.write("\n")
    
file.write("\n")

for row in confusion_matrix:
    file.write(str(row))
    file.write("\n")

file.close()

############################ 0.5 momentum ##################################
perceptron = MNISTClassifier(100)

all_accuracies = train(training_file, test_file, perceptron, epochs, momentum=0.5)
accuracy_test, confusion_matrix = test(test_file, perceptron)

file = open("accuaracies_and_conf_mat_0_5_momentum.txt", "w+")
for row in all_accuracies:
    for element in row:
        file.write(str(element))
        file.write(" ")
    file.write("\n")
    
file.write("\n")

for row in confusion_matrix:
    file.write(str(row))
    file.write("\n")

file.close()

############################# 15 000  training examples #########################

perceptron = MNISTClassifier(100)

all_accuracies = train(training_file, test_file, perceptron, epochs, num_training_ex=15000)
accuracy_test, confusion_matrix = test(test_file, perceptron)

file = open("accuaracies_and_conf_mat_15000.txt", "w+")
for row in all_accuracies:
    for element in row:
        file.write(str(element))
        file.write(" ")
    file.write("\n")
    
file.write("\n")

for row in confusion_matrix:
    file.write(str(row))
    file.write("\n")

file.close()

############################# 30 000  training examples #########################

perceptron = MNISTClassifier(100)

all_accuracies = train(training_file, test_file, perceptron, epochs, num_training_ex=30000)
accuracy_test, confusion_matrix = test(test_file, perceptron)

file = open("accuaracies_and_conf_mat_30000.txt", "w+")
for row in all_accuracies:
    for element in row:
        file.write(str(element))
        file.write(" ")
    file.write("\n")
    
file.write("\n")

for row in confusion_matrix:
    file.write(str(row))
    file.write("\n")

file.close()
