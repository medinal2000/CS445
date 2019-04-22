import csv
import numpy
from mnist_classifier import MNISTClassifier

def train(training_file, test_file, model, epochs, learning_rate):
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
            for row in file_reader:
                input_vec, label = get_inputs(row)
                result = model.forward(input_vec)
                
                # update weights if necessary
                if result != label:
                    model.back_prop(learning_rate, input_vec, label)
                    
            train_accuracy = test(training_file, model)[0]
            test_accuracy = test(test_file, model)[0]
            all_accuracies[epoch+1][0] = epoch + 1
            all_accuracies[epoch+1][1] = train_accuracy
            all_accuracies[epoch+1][2] = test_accuracy
            print(train_accuracy, test_accuracy)
            
            if(train_accuracy - all_accuracies[epoch][1]) < 0.005:
                break
            
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


        
perceptron = MNISTClassifier()
learning_rate = 0.0001
epochs = 70

training_file ='mnist_train.csv'
test_file = 'mnist_test.csv'

all_accuracies = train(training_file, test_file, perceptron, epochs, learning_rate)
accuracy_test, confusion_matrix = test(test_file, perceptron)

file = open("all_accuaracies.txt", "w+")
for row in all_accuracies:
    for element in row:
        file.write(str(element))
        file.write(" ")
    file.write("\n")

file.close()
print(confusion_matrix)
