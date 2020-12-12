from neural import *
import torchvision
import torchvision.transforms as transforms

class Classifier(Neural):

    def __init__(self, layers=None, optimizer=None, loss_fn=nn.MSELoss(), device=torch.device('cpu'), classes=[]):
        super().__init__(layers=layers, optimizer=optimizer, loss_fn=loss_fn, device=device)
        self.classes = classes

    def get_most_probable(self, probs, single_entry=False, return_classes=False):
        """
        Returns indices of highest probabilities in classifier tensor.
        Returns list of possible values
        """
        new_list = []
        for poss_inputs in probs:
            max_value = torch.max(poss_inputs)
            most_probable = [ index for index, prob in zip(range(len(poss_inputs)), poss_inputs) if prob == max_value]
            new_list.append(most_probable)
        return new_list if not return_classes else self.get_probable_classes(new_list)

    def get_probable_classes(self, probs):
        """ Returns classes of most probable options."""
        if self.classes is None or len(self.classes) < 1:
            return None
        classes_list = list()
        class_len = len(self.classes)
        for max_value_list in probs:
            new_classes = [ self.classes[index] for index in max_value_list if index >=0 and index < class_len]
            new_classes = list()
            for index in max_value_list:
                if index >= 0 and index < class_len:
                    new_classes.append(self.classes[index])
            classes_list.append(new_classes)
        return classes_list

    def model_accuracy(self, test_loader, single_entry=False):
        """
        Returns accuracy of whole model by count of correct predictions.
        """
        class_len = len(self.classes)
        correct_count, total_count = 0, 0
        with torch.no_grad():
            for in_data, correct in test_loader:
                guess_indices = self.get_most_probable(self.forward(in_data, single_entry))
                for guess, actual in zip(guess_indices, correct):
                    total_count += 1
                    correct_count += 1 if actual in guess else 0
            accuracy = correct_count / total_count
            print(" Model accuracy: ",(100*accuracy),"%")
            return accuracy

    def class_accuracy(self, test_loader, single_entry=False):
        """
        Returns accuracy for each class in model by count of correct predictions.
        """
        class_len = len(self.classes)
        correct_counts = [ 0 for i in range(class_len)]
        total_counts = deepcopy(correct_counts)
        with torch.no_grad():
            for in_data, correct in test_loader:
                guess_indices = self.get_most_probable(self.forward(in_data, single_entry))
                for guess, actual in zip(guess_indices, correct):
                    total_counts[actual] += 1
                    correct_counts[actual] += 1 if actual in guess else 0
            accuracy = [ correct/total for correct, total in zip(correct_counts, total_counts)]
            print(" Accuracy by class:")
            for class_name, acc in zip(self.classes, accuracy):
                print(class_name, " accuracy: ", (100*acc), "%")
            return accuracy

    def imshow(self):
        return