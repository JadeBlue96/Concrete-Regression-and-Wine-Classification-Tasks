import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
import pprint
import pandas as pd
from scipy import interp

#machine learning helper packages
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, f1_score, classification_report, confusion_matrix, auc, roc_curve, precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import label_binarize


class DTClassifier(object):
    """
    The decision tree classifier with customizable hyperparameters as attributes.
    
    Attributes
    ----------
    criterion : string (['entropy', 'gini']) 
        The criterion for calculating the Information Gain the defines the tree splits.
    min_samples_per_split : int
        The minimum amount of nodes for each tree layer
    max_depth : int
        Pruning the tree by limiting the maximum depth
    y_col : string
        the name of the prediction column
    
    """
    
    def __init__(self, criterion='entropy', min_samples_per_split=2, max_depth=5,
                 cv_k=10, y_col='Y_WC'):
        """Parameterized constructor for initializing the decision tree object.
        
        Parameters
        ----------
        criterion : string (['entropy', 'gini']) 
            The criterion for calculating the Information Gain the defines the tree splits.
        min_samples_per_split : int
            The minimum amount of nodes for each tree layer
        max_depth : int
            Pruning the tree by limiting the maximum depth
        cv_k : int
            The number of splits for Cross Validation
        y_col : string
            the name of the prediction column
        
        """
        self._criterion = criterion
        self._min_samples_per_split = min_samples_per_split
        self._max_depth = max_depth
        self._cv_k = cv_k
        self._y_col = y_col
    
    def check_purity(self, labels):
        '''
        Checks whether only one class is present in the list of samples.
    
        Args:
            labels(pd.Series): the annotations for the current tree split
            
        Returns:
            True, False
        '''
        unique_labels = labels.unique()
        if len(unique_labels) == 1:
            return True
        return False
    
    def infer_majority(self, labels):
        '''
        Makes a new prediction based on the majority class in the tree split.
    
        Args:
            labels(pd.Series): the annotations for the current tree split
            
        Returns:
            unique_labels(pd.Series): the predicted label classes (with the highest amount of samples)
        '''
        unique_labels = labels.unique()
        unique_values = []
        for i in range(len(unique_labels)):
            unique_values.append(len(labels[labels == unique_labels[i]]))
        return unique_labels[np.argmax(unique_values)]
    
    def get_tree_splits(self, attributes):
        '''
        Creates a dictionary of all possible splits in each data point over each attribute.
        This is done by sorting the values and fitting a line between each two pairs of data points.
    
        Args:
            attributes(pd.DataFrame): the input features from the dataset
            
        Returns:
            potential_splits(dict): a dictionary containing all the possible splits
        '''
        potential_splits = {}
        for col_idx in range(attributes.shape[1]):
            potential_splits[col_idx] = []
            #print(attributes.iloc[:, col_idx])
            unique_attr = np.sort(attributes.iloc[:, col_idx].unique())
            for attr_idx in range(1, len(unique_attr)):
                cur_val = unique_attr[attr_idx]
                prev_val = unique_attr[attr_idx - 1]
                split_val = (cur_val + prev_val) / 2
                potential_splits[col_idx].append(split_val)
        
        return potential_splits
                
    def split_condition(self, data, split_attr, split_val):
        '''
        Splits the input features DataFrame in two, based on an input attribute value. 
    
        Args:
            data(pd.DataFrame): the input features
            split_attr(int): the index of the target attribute column
            split_val(int): the center-point value for the split
            
        Returns:
            data_left(pd.DataFrame): the data points <= the split value
            data_right(pd.DataFrame): the data points > the split value
        '''
        split_attr_data = data.iloc[:, split_attr]
        data_left = data[split_attr_data <= split_val]
        data_right = data[split_attr_data > split_val]
        
        return data_left, data_right
    
    def get_entropy(self, labels):
        '''
        Calculates the entropy for the current split. 
    
        Args:
            labels(pd.Series): the current annotations for the split
            
        Returns:
            entropy(float): the Entropy value
        '''
        unique_labels = labels.unique()
        unique_values = []
        for i in range(len(unique_labels)):
            unique_values.append(len(labels[labels == unique_labels[i]]))

        probs = (unique_values / np.sum(unique_values))
        entropy = np.sum(probs * (-np.log(probs)), axis=0)
        
        return entropy
    
    def get_full_entropy(self, labels_left, labels_right):
        '''
        Calculates the combined entropy from the left and right child nodes of the tree
        to get the total Information Gain for the current depth.
    
        Args:
            labels_left(pd.Series): the annotations <= the split value
            labels_right(pd.Series): the annotations > the split value
            
        Returns:
            entropy(float): the Entropy value
        '''
        n_left = len(labels_left)
        n_right = len(labels_right)
        n_total = n_left + n_right
        prob_labels_left = n_left / n_total
        prob_labels_right = n_right / n_total
        full_entropy = (prob_labels_left * self.get_entropy(labels_left)) + (
            prob_labels_right * self.get_entropy(labels_right))
        
        return full_entropy
    
    def get_gini_coeff(self, labels):
        '''
        Calculates the Gini coefficient for the current data split.
    
        Args:
            labels(pd.Series): the current annotations for the split
            
        Returns:
            g(float): the Gini coefficient value
        '''
        # Mean absolute difference
        mad = np.abs(np.subtract.outer(labels, labels)).mean()
        # Relative mean absolute difference
        rmad = mad/np.mean(labels)
        # Gini coefficient
        g = 0.5 * rmad
        return g
    
    def get_full_gini(self, labels_left, labels_right):
        '''
        Calculates the combined Gini index from the left and right child nodes of the tree
        to get the total Information Gain for the current depth.
    
        Args:
            labels_left(pd.Series): the annotations <= the split value
            labels_right(pd.Series): the annotations > the split value
            
        Returns:
            full_gini(float): the combined Gini coefficient
        '''
        n_left = len(labels_left)
        n_right = len(labels_right)
        n_total = n_left + n_right
        prob_labels_left = n_left / n_total
        prob_labels_right = n_right / n_total

        full_gini = (prob_labels_left * self.get_gini_coeff(labels_left)) + (
            prob_labels_right * self.get_gini_coeff(labels_right))

        return full_gini
    
    def get_best_split(self, attributes, labels):
        '''
        Estimates and returns the best tree split, based on the chosen criterion.
    
        Args:
            attributes(pd.DataFrame): the input features of the current split
            labels(pd.Series): the input annotations of the current split
            criterion(str): ['entropy', 'gini'] - the measure to calculate the Information Gain
        Returns:
            best_split_attr(pd.DataFrame): the best found attribute to split on for the chosen criterion
            best_split_val(pd.Series): the best found split value of the attribute
        '''
        print('Finding best split..')
        best_measure = 10000
        all_splits = self.get_tree_splits(attributes)
        attributes_full = attributes.copy()
        attributes_full['label'] = labels
        for attr_idx in all_splits:
            for attr_val in all_splits[attr_idx]:
                attr_left, attr_right = self.split_condition(attributes_full, attr_idx, attr_val)
                if self._criterion == 'entropy':
                    cur_measure = self.get_full_entropy(attr_left.iloc[:, -1], attr_right.iloc[:, -1])
                else:
                    cur_measure = self.get_full_gini(attr_left.iloc[:, -1], attr_right.iloc[:, -1])

                if cur_measure <= best_measure:
                    best_measure = cur_measure
                    best_split_attr = attr_idx
                    best_split_val = attr_val
                    print('Index {}: Better split found with loss {}'.format(attr_idx, round(cur_measure, 3)))

        print('Best Split Attribute Index: ', best_split_attr)
        print('Best Split Value: ', best_split_val)
        return best_split_attr, best_split_val
    
    def make_decision_tree(self, attributes, labels, counter=0, split_label='root'):
        '''
        Implements the decision tree by recursively searching for the best split attribute and value.
        After the tree is successfully built, it is traversed and saved to a dictionary for printing.
    
        Args:
            attributes(pd.DataFrame): the input features of the dataset
            labels(pd.Series): the input annotations of the dataset
            counter(int): global counter for the recursion level
            split_label(str): the name of the current parent node that is being split
        Returns:
            tree_predictions(list): the list of predicted labels
        '''

        #print('Counter:', counter)
        # base case
        if (self.check_purity(labels)) or (len(attributes) < self._min_samples_per_split) or (counter >= self._max_depth):
            tree_predictions = self.infer_majority(labels)
            #print('Prediction:', tree_predictions)
            return tree_predictions

        counter += 1
        print()
        print('Split position: ', split_label)
        print()
        # helper functions 
        split_attr, split_val = self.get_best_split(attributes, labels)
        # merge dataset and perform split
        attributes_full = attributes.copy()
        attributes_full['label'] = labels
        attr_left, attr_right = self.split_condition(attributes_full, split_attr, split_val)

        # split the dataset into attributes and labels again
        attr_labels_left = attr_left['label']
        attr_left = attr_left.drop('label', axis=1)
        attr_labels_right = attr_right['label']
        attr_right = attr_right.drop('label', axis=1)

        #create subtrees
        node = "{} <= {}".format(attributes.columns[split_attr], split_val)
        sub_tree = {node: []}

        #recursively search through all the decisions
        decision_yes = self.make_decision_tree(attr_left, attr_labels_left, counter, 'left')
        decision_no = self.make_decision_tree(attr_right, attr_labels_right, counter, 'right')

        sub_tree[node].append(decision_yes)
        sub_tree[node].append(decision_no)

        return sub_tree
    
    def print_tree(self, decision_tree):
        '''
        Prints the tree with improved spacing to highlight the splits.
        
        Args:
            decision_tree(dict): the learned decision tree in dictionary format
        Returns:
        
        '''
        pp = pprint.PrettyPrinter()
        pp.pprint(decision_tree)
        
    def predict_sample(self, decision_tree, sample):
        '''
        Predicts the class for a single example by traversing the tree recursively to find the correct decision node.
        
        Args:
            decision_tree(dict): the learned decision tree in dictionary format
            sample(pd.Series): a series of attribute values for prediction
        Returns:
            prediction(str): a value representing the predicted class
        '''
        decisions = list(decision_tree.keys())[0]
        feature_name, comparison_operator, value = decisions.split(" ")

        # determine split based on sample
        if sample[feature_name] <= float(value):
            prediction = decision_tree[decisions][0]
        else:
            prediction = decision_tree[decisions][1]

        # base case
        if not isinstance(prediction, dict):
            return prediction

        # traverse the tree recursively to find the decision
        rec_tree = prediction
        return self.predict_sample(rec_tree, sample)
    
    def evaluate_tree(self, decision_tree, test_attr, test_labels):
        '''
        Computes the classification accuracy by counting the incorrect and correct samples in the prediction list.
    
        Args:
            decision_tree(dict): the learned decision tree in dictionary format
            test_attr(pd.DataFrame): the input test features for prediction
            test_labels(pd.Series): the input test labels for prediction
        Returns:
            
        '''
        predictions = []
        for i in range(len(test_attr)):
            prediction = self.predict_sample(decision_tree, test_attr.iloc[i])
            predictions.append(prediction)
        correct_preds = (predictions == test_labels)
        n_corr = len(correct_preds[correct_preds == True])
        n_incorr = len(correct_preds[correct_preds == False])

        accuracy = round(n_corr / (n_corr + n_incorr), 3)
        print('Correct predictions:', n_corr)
        print('Incorrect predictions:', n_incorr)
        print('Classification accuracy:', accuracy)
        
    def tree_get_predictions(self, decision_tree, test_attr):
        '''
        Returns a list of all the predicted samples from a DataFrame of features.
    
        Args:
            decision_tree(dict): the learned decision tree in dictionary format
            test_attr(pd.DataFrame): the input test features for prediction
        Returns:
            
        '''
        predictions = []
        for i in range(len(test_attr)):
            prediction = self.predict_sample(decision_tree, test_attr.iloc[i])
            predictions.append(prediction)
        return predictions
    
    def plot_roc(self, labels_test, labels_pred, class_names=[1, 2, 3]):
        '''
        Computes and plots the ROC curve for each class label along with the average curves.
    
        Args:
            labels_test(pd.Series): the true labels in the test set
            labels_pred(pd.Series): the predicted labels of the model
            class_names(list): the list of possible annotation names
        Returns:
            roc_auc['micro'](float): the micro average AUC value
            roc_auc['macro'](float): the macro average AUC value
        '''
        # for multiclass cases
        plt.figure(figsize=(8,8))
        labels_test_bin = label_binarize(labels_test, classes=class_names)
        labels_pred_bin = label_binarize(labels_pred, classes=class_names)
        n_classes = len(class_names)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        # Compute ROC curve and ROC area for each class
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(labels_test_bin[:, i], labels_pred_bin[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(labels_test_bin.ravel(), labels_pred_bin.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='maroon', linestyle=':', linewidth=4)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        lw = 2
        colors = cycle(['aqua', 'darkorange', 'olive'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([-0.05, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()
        return roc_auc['micro'], roc_auc['macro']
    
    def plot_cm(self, labels_test, labels_pred_test, class_names=[1, 2, 3]):
        '''
        Computes and plots the Confusion Matrix for each class label.
    
        Args:
            labels_test(pd.Series): the true labels in the test set
            labels_pred_test(pd.Series): the predicted labels of the model
            class_names(list): the list of possible annotation names
        Returns:
        
        '''
        cm = confusion_matrix(labels_test, labels_pred_test)
        print(cm)
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm, cmap=plt.get_cmap('gray'))
        plt.title('Confusion matrix')
        fig.colorbar(cax)
        ax.set_xticklabels([''] + class_names)
        ax.set_yticklabels([''] + class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        
    def cv_train_eval_tree(self, data, class_names=['1', '2', '3']):
        '''
        Evaluates the tree using K-fold Cross Validation by splitting the training set.
        Evaluates each fold separately using all performance metrics.
    
        Args:
            data(pd.DataFrame): the input features
            class_names(list): the list of possible annotation names
        Returns:
        
        '''
        kfold = StratifiedKFold(n_splits=self._cv_k, shuffle=True, random_state=42)
        # enumerate splits
        counter = 0
        roc_auc = []; fpr = []; tpr = []; train_acc = []; val_acc = []; train_precision_recall_f1_score = []; val_precision_recall_f1_score = [];
        for ind_train, ind_val in kfold.split(data, data[self._y_col]):
            counter += 1
            data_train = data.iloc[ind_train]
            data_val = data.iloc[ind_val]
            data_labels_train = data_train[self._y_col]
            data_train = data_train.drop(self._y_col, axis=1)
            data_labels_val = data_val[self._y_col]
            data_val = data_val.drop(self._y_col, axis=1)
            print('CV Split {} ..'.format(counter))
            print()
            decision_tree = self.make_decision_tree(data_train, data_labels_train)
            print('Evaluating model..')
            print()
            labels_pred_train = self.tree_get_predictions(decision_tree, data_train)
            labels_pred_val = self.tree_get_predictions(decision_tree, data_val)

            print('Accuracy score for training data.')
            train_acc_split = accuracy_score(data_labels_train, labels_pred_train)
            train_acc.append(train_acc_split)
            print('Accuracy score for validation data.')
            val_acc_split = accuracy_score(data_labels_val, labels_pred_val)
            val_acc.append(val_acc_split)

            print('Classification report for training data.')
            print(classification_report(data_labels_train, labels_pred_train, target_names=class_names))
            t_precision, t_recall, t_f1score, _ = precision_recall_fscore_support(data_labels_train, labels_pred_train)
            train_precision_recall_f1_score.append((t_precision, t_recall, t_f1score))
            print('Classification report for validation data.')
            print(classification_report(data_labels_val, labels_pred_val, target_names=class_names))
            v_precision, v_recall, v_f1score, _ = precision_recall_fscore_support(data_labels_val, labels_pred_val)
            val_precision_recall_f1_score.append((v_precision, v_recall, v_f1score))

            print('Confusion matrix for validation data.')
            self.plot_cm(data_labels_val, labels_pred_val, class_names)

            print('Plotting ROC Curve for validation data..')
            roc_auc_micro, roc_auc_macro = self.plot_roc(data_labels_val, labels_pred_val)
            roc_auc.append((roc_auc_micro, roc_auc_macro))

        print()
        print('Mean classification scores...')
        print()
        print('Training accuracy:', round(np.mean(train_acc), 3))
        print('Validation accuracy:', round(np.mean(val_acc), 3))
        print('Training precision:', round(np.mean([i[0] for i in train_precision_recall_f1_score]), 3))
        print('Training recall:', round(np.mean([i[1] for i in train_precision_recall_f1_score]), 3))
        print('Training f1-score:', round(np.mean([i[2] for i in train_precision_recall_f1_score]), 3))
        print('Validation precision:', round(np.mean([i[0] for i in val_precision_recall_f1_score]), 3))
        print('Validation recall:', round(np.mean([i[1] for i in val_precision_recall_f1_score]), 3))
        print('Validation f1-score:', round(np.mean([i[2] for i in val_precision_recall_f1_score]), 3))
        print('AUC Mean Micro Average:', round(np.mean([i[0] for i in roc_auc]), 3))
        print('AUC Mean Macro Average:', round(np.mean([i[1] for i in roc_auc]), 3))
		

if __name__ == "__main__":

	wine_data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
	wine_data.columns = ['Y_WC', 'Alcohol', 'Malic_Acid', 'Ash', 'Alcalinity_Of_Ash', 'Magnesium', 'Total_Phenols', 'Flavanoids', 
             'Nonflavanoid_Phenols', 'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280_OD315_Of_Diluted_Wines', 
             'Proline']
	wine_labels = wine_data['Y_WC']
	wine_features = wine_data.drop('Y_WC', axis=1)
	wine_train = wine_features.sample(frac=0.7, random_state=42) #random state is a seed value
	wine_labels_train = wine_labels[wine_train.index]
	wine_test = wine_features.drop(wine_train.index)
	wine_labels_test = wine_labels[wine_test.index]
	
	decision_tree = DTClassifier()
	print(decision_tree.check_purity(wine_labels_train[6:12]))
	print(decision_tree.infer_majority(wine_labels_train[6:12]))
	print(decision_tree.get_tree_splits(wine_train))
	
	split_val = 12
	split_attr = 1
	wine_left, wine_right = decision_tree.split_condition(wine_data, split_attr, split_val)
	
	print(decision_tree.get_entropy(wine_right.iloc[:, 0]))
	print(decision_tree.get_full_entropy(wine_left.iloc[:, 0], wine_right.iloc[:, 0]))
	print(decision_tree.get_gini_coeff(wine_right.iloc[:, 0]))
	best_split_attr_ent, best_split_val_ent = decision_tree.get_best_split(wine_train, wine_labels_train)
	tree = decision_tree.make_decision_tree(wine_train, wine_labels_train)
	decision_tree.print_tree(tree)
	
	sample = wine_test.iloc[25]
	print(sample)
	print('Sample true label:', wine_labels_test.iloc[25])
	decision_tree.predict_sample(tree, sample)
	decision_tree.evaluate_tree(tree, wine_train, wine_labels_train)
	decision_tree.cv_train_eval_tree(wine_data)

