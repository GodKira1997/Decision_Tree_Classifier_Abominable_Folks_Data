"""
file: HW_06_KALLURWAR_Anurag_Trainer.py
description: This program is a mentor/training program that creates a
decision tree based on the training data and outputs a classifer file to
predict future/unseen data.
language: python3
author: Anurag Kallurwar, ak6491@rit.edu
"""


import warnings
import sys
import os
import math
import numpy as np
import pandas as pd


# CONSTANTS
ROUND = 3
TARGET = 'ClassID'
TARGET2 = 'ClassName'
PREDICTION = 'PredictedClassID'
MIN_LEAF_NODE_SIZE = 23
MAX_CLASS_PERCENTAGE = 0.9
MAX_DEPTH = 7
NUM_VALUES = 20
TEMPLATE = 'template.txt'
CLASSIFEIR_FILE = 'HW_06_Classifier_Kallurwar.py'


# CLASSES
class TreeNode():
    """
    This class represents a node of the decision tree
    """
    # Class Variables
    __slots__ = "attribute", "threshold", "gain_ratio", "class_a_count", \
                "class_b_count", "left", "right"
    attribute: str
    threshold: float
    gain_ratio: float

    # Class Methods
    def __init__(self, attribute: str, threshold: float, gain_ratio: float = 0,
                 left = None, right = None):
        """
        Constructor
        :param attribute: Best attribute used for splitting
        :param threshold: Best threshold for splitting
        :param gain_ratio: Gain Ratio for this split
        :param left: Left child of node or leaf node containing
        integer class_label
        :param right: Right child of node or leaf node containing
        integer class_label
        """
        self.attribute = attribute
        self.threshold = threshold
        self.gain_ratio = gain_ratio
        self.left = left
        self.right = right

    def print_tree(self, tabs = 1):
        """
        Pretty print the decision tree
        :return: None
        """
        print("NODE: '" + self.attribute + "' | THRESHOLD = " + \
                 str(round(self.threshold, ROUND)) + " | GAIN_RATIO = " +
              str(round(self.gain_ratio, ROUND)))
        tabspace = "\t" * tabs
        if isinstance(self.left, TreeNode):
            print(tabspace + "< :", end=" ")
            self.left.print_tree(tabs + 1)
        else:
            print(tabspace + "< :" + str(self.left))
        if isinstance(self.right, TreeNode):
            print(tabspace + ">= :", end=" ")
            self.right.print_tree(tabs + 1)
        else:
            print(tabspace + ">= :" + str(self.right))

    def __str__(self):
        """
        String Format output for the class
        :return: str
        """
        output = "NODE: '" + self.attribute + "' | " + "THRESHOLD = " + \
                 str(round(self.threshold, ROUND)) + " | < [" + str(
            self.left) + "] | >= [" + str(self.right) + "]"
        return output


# FUNCTIONS
def clean_data(df: pd.DataFrame):
    """
    Cleaning the dataframe
    :param df: input dataframe
    :return: cleaned dataframe
    """
    return df.dropna()


def read_file(file_name: str):
    """
    Read the CSV file and return dataframe
    :param thread_index: Index of thread
    :param file_paths: filename
    :return: dataframe
    """
    print("Reading file: " + file_name)
    # Skipping first line containing "HEADER" string
    dataframe = pd.read_csv(file_name, low_memory=False)
    dataframe = clean_data(dataframe)
    return dataframe


def calculate_counts_and_probabilities(input_df: pd.DataFrame, feature_name:
str, threshold: float):
    """
    Calculate counts and probabilities based on the splitting
    :param input_df: Input dataframe
    :param feature_name: feature used for splitting the node
    :param threshold: threshold for splitting the node
    :return: split_counts, split_probabilities, parent_counts, 
    parent_probabilities
    """
    # Calculating counts after splitting the node for left and right children
    split_counts = [0, 0]
    split_left = input_df[input_df[feature_name] < threshold]
    # (class a, class b) left child
    split_counts[0] = (split_left[split_left[TARGET] == 1].shape[0],
                       split_left[split_left[TARGET] != 1].shape[0])
    split_right = input_df[input_df[feature_name] >= threshold]
    # (class a, class b) right child
    split_counts[1] = (split_right[split_right[TARGET] == 1].shape[0],
                       split_right[split_right[TARGET] != 1].shape[0])

    if sum(split_counts[0]) == 0 or sum(split_counts[1]) == 0:
        return None, None, None, None

    # Calculating probabilities after splitting the node
    split_probabilities = []
    for i in range(2):
        probability_a = split_counts[i][0] / sum(split_counts[i])
        probability_b = split_counts[i][1] / sum(split_counts[i])
        split_probabilities.append((probability_a, probability_b))

    # Calculating counts for the node
    class_a_count = sum([x[0] for x in split_counts])
    class_b_count = sum([x[1] for x in split_counts])
    parent_counts = (class_a_count, class_b_count)

    # Calculating probabilities for the node
    probability_a = parent_counts[0] / sum(parent_counts)
    probability_b = parent_counts[1] / sum(parent_counts)
    parent_probabilities = (probability_a, probability_b)

    return split_counts, split_probabilities, parent_counts, \
           parent_probabilities


def calculate_entropy(probabilities: tuple):
    """
    Calculating entropy after splitting
    :param probabilities: probabilities of classes after splitting
    :return: Entropy
    """
    entropy = 0
    # entropy for class a
    if probabilities[0] > 0:
        entropy -= (probabilities[0] * math.log(probabilities[0], 2))
    # entropy for class b
    if probabilities[1] > 0:
        entropy -= (probabilities[1] * math.log(probabilities[1], 2))
    return entropy


def calculate_information_gain_ratio(input_df: pd.DataFrame, feature_name:
str, threshold: float):
    """
    Calculates the Gain Ratio for this feature after splitting
    :param input_df: Input dataframe
    :param feature_name: feature used for splitting the node
    :param threshold: threshold for splitting the node
    :return: GainRATIO
    """
    # Calculating split proababilities and parent probabilities
    split_counts, split_probabilities, parent_counts, parent_probabilities = \
        calculate_counts_and_probabilities(input_df, feature_name, threshold)

    # if counts are None
    if split_counts == None:
        return 0

    # Entropy of left and right children
    entropies = [calculate_entropy(split_probabilities[0]),
                 calculate_entropy(split_probabilities[1])]
    # Entropy of parent node
    entropy_parent = calculate_entropy(parent_probabilities)

    # InformationGAIN
    total = sum(parent_counts)
    information_gain = entropy_parent
    for i in range(2):
        information_gain -= (sum(split_counts[i]) / total) * entropies[i]

    # SplitINFO
    split_info = 0
    for i in range(2):
        if sum(split_counts[i]) != 0:
            split_info -= ((sum(split_counts[i]) / total) * math.log(sum(
                split_counts[i]) / total, 2))

    # GainRATIO
    gain_ratio = information_gain / split_info
    return gain_ratio


def find_classifier_threshold(input_df: pd.DataFrame, feature_name: str):
    """
    Find the best threshold for classification by maximizing the GainRatio.
    :param input_df: Input dataframe
    :param feature_name: feature used for splitting the node
    :return: best threshold, GainRATIO
    """
    # print("Searching the best threshold for these projections...")
    # Threshold range
    min_value = input_df[feature_name].min()
    max_value = input_df[feature_name].max()
    # print(min_value, max_value)
    # num_values = input_df.shape[0]
    # Offset
    delta_value = (max_value - min_value) / NUM_VALUES
    # Initializations
    best_gain_ratio = 0
    best_threshold = 0
    false_positive_population = 0
    false_negative_population = 0
    threshold = min_value - delta_value
    # Searching the best threshold for classification
    while threshold <= (max_value - delta_value):
        gain_ratio = calculate_information_gain_ratio(input_df, feature_name,
                                                      threshold)
        # print(threshold, gain_ratio)
        # Maximizing the GainRATIO and selecting best threshold
        if gain_ratio >= best_gain_ratio:
            best_gain_ratio = gain_ratio
            best_threshold = threshold
        threshold += delta_value
    return best_threshold, best_gain_ratio


def find_best_split_attribute(input_df: pd.DataFrame, remaining_attributes:
list):
    """
    Find the best feature for splitting the node
    :param input_df: Input dataframe
    :param remaining_attributes: Attributes yet to be selected
    :return: TreeNode object with selected feature
    """
    # Initializations
    best_gain_ratio = 0
    best_feature = ''
    best_threshold = 0
    # For every feature in the attributes yet to be selected
    for feature in remaining_attributes:
        threshold, gain_ratio = find_classifier_threshold(input_df, feature)
        # print(feature, threshold, gain_ratio)
        # Maximizing the GainRATIO and selecting best feature
        if gain_ratio >= best_gain_ratio:
            best_gain_ratio = gain_ratio
            best_feature = feature
            best_threshold = threshold
    # print(best_feature, best_threshold, best_gain_ratio)
    # Returning a TreeNode object for the best feature
    return TreeNode(best_feature, best_threshold, best_gain_ratio)


def check_class_percentage(input_df: pd.DataFrame):
    """
    Calculating class percentage
    :param input_df: Input dataframe
    :return: True, if one class is 90% more than other else False
    """
    class_a_count = input_df[input_df[TARGET] == 1].shape[0]
    class_b_count = input_df[input_df[TARGET] != 1].shape[0]
    percentage_a = class_a_count / (class_a_count + class_b_count)
    percentage_b = class_b_count / (class_a_count + class_b_count)
    return (percentage_a >= MAX_CLASS_PERCENTAGE or percentage_b >=
            MAX_CLASS_PERCENTAGE)


def get_majority_class(input_df: pd.DataFrame):
    """
    Get the class with majority data
    :return: Majority ClassLabel
    """
    class_a_count = input_df[input_df[TARGET] == 1].shape[0]
    class_b_count = input_df[input_df[TARGET] != 1].shape[0]
    if class_a_count > class_b_count:
        return 1
    return -1


def split_node(input_df: pd.DataFrame, remaining_attributes: list, depth: int
= 0):
    """
    Split the node by using the best feature and simultaneoulsy create a
    decision tree
    :param input_df: Input Dataframe
    :param remaining_attributes: Attributes yet to be selected for splitting
    :param depth: Depth of decision tree
    :return: TreeNode object of the selected feature
    """
    # if len(remaining_attributes) <= 0:
    #     return None
    # If depth is more than or equal to MAX depth, then return majority class
    if depth >= MAX_DEPTH:
        return get_majority_class(input_df)
    # if only 23 datapoints left in the node, then return majority class
    if input_df.shape[0] <= 23:
        return get_majority_class(input_df)
    # if one class is 90% more than other, then return majority class
    if check_class_percentage(input_df):
        return get_majority_class(input_df)
    # Calculating the best feature for splitting
    best_node = find_best_split_attribute(input_df, remaining_attributes)
    # Updating the remaining attributes
    remaining_attributes_new = [x for x in remaining_attributes if
                                x != best_node.attribute]
    # Splitting the data based on the selected feature
    split_left = input_df[input_df[best_node.attribute] < best_node.threshold]
    split_right = input_df[input_df[best_node.attribute] >= best_node.threshold]
    # Checking Left child node of the split
    best_node.left = split_node(split_left, remaining_attributes_new, depth + 1)
    # Checking right child node of the split
    best_node.right = split_node(split_right, remaining_attributes_new,
                                 depth + 1)
    return best_node


def construct_decision_tree(input_df: pd.DataFrame):
    """
    Construct a decision tree for the input
    :param input_df: Input DataFrame
    :return: TreeNode object representing the root of decision tree
    """
    remaining_attributes = input_df.columns.tolist()
    remaining_attributes.remove(TARGET)
    remaining_attributes.remove(TARGET2)
    decision_tree = split_node(input_df, remaining_attributes)
    return decision_tree


def predict(row: list, decision_tree: TreeNode, attributes: list):
    """
    Predict the class label for the input row/datapoint
    :param row: Single datapoint with features
    :param decision_tree: Decision tree
    :param attributes: Attributes/Feature names
    :return: Predicted class label
    """
    row_dictionary = dict()
    for index in range(len(attributes)):
        row_dictionary[attributes[index]] = row[index]
    tree_node = decision_tree
    class_label = 0
    while isinstance(tree_node, TreeNode):
        predict_node = tree_node
        if row_dictionary[tree_node.attribute] < tree_node.threshold:
            tree_node = tree_node.left
        else:
            tree_node = tree_node.right
    class_label = tree_node
    return class_label


def create_confusion_matrix(input_df: pd.DataFrame, decision_tree: TreeNode):
    """
    Create a confusion matrix for the training data and find the accuracy
    :param input_df: Input Dataframe
    :param decision_tree: Decision tree
    :return: confusion matrix, accuracy
    """
    attributes = input_df.columns.tolist()
    input_df[PREDICTION] = [predict(row, decision_tree, attributes) for
                            index, row in input_df.iterrows()]
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    # Calculating the confusion matrix for the training data
    for index, row in input_df.iterrows():
        actual = row[TARGET]
        predicted = row[PREDICTION]
        if actual == 1 and predicted == 1:
            true_positives += 1
        elif actual == -1 and predicted == -1:
            true_negatives += 1
        elif actual == -1 and predicted == 1:
            false_positives += 1
        elif actual == 1 and predicted == -1:
            false_negatives += 1

    # Confusion matrix
    confusion_matrix = {
        'TP': true_positives,
        'TN': true_negatives,
        'FP': false_positives,
        'FN': false_negatives
    }

    # Calculating Accuracy of the decision tree
    total = true_positives + true_negatives + false_positives + false_negatives
    accuracy = ((true_positives + true_negatives) / total) * 100
    return confusion_matrix, accuracy


def create_decision_tree_classifer(treeNode: TreeNode, tabs = 2):
    """
    Create simple if-else logic for the decision tree
    :param treeNode: node of the decision tree
    :param tabs: number of tabs
    :return: list of str lines
    """
    attribute = treeNode.attribute
    threshold = treeNode.threshold
    tabspace = "\t" * tabs
    result = []
    result += [tabspace + f"if row_dictionary['{attribute}'] < {threshold}:"
               + "\n"]
    if isinstance(treeNode.left, TreeNode):
        result += create_decision_tree_classifer(treeNode.left, tabs + 1)
    else:
        result += [tabspace + '\t' + f"class_label = {treeNode.left}" + "\n"]
    result += [tabspace + "else:" + "\n"]
    if isinstance(treeNode.right, TreeNode):
        result += create_decision_tree_classifer(treeNode.right, tabs + 1)
    else:
        result += [tabspace + "\t" + f"class_label = {treeNode.right}" + "\n"]
    return result


def create_classifer_file(decision_tree: TreeNode):
    """
    Creates a python file for the simple if-else logic classifier based on
    the decision tree
    :param decision_tree: Decision Tree
    :return: None
    """
    with open(TEMPLATE, 'r') as file:
        content = file.readlines()
    modified_content = [line.replace("\\t", "\t") for line in content]
    start_index = 0
    for index in range(len(modified_content)):
        if "#START" in content[index]:
            start_index = index
            break

    output_classifier_lines = create_decision_tree_classifer(decision_tree)
    content = modified_content[:start_index + 1] + output_classifier_lines + \
              modified_content[start_index + 1:]
    # print(content)
    with open(CLASSIFEIR_FILE, 'w') as file:
        file.writelines(content)


def process(abonimable_df: pd.DataFrame):
    """
    This function compute features and creates a one rule classifier for the
    given data by selecting the best feature
    :param abonimable_df: Input data
    :return: None
    """
    # Part 1.
    # Quantize Data
    print("\n\n===== Quantizing Data =====")
    # Quantizing Age to nearest 1 years
    abonimable_df['Age'] = abonimable_df['Age'].round(0)
    # Quantizing Ht to nearest 1 cm
    abonimable_df['Ht'] = abonimable_df['Ht'].round(0)
    # Quantizing anything else to nearest 1/2 value
    columns_to_be_quantized = ['TailLn', 'BangLn', 'HairLn', 'Reach']
    # For every value in the above columns is being rounded
    abonimable_df = abonimable_df.apply(lambda value: (round(value * 2) / 2)
    if value.name in columns_to_be_quantized else value)

    # Part 2
    # Feature Generation
    print("\n\n===== Generating Features =====")
    abonimable_df['Shagginess'] = abonimable_df['HairLn'] - abonimable_df[
        'BangLn']
    abonimable_df['ApeFactor'] = abonimable_df['Reach'] - abonimable_df['Ht']
    print(abonimable_df)

    # Part 3.
    # Decision Tree Construction
    print("\n\n===== Decision Tree Construction =====")
    decision_tree = construct_decision_tree(abonimable_df)
    print("==FIRST SPLIT==")
    print("Best Attribute: " + decision_tree.attribute)
    print("Best Attribute's Threshold: " + str(decision_tree.threshold))
    # print(decision_tree)
    # decision_tree.print_tree()

    # Part 4.
    # Confusion Matrix and accuracy
    print("\n\n===== Training Data Classification =====")
    confusion_matrix, accuracy = create_confusion_matrix(abonimable_df,
                                                         decision_tree)
    print("Confusion Matrix: (target ClassId = +1)")
    print(confusion_matrix)
    print("Accuracy: " + str(round(accuracy, ROUND)) + "%")

    # Classifier File
    print("\n\n===== Classifer File Output =====")
    create_classifer_file(decision_tree)


def main():
    """
    The main function
    :return: None
    """
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if len(sys.argv) < 2:
        print("Missing Argument!")
        print("Usage: HW_06_KALLURWAR_Anurag_Trainer.py <filename.csv>")
        return
    file_name = sys.argv[1].strip()
    if not os.path.isfile(os.getcwd() + "\\" + file_name):
        print("Please put " + file_name + " in the execution folder!")
        return
    abonimable_df = read_file(file_name)
    # print(abonimable_df)
    process(abonimable_df)


if __name__ == '__main__':
    main()  # Calling Main Function
