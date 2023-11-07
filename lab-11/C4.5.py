import pandas as pd
import numpy as np
import math
data = pd.read_csv("data_dt.csv")
def entropy(target_column):
    elements, counts = np.unique(target_column, return_counts=True)
    entropy = 0
    for count in counts:
        probability = count / len(target_column)
        entropy -= probability * math.log2(probability)
    return entropy

def information_gain(data, feature, target):
    total_entropy = entropy(data[target])
    values, counts = np.unique(data[feature], return_counts=True)
    weighted_entropy = 0
    for value, count in zip(values, counts):
        subset = data[data[feature] == value]
        subset_entropy = entropy(subset[target])
        weighted_entropy += (count / len(data)) * subset_entropy
    return total_entropy - weighted_entropy

def gain_ratio(data, feature, target):
    info_gain = information_gain(data, feature, target)
    values, counts = np.unique(data[feature], return_counts=True)
    split_info = 0
    for count in counts:
        probability = count / len(data)
        split_info -= probability * math.log2(probability)
    return info_gain / split_info

def best_split(data, features, target):
    gain_ratios = []
    for feature in features:
        gain_ratios.append(gain_ratio(data, feature, target))
    return features[np.argmax(gain_ratios)]

def c45(data, original_data, features, target_attribute_name, parent_node_class=None):
    # If all target values are the same, return this value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

    # If dataset is empty, return the mode target feature value in the original dataset
    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name],
                                                                                   return_counts=True)[1])]

    # If there are no features left to split the data, return the mode target feature value of the current node
    elif len(features) == 0:
        return parent_node_class

    # Otherwise, grow the tree
    parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], 
                                                                                   return_counts=True)[1])]

    best_feature = best_split(data, features, target_attribute_name)
    tree = {best_feature: {}}

    features = [i for i in features if i != best_feature]

    for value in np.unique(data[best_feature]):
        value = value
        sub_data = data.where(data[best_feature] == value).dropna()
        subtree = c45(sub_data, data, features, target_attribute_name, parent_node_class)
        tree[best_feature][value] = subtree

    return tree

# Function to classify a new sample
def classify(sample, tree, default=None):
    attribute = next(iter(tree))

    if sample[attribute] in tree[attribute].keys():
        result = tree[attribute][sample[attribute]]
        if isinstance(result, dict):
            return classify(sample, result)
        else:
            return result
    else:
        return default
def display_tree(tree, depth=0, prefix="Root: "):
    if isinstance(tree, dict):
        for key, value in tree.items():
            print("  " * depth + prefix + key)
            if isinstance(value, dict):
                display_tree(value, depth + 1, "Feature: ")
            else:
                print("  " * (depth + 1) + "Class: " + value)
    else:
        print("  " * depth + "Class: " + tree)

# Build the C4.5 decision tree
target_attribute_name = "Decision"
features = data.columns.difference([target_attribute_name])
tree = c45(data, data, features, target_attribute_name)

# Classify a new sample
new_sample = {"Outlook": "Sunny", "Temperature": "Low", "Humidity": "Medium", "Wind": "Weak"}
display_tree(tree)
result = classify(new_sample, tree, "No")

print(f"Prediction for the new sample: {result}")