from __future__ import division

import math
import copy

"""


class tCSV():

    def __init__(self, classifier):
        """

        :param classifier: In our example type of glass is classifier
        """
        self.rows = []
        self.attributes = []
        self.attribute_types = []
        self.classifier = classifier
        self.class_values = []
        self.class_col_index = None


# the node class that will make up the tree
class decisionTreeNode():
    def __init__(self, is_leaf_node, classification, attribute_split_index, attribute_split_value, parent, left_child,
                 right_child, height):
        """

        :param is_leaf_node: Checks if Node if Leaf Node
        :param classification: Value of classifier
        :param attribute_split_index: Column of split value
        :param attribute_split_value: Numeric split value
        :param parent: Parent Node
        :param left_child:
        :param right_child:
        :param height:
        """
        self.classification = None
        self.attribute_split = None
        self.attribute_split_index = None
        self.attribute_split_value = None
        self.parent = parent
        self.left_child = None
        self.right_child = None
        self.height = None
        self.is_leaf_node = True


def FindDecisionTree(dataset, parent_node, classifier):
    """

    :param dataset: Part of Glass Data set
    :param parent_node:
    :param classifier:
    :return:
    """
    node = decisionTreeNode(True, None, None, None, parent_node, None, None, 0)
    if parent_node is None:
        node.height = 0
    else:
        node.height = node.parent.height + 1

    ones = valueOccurrences(dataset.rows)
    k = len(dataset.rows)
    if ones[0] == k:
        node.classification = 'build wind non-float'
        node.is_leaf_node = True
        return node
    elif ones[1] == k:
        node.is_leaf_node = True
        node.classification = 'build wind float'
        return node
    elif ones[2] == k:
        node.is_leaf_node = True
        node.classification = 'vehic wind float'
        return node
    elif ones[3] == k:
        node.is_leaf_node = True
        node.classification = 'headlamps'
        return node
    elif ones[4] == k:
        node.is_leaf_node = True
        node.classification = 'containers'
        return node
    elif ones[5] == k:
        node.is_leaf_node = True
        node.classification = 'tableware'
    else:
        node.is_leaf_node = False

    # The index of the attribute we will split on
    splitting_attribute = None

    # The information gain given by the best attribute
    maximum_info_gain = 0
    minimum_info_gain = 0.01
    split_val = None

    entropy = entropyCal(dataset, classifier)

    # for each column of data
    for attr_index in range(1, len(dataset.attributes)):

        if (dataset.attributes[attr_index] != classifier):
            local_max_gain = 0
            local_split_val = None

            attr_value_list = [example[attr_index] for example in
                               dataset.rows]
            attr_value_list = list(set(attr_value_list))

            if (len(attr_value_list) > 100):
                attr_value_list = sorted(attr_value_list)
                total = len(attr_value_list)
                ten_percentile = int(total / 10)
                new_list = []
                for x in range(1, 10):
                    new_list.append(attr_value_list[x * ten_percentile])
                attr_value_list = new_list

            for val in attr_value_list:

                current_gain = gainRatio(attr_index, dataset, val, entropy)

                if current_gain > local_max_gain:
                    local_max_gain = current_gain
                    local_split_val = val

            if local_max_gain > maximum_info_gain:
                maximum_info_gain = local_max_gain
                split_val = local_split_val
                splitting_attribute = attr_index

    if maximum_info_gain <= minimum_info_gain or node.height > 20:
        node.is_leaf_node = True
        node.classification = classify_leaf(dataset, classifier)
        return node

    node.attribute_split_index = splitting_attribute
    node.attribute_split = dataset.attributes[splitting_attribute]
    node.attribute_split_value = split_val

    left_dataset = tCSV(classifier)
    right_dataset = tCSV(classifier)

    left_dataset.attributes = dataset.attributes
    right_dataset.attributes = dataset.attributes

    left_dataset.attribute_types = dataset.attribute_types
    right_dataset.attribute_types = dataset.attribute_types

    for row in dataset.rows:
        if splitting_attribute is not None and row[splitting_attribute] >= split_val:
            left_dataset.rows.append(row)
        elif splitting_attribute is not None:
            right_dataset.rows.append(row)

    node.left_child = FindDecisionTree(left_dataset, node, classifier)
    node.right_child = FindDecisionTree(right_dataset, node, classifier)

    return node


def uniformity(dataset):
    """

    :param dataset: Glass Dataset
    :return: All the numeric data is converted to float for uniformity.
    """
    a = len(dataset.rows[0])
    for row in dataset.rows:
        for j in range(a):
            if dataset.attribute_types[j] == 'true':
                row[j] = float(row[j])


def classify_leaf(dataset, classifier):
    """

    :param dataset:
    :param classifier:
    :return: Class of a given instance of data Set at an extreme of tree
    """
    ones = valueOccurrences(dataset.rows)
    z = ones.index(max(ones))

    if z == 0:
        return 'build wind non-float'
    elif z == 1:
        return 'build wind float'
    elif z == 2:
        return 'vehic wind float'
    elif z == 3:
        return 'headlamps'
    elif z == 4:
        return 'containers'
    elif z == 5:
        return 'tableware'


def get_classification(instance, node, class_index):
    '''

    :param instance: Given data instance to be classified
    :param node: Starts with root node and traverse the tree
    :param class_index: column of classifier; in this case 10
    :return: Classification of a previously unseen data instance
    '''
    if node.is_leaf_node:
        return node.classification
    else:
        if instance[node.attribute_split_index] >= node.attribute_split_value:
            return get_classification(instance, node.left_child, class_index)
        else:
            return get_classification(instance, node.right_child, class_index)


def entropyCal(dataset, classifier):
    """

    :param dataset:
    :param classifier:
    :return: Entropy(Decision) = – p(Yes).log2p(Yes) – p(No).log2p(No);
     if classifier has only 2 values; Yes and No
    """
    ones = valueOccurrences(dataset.rows)
    entropy = 0
    base = len(dataset.rows)

    for i in ones:
        ent_a = i / base
        if ent_a != 0:
            entropy += ent_a * math.log(ent_a, 2)
    entropy = -entropy

    return entropy


def gainRatio(attr_index, dataset, val, entropy):
    """

    :param attr_index: column of attribute
    :param dataset:
    :param val: value of attribute
    :param entropy: Entropy of parent node
    :return: gain ratio calculated as Entropy(Decision) – ∑ [ p(Decision|Wind).Entropy(Decision|Wind) ]
    """
    classifier = dataset.attributes[attr_index]
    attr_entropy = 0
    total_rows = len(dataset.rows)
    gain_upper_dataset = tCSV(classifier)
    gain_lower_dataset = tCSV(classifier)
    gain_upper_dataset.attributes = dataset.attributes
    gain_lower_dataset.attributes = dataset.attributes
    gain_upper_dataset.attribute_types = dataset.attribute_types
    gain_lower_dataset.attribute_types = dataset.attribute_types

    for rows in dataset.rows:
        if rows[attr_index] >= val:
            gain_upper_dataset.rows.append(rows)
        elif rows[attr_index] < val:
            gain_lower_dataset.rows.append(rows)

    if len(gain_upper_dataset.rows) == 0 or len(gain_lower_dataset.rows) == 0:
        return -1

    attr_entropy += entropyCal(gain_upper_dataset, classifier) * (len(gain_upper_dataset.rows) / total_rows)
    attr_entropy += entropyCal(gain_lower_dataset, classifier) * (len(gain_lower_dataset.rows) / total_rows)

    return entropy - attr_entropy


def valueOccurrences(instances):
    """

    :param instances: Part of whole data Set
    :return: Number of occurrences of each value in classifier in given Data Set
    """
    count = [0, 0, 0, 0, 0, 0]
    class_col_index = 10
    for i in instances:
        if i[class_col_index] == "build wind non-float":
            count[0] += 1
        elif i[class_col_index] == "build wind float":
            count[1] += 1
        elif i[class_col_index] == "vehic wind float":
            count[2] += 1
        elif i[class_col_index] == "headlamps":
            count[3] += 1
        elif i[class_col_index] == "containers":
            count[4] += 1
        elif i[class_col_index] == "tableware":
            count[5] += 1
    return count


def kFoldVal(dataset, classifier, K):
    accuracy = []
    training_set = copy.deepcopy(dataset)
    test_set = copy.deepcopy(dataset)
    training_set.rows,test_set.rows = [], []

    for ki in range(K):
        training_set.rows = [x for i, x in enumerate(dataset.rows) if i % K != ki]
        test_set.rows = [x for i, x in enumerate(dataset.rows) if i % K == ki]

        root = FindDecisionTree(training_set, None, classifier)

        results = []
        for instance in test_set.rows:
            result = get_classification(instance, root, test_set.class_col_index)
            results.append(str(result) == str(instance[-1]))

        acc = float(results.count(True)) / float(len(results))

        accuracy.append(acc)
        del root

    mean_accuracy = math.fsum(accuracy) / K

    return mean_accuracy


if __name__ == "__main__":
    dataset = tCSV("")

    f = open("csv_result-glass.csv")
    file = f.read()
    dataset.rows = [rows.split(',') for rows in file.splitlines()]

    dataset.attributes = dataset.rows.pop(0)
    print("attributes are: ")
    print(dataset.attributes)

    # true implies numeric data
    # false implies nominal data
    dataset.attribute_types = ['true', 'true', 'true', 'true', 'true', 'true', 'true', 'true', 'true', 'true', 'false']

    dataset.classifier = dataset.attributes[-1]

    dataset.class_col_index = 10
    print("classifier is %d" % int(dataset.class_col_index))

    class_set = [example[10] for example in dataset.rows]
    class_set = list(set(class_set))

    dataset.class_values = class_set

    print("classifier values are : ")
    print(class_set)

    # Converting all numeric value to float
    uniformity(dataset)

    # Performing K fold validation on glass data set
    for k in range(5,16):
        accur = kFoldVal(dataset, dataset.classifier, k)
        print("%d fold Mean Accuracy is %f " % (k , accur))
