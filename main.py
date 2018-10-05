import numpy as np
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction.text import CountVectorizer


# Global variable, stores the original training set
TRAINING = []

# Global variable, stores the original training label
LABELS = []


def load_data(fake="./clean_fake.txt", real="./clean_real.txt"):
    """
    Loads the data from file 'clean_fake.txt' and 'clean_real.txt',
    preprocesses it using a vectorizer,
    and splits the entire dataset randomly into 70% training,
    15% validation, and 15% test examples.
    """

    # First, we read in the files.
    with open(fake, 'r') as f1:
        fake = f1.readlines()

    with open(real, 'r') as f2:
        real = f2.readlines()

    f1.close()
    f2.close()

    # Concatenate fake_headlines and real_headlines into one list. Note that fake goes first!
    news_headlines = fake + real

    # Preprocesses the data using a vectorizer.
    vectorizer = CountVectorizer()
    data = vectorizer.fit_transform(news_headlines)

    # Get feature's names.
    features = vectorizer.get_feature_names()
    data = data.toarray()

    # Make a corresponding label list, 0 for fake news, 1 for real news.
    labels = [0 for x in range(len(fake))] + [1 for y in range(len(real))]
    labels = np.asarray(labels)

    # Splits the dataset randomly into 70% training set, and 30% remaining to be further split.
    headlines_train, headlines_rem, labels_train, labels_rem, index_1 \
        = split_dataset(data, labels, test_ratio=0.3)

    # Splits the remaining 30% dataset into 50% validation set and 50% test set.
    headlines_vali, headlines_test, labels_vali, labels_test, index_2 \
        = split_dataset(headlines_rem, labels_rem, test_ratio=0.5)

    # Prepare the data set for computing IG.
    compute_IG_setup(news_headlines, labels, index_1, test_ratio=0.3)

    return headlines_train, labels_train, headlines_vali, labels_vali, \
           headlines_test, labels_test, news_headlines, features


def select_model():
    """
    Trains the decision tree classifier by 5 different
    values of max_depth, and 2 different split criteria
    (information gain and Gini coefficient).
    Return the optimal max_depth and split criterion.
    """

    # first, we load the data.
    headlines_train, labels_train, headlines_vali, labels_vali, headlines_test, labels_test, \
        news_headlines, features = load_data()

    opt_depth, opt_criterion = 5, 'gini'
    accuracy = 0
    my_tree = None

    for maxDepth in [5, 15, 25, 50, 100]:
        for splitCriterion in ['gini', 'entropy']:
            curr_tree = training(maxDepth, splitCriterion, headlines_train, labels_train, headlines_vali, labels_vali)
            curr_accuracy = get_accuracy(curr_tree, headlines_vali, labels_vali)
            if curr_accuracy > accuracy:
                accuracy = curr_accuracy
                opt_depth, opt_criterion = maxDepth, splitCriterion
                my_tree = curr_tree

    print("The best hyperparameters are: max_depth =", opt_depth, "split_criterion =", opt_criterion)
    print("The validation accuracy is:", accuracy, "\n")

    # Next, stick with the hyperparameters achieved the highest validation accuracy.
    # Extract and visualize the tree.
    visualizer(my_tree, features)

    # Report the information gain for the topmost split from the previous part, and for several other key words
    print("the information gain for the topmost split from the previous part (with split_key_word = 'the') is:",
          compute_information_gain(TRAINING, LABELS, 'the'), "\n")

    print("the information gain for split of split_key_word = 'trumps' is:",
          compute_information_gain(TRAINING, LABELS, 'trumps'), "\n")

    print("the information gain for split of split_key_word = 'hillary' is:",
          compute_information_gain(TRAINING, LABELS, 'hillary'), "\n")

    print("the information gain for split of split_key_word = 'donald' is:",
          compute_information_gain(TRAINING, LABELS, 'donald'), "\n")

    print("the information gain for split of split_key_word = 'turnbull' is:",
          compute_information_gain(TRAINING, LABELS, 'turnbull'), "\n")


def compute_information_gain(data, labels, key_word,
                             root_fake=None, root_real=None, left_leaf_fake=None, left_leaf_real=None,
                             right_leaf_fake=None, right_leaf_real=None):
    """Computes the information gain of a split on the training data.
    """
    if not (root_fake is None and root_real is None and left_leaf_fake is None
            and left_leaf_real is None and right_leaf_fake is None and right_leaf_real is None):
        return IG(entropy(root_fake, root_real), left_leaf_fake, left_leaf_real, right_leaf_fake, right_leaf_real)

    return split_by_word_and_compute_IG(data, labels, key_word)


# Helper function
def training(maxDepth, splitCriterion, headlines_train, labels_train, headlines_vali, labels_vali):
    """Trains the decision tree classifier using the given values of max_depth split criteria.
    Return the resulting tree model.
    """

    # the decision tree classifier.
    my_tree = DecisionTreeClassifier(max_depth=maxDepth, criterion=splitCriterion)
    my_tree.fit(headlines_train, labels_train)

    # Evaluates the performance of the model on the validation set.
    # Print the resulting accuracy of our DecisionTree model.
    print("Given max_depth:", maxDepth, ", and splitCriterion:", splitCriterion,
          ", the resulting accuracy of our decision tree model is:",
          get_accuracy(my_tree, headlines_vali, labels_vali), "\n")

    return my_tree


def visualizer(tree, features):
    """visualize the given decision tree
    """
    dat = export_graphviz(tree,
                          out_file='./mytree.dot',
                          feature_names=features,
                          class_names=['fake', 'real'],
                          max_depth=3,
                          rounded=True,
                          filled=True)

    # Remove comment to render plot.
    # graph = graphviz.Source(dat)
    # graph.render("news headline")


def split_dataset(dataset, labels, test_ratio):
    """Split the given dataset into training set and test set randomly.
    the split ratio is given by test_ratio.
    """
    shuffled_indices = np.random.permutation(len(dataset))
    dataset = dataset[shuffled_indices]
    labels = labels[shuffled_indices]
    train_set_size = int(len(dataset) * (1 - test_ratio))
    return dataset[:train_set_size], dataset[train_set_size:], labels[:train_set_size], labels[train_set_size:], \
        shuffled_indices


def get_accuracy(tree, data, labels):
    correct = 0  # counter for correct prediction.
    for i in range(len(data)):
        if tree.predict(data[i].reshape(1, -1)) == labels[i]:
            correct = correct + 1

    return correct / len(data)


def entropy(fake_num, real_num):
    """Compute the entropy given the number of fake news and real news in a dataset."""
    total = fake_num + real_num
    p_fake = fake_num / total
    p_real = 1 - p_fake
    if (p_fake == 0) or (p_real == 0):
        return 0
    return -(p_fake * np.log2(p_fake) + p_real * np.log2(p_real))


def IG(root_entropy, left_leaf_fake, left_leaf_real, right_leaf_fake, right_leaf_real):
    """Compute a information gain of a split."""
    total = left_leaf_fake + left_leaf_real + right_leaf_fake + right_leaf_real
    left_total = left_leaf_fake + left_leaf_real
    right_total = right_leaf_fake + right_leaf_real
    left_entropy = entropy(left_leaf_fake, left_leaf_real)
    right_entropy = entropy(right_leaf_fake, right_leaf_real)
    return root_entropy - ((left_total / total) * left_entropy + (right_total / total) * right_entropy)


def split_by_word_and_compute_IG(data, labels, key_word):
    """Split the dataset into two parts,
    one part contains the headline with the key word,
    the other part contains the headline that doesn't have the key word.
    """
    with_key_word, with_key_word_labels = [], []
    without_key_word, without_key_word_labels = [], []

    for i in range(len(data)):
        headline = data[i]
        label = labels[i]
        if key_word in headline.split():
            with_key_word.append(headline)
            with_key_word_labels.append(label)
        else:
            without_key_word.append(headline)
            without_key_word_labels.append(label)

    root_fake_num, root_real_num = 0, 0

    for x in labels:
        if x == 0:
            root_fake_num += 1
        elif x == 1:
            root_real_num += 1

    # without keyword goes left
    left_leaf_fake, left_leaf_real = 0, 0
    for x in without_key_word_labels:
        if x == 0:
            left_leaf_fake += 1
        elif x == 1:
            left_leaf_real += 1

    # wit keyword goes right
    right_leaf_fake, right_leaf_real = 0, 0
    for x in with_key_word_labels:
        if x == 0:
            right_leaf_fake += 1
        elif x == 1:
            right_leaf_real += 1

    return IG(entropy(root_fake_num, root_real_num), left_leaf_fake, left_leaf_real, right_leaf_fake, right_leaf_real)


def compute_IG_setup(news_headlines, labels, shuffled_indices, test_ratio):
    temp1 = (np.asarray(news_headlines))[shuffled_indices]
    temp2 = (np.asarray(labels))[shuffled_indices]
    train_set_size = int(len(news_headlines) * (1 - test_ratio))
    TRAINING[:] = temp1[:train_set_size]
    LABELS[:] = temp2[:train_set_size]


if __name__ == '__main__':
    select_model()
