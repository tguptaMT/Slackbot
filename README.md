# Jarvis: Slack implementation of a multinomial Naïve Bayes classifier

Jarvis is a custom slackbot implementation of a multinomial Naïve Bayes classifier in Python 3. It uses sqlite3 databse to store training examples for classifying natural language. Jarvis's performance appears to be a function of size and composition of the training dataset. When the overall training dataset is relatively small, Jarvis doesn't perform very well. However, the accuracy improves significantly as more training examples were added to the specified classes. Furthermore, the classification prediction also seems to improve after balancing the size of training dataset across classes. Unbalanced classes seem to compound the classification error especially when overall training dataset is small. In such scenario, the classifier defaults to the class with the largest training sample.

Jarvis performance was estimated by data segmentation based cross validation using the train_test_split function. With my training dataset, Jarvis’s classification accuracy for segmented training-test data was around 68%. Accuracy scores were
measured by importing the metrics module from sklearn. A total of 5000 predictions were made after randomly shuffling and splitting training data into training and test subsets. A maximum accuracy of 96.15% and minimum accuracy of 34.61% was obtained in these predictions.

Often, misclassification was observed in cases where novel test strings were used with synonyms or context similar to training examples. For example, while the test string – “I would like to eat” was accurately classified as training category 'FOOD', the string – “I am hungry” was misclassified as another training category - “GREET”. Similarly, Jarvis occasionally failed to accurately classify test strings that contained one of more tokens that could be classified in more than one specified classes.
For instance, “Have a nice day” should be label: 'GREETING' whereas, “It’s a nice day” should be label: WEATHER. Contextual feature extraction and synonym resolution should be able to able to resolve these issues.

Tarun Gupta, University of Vermont, 2016

