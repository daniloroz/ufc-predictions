#Naive Bayes Model

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB

cnb = ComplementNB()
bnb = BernoulliNB()
mnb = MultinomialNB()
gnb = GaussianNB()

#comment out the function of naive bayes you do not want to use.

y_pred = cnb.fit(X_train_clean, y_train).predict(X_test_clean)
y_pred = bnb.fit(X_train_clean, y_train).predict(X_test_clean)
y_pred = mnb.fit(X_train_clean, y_train).predict(X_test_clean)
y_pred = cgnb.fit(X_train_clean, y_train).predict(X_test_clean)

print("Number of mislabeled points out of a total %d points : %d" % (X_test_clean.shape[0], (y_test != y_pred).sum()))

cm = metrics.confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);
