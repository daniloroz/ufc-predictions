#Naive Bayes Model

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB

cnb = ComplementNB()
bnb = BernoulliNB()
mnb = MultinomialNB()
gnb = GaussianNB()

#comment out naive bayes you do not want to use.

y_pred = cnb.fit(X_train_clean, y_train).predict(X_test_clean)
y_pred = bnb.fit(X_train_clean, y_train).predict(X_test_clean)
y_pred = mnb.fit(X_train_clean, y_train).predict(X_test_clean)
y_pred = cgnb.fit(X_train_clean, y_train).predict(X_test_clean)

print("Number of mislabeled points out of a total %d points : %d" % (X_test_clean.shape[0], (y_test != y_pred).sum()))