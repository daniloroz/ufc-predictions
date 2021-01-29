from sklearn.linear_model import LogisticRegression

# all parameters not specified are set to their defaults
model = LogisticRegression(solver='liblinear')
model.fit(X_train_clean, y_train)

predictions = model.predict(X_test_clean)

score = model.score(X_test_clean, y_test)
print(score)
