import pickle

def load_classifier():
  classifier, tfv = pickle.load(open('classifier.p', 'rb'))

  def predict(X):
    return [p[1] for p in classifier.predict_proba(tfv.transform(X))]

  return predict

predict = load_classifier()

X = ['you will never believe what happens next', 'Obama declares war']

print(predict(X))
