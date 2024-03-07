from sklearn.neural_network._multilayer_perceptron import MLPClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
data = iris.data
labels = iris.target
mlp = MLPClassifier(random_state=2)

print(f"mlp: {mlp}")
print(f"data : {data}")
#Data 150개의 iris꽃의 특성을 보여줌.
# Numpy구성
# [0] 꽃 받침 길이
# [1] 꽃 받침 너비
# [2] 꽃 잎 길이
# [3] 꽃 잎 너비
print(f"labels : {labels}")

#학습
mlp.fit(data,labels)

#예측
pred = mlp.predict(data)

#계산
diff = labels != pred

#차이
print(f"diff : {diff.sum()}")

print(f"정확도 : {accuracy_score(labels,pred)}")