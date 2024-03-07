from sklearn.neural_network._multilayer_perceptron import MLPClassifier
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
iris = datasets.load_iris()
data = iris.data
labels = iris.target
mlp = MLPClassifier(random_state=1)

# test_size의 비율만큼 ( 0.5 ) 학습용 테스트 데이터, 라벨로 나눕니다.
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size= 0.5, random_state=1)

#표준편차 Scaler 로드
scaler = StandardScaler()

print(data_train)
data_train_std = scaler.transform(data_train)
print(data_train_std)
data_test_std = scaler.transform(data_test)

data_train = data_train_std
data_test = data_test_std

#학습
mlp.fit(data_train,labels_train)

#예측
pred = mlp.predict(data_test)

print(f"잘 못 분류된 예시 {(labels_test != pred).sum()}")

print(f"정확도 : {accuracy_score(labels_test,pred)}")