import numpy
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.neural_network._multilayer_perceptron import MLPClassifier
from sklearn import datasets

iris = datasets.load_iris()
data = iris.data[:, :2]
labels = iris.target

print(f"data : {data}")

markers = ('s','*','^')

colors= ('blue','green','red')

cmap = ListedColormap(colors)

x_min, x_max = data[:,0].min() - 1, data[:,0].max() + 1
y_min, y_max = data[:,1].min() - 1, data[:,1].max() + 1
resolution = 0.01
x, y = numpy.meshgrid(
    numpy.arange(x_min,x_max,resolution),
    numpy.arange(y_min,y_max,resolution)
)


mlp = MLPClassifier(random_state=1,max_iter=1)

mlp.fit(data,labels)

z = mlp.predict(
    numpy.array(
        [x.ravel(),y.ravel()]
    ).T
)

z = z.reshape(x.shape)

plt.pcolormesh(x,y,z,cmap=cmap)
plt.xlim(x.min(),x.max())
plt.ylim(y.min(),y.max())


classes = ["setosa", "versicolor", "verginica"]

for index,cl in enumerate(numpy.unique(labels)):
    plt.scatter(data[labels == cl, 0], data[labels == cl, 1], color=cmap(index), marker=markers[index],s=60,label=classes[index])

plt.xlabel("꽃잎 길이")
plt.ylabel("꽃 받침 길이")

plt.legend(loc='upper left')

plt.show()