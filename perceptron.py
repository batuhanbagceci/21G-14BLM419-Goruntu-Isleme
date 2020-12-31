#!/usr/bin/env python
# coding: utf-8

import numpy as np

class Perceptron(object):
    def __init__(self, input_size, lr=1, epochs=10):
        self.W = np.zeros(input_size+1)
        self.epochs = epochs
        self.lr = lr
        
    def activation_fn(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, x):
        x = np.insert(x, 0, 1)
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a
    
    def fit(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                y = self.predict(X[i])
                e = d[i] - y
                self.W = self.W + self.lr * e * np.insert(X[i], 0, 1)


X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])


d = np.array([0, 0, 0, 1])


print(X)


print(d)


perceptron = Perceptron(input_size=2)


print(perceptron.W)


perceptron.fit(X, d)


print(perceptron.W)


perceptron.predict(np.asarray([1,1]))


mp = Perceptron(5)
x = np.asarray([-10,-2,-30,4,-50])
print(mp.predict(x))


print(mp.W)


mp.activation_fn(10)


# Soru 1:
# __init__ fonksiyonu perceptronu oluşturan constructor fonksiyonudur.
# activation_fn fonksiyonu matrislerin nokta çarpımının sonucuna basamak fonksiyonu uygulamak için kullanılıyor.
# predict fonksiyonu girdi olarak verilen dizi için tahmin sonucu üretiyor.
# fit fonksiyonu perceptronun supervised olarak eğitilmesini sağlıyor.


# Soru 2:
#X = np.array([
#        [0, 0],
#        [0, 1],
#        [1, 0],
#        [1, 1]
#    ])

# d = np.array([0, 1, 1, 0])

# perceptron = Perceptron(input_size=2)
# perceptron.fit(X, d)

# perceptron.W
# Output: array([ 0., -1.,  0.])

# perceptron.predict(np.asarray([0,0]))
# Output: 1

# perceptron.predict(np.asarray([0,1]))
# Output: 1

# perceptron.predict(np.asarray([1,0]))
# Output: 0

# perceptron.predict(np.asarray([1,1]))
# Output: 0

# Soru 3:
# X değerleri imzaların yükseklik*genişlik*3 olan bir sütun vektör, d değerleri imzaların karşılıkları olurdu.
