from object_oriented.Layer_Dense import Layer_Dense

#Input Values
X = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]

def main():
    layer1 = Layer_Dense(4,5)
    layer2 = Layer_Dense(5,2)

    layer1.forward(X)
    layer2.forward(layer1.output)
    print(layer2.output)



