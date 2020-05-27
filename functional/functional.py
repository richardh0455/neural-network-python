import functools

input_sets = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]

# Each layer is a tuple in the form of (weight_set {2D Array}, biases {1D Array})
layers = [([[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]], [2, 3, 0.5]),
          ([[0.1, -0.14, 0.5], [-0.5, 0.12, -0.33], [-0.44, 0.73, -0.13]], [-1, 2, -0.5])
         ]


def apply_rectified_linear(input):
    return max(0, input)


def calculate_node_value(inputs, activation_function, weights, bias):
    input_product = map(lambda input, weight: input * weight, inputs, weights)
    summed_products = functools.reduce(lambda a, b: a + b, input_product, 0) + bias
    activated_value = activation_function(summed_products)
    return activated_value


def calculate_layer_values(inputs, activation_function, weight_sets, biases):
    mapped_layer_partial = functools.partial(calculate_node_value, inputs, activation_function)
    mapped_layer_complete = map(mapped_layer_partial, weight_sets, biases)
    return list(mapped_layer_complete)


def calculate_layers(layers, inputs):
    next_layer = next(layers, None)
    if next_layer is None:
        return inputs
    else:
        weight_set = next_layer[0]
        biases = next_layer[1]
        next_inputs = calculate_layer_values(inputs, apply_rectified_linear, weight_set, biases)
        return calculate_layers(layers, next_inputs)


def calculate_batch_values(input_sets, layers):
    mapped_batch = map(lambda input_set: calculate_layers(iter(layers), input_set), input_sets)
    print(list(mapped_batch))


def main():
    calculate_batch_values(input_sets, layers)


main()