from collections import OrderedDict

import tensorflow as tf
import matplotlib.pyplot as plt

ITERATIONS = 1500
LEARNING_RATE = 0.01

def get_population_to_profit_dataset():
    with open("ex1/ex1data1.txt", "r") as data_file:
        data = data_file.read()

    return OrderedDict((float(column) for column in row.split(",")) for row in data.splitlines())

def graph_population_to_profit(data, theta):
    intercept, slope = theta

    plt.scatter(data.keys(), data.values(), marker='x', c='red', alpha=0.5)
    plt.ylabel("Profit in $10,000s")
    plt.xlabel("Population of City in 10,000s")

    plt.plot(data.keys(), [slope * i + intercept for i in data.keys()], "--")

    plt.savefig("ex1/population-to-profit.png", dpi=300)
    plt.show()

def main():
    # Parse data
    population_to_profit = get_population_to_profit_dataset()

    # Count examples in the dataset
    examples_count = len(population_to_profit)

    # Data placeholders for training
    x = tf.placeholder(tf.float32, shape=(examples_count,))
    y = tf.placeholder(tf.float32, shape=(examples_count,))

    # Initialize fitting parameters
    theta = tf.Variable(tf.zeros([2, 1], tf.float32))

    # Add a column of ones to x and transpose it to a row vector
    input_data = tf.transpose(tf.pack([tf.ones([examples_count], tf.float32), x]))

    # Linear model (hypnosis function)
    hypnosis = tf.reduce_sum(tf.mul(tf.transpose(theta), input_data), 1)

    # Cost function we want to minimize
    cost_function = tf.div(tf.reduce_sum(tf.pow(tf.sub(hypnosis, y), 2)), 2 * examples_count)

    # Gradient descent
    train = theta.assign_sub(
        tf.reshape(tf.mul(LEARNING_RATE / examples_count, 
            tf.reduce_sum(tf.mul(tf.transpose(tf.reshape(tf.tile(tf.sub(hypnosis, y), [2]), [2, examples_count])), 
               input_data), 0)
        ), [2, 1]))

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        
        feed_dict = { 
            x: population_to_profit.keys(), 
            y: population_to_profit.values() 
        }

        print "Cost function before training:", session.run(cost_function, feed_dict=feed_dict)

        for i in xrange(ITERATIONS):
            final_theta = session.run(train, feed_dict=feed_dict)

            if i % 100 == 0:
                print (len("Cost function before training:") - 2) * ' ' + '->', session.run(cost_function, feed_dict=feed_dict)

    # Graph data
    graph_population_to_profit(population_to_profit, final_theta)
    
if __name__ == '__main__':
    main()