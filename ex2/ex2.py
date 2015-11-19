from collections import OrderedDict

import tensorflow as tf
import matplotlib.pyplot as plt

ITERATIONS = 1500
LEARNING_RATE = 10.0
MAX_SCORE = 100.0

def get_exams():
    with open("ex2/ex2data1.txt", "r") as data_file:
        data = data_file.read()

    return [[float(column) for column in row.split(",")] for row in data.splitlines()]

def graph_exams(exams, theta):
    plt.scatter([first_score / MAX_SCORE for first_score, _, is_admitted in exams if is_admitted == 1.0], 
                [second_score / MAX_SCORE for _, second_score, is_admitted in exams if is_admitted == 1.0], 
                marker='+', c='black')
    
    plt.scatter([first_score / MAX_SCORE for first_score, _, is_admitted in exams if is_admitted == 0.0], 
                [second_score / MAX_SCORE for _, second_score, is_admitted in exams if is_admitted == 0.0], 
                marker='o', c='yellow')

    x = [min(exam[1] for exam in exams) / MAX_SCORE,
         max(exam[1] for exam in exams) / MAX_SCORE]

    y = [(-1.0 / theta[2]) * (theta[1] * x[0] + theta[0]),
         (-1.0 / theta[2]) * (theta[1] * x[1] + theta[0])]

    plt.plot(x, y, "--")

    plt.ylabel("Exam 2 score (in %)")
    plt.xlabel("Exam 1 score (in %)")

    plt.savefig("ex2/exams.png", dpi=300)
    plt.show()

def main():
    # Parse data
    exams = get_exams()

    # Count examples in the dataset
    examples_count = len(exams)

    # Data placeholders for training
    x1 = tf.placeholder(tf.float32, shape=(examples_count,))
    x2 = tf.placeholder(tf.float32, shape=(examples_count,))
    y = tf.placeholder(tf.float32, shape=(examples_count,))

    # Data placeholder for predicting
    score1 = tf.placeholder(tf.float32)
    score2 = tf.placeholder(tf.float32)

    # Initialize fitting parameters
    theta = tf.Variable(tf.zeros([3, 1], tf.float32))

    # Add a column of ones to x1 and x2 and transpose it to a row vector
    input_data = tf.transpose(tf.pack([tf.ones([examples_count], tf.float32), 
                                       tf.div(x1, MAX_SCORE), 
                                       tf.div(x2, MAX_SCORE)]))

    # Hypnosis function
    hypnosis = tf.div(1.0, tf.add(1.0, tf.exp(tf.reduce_sum(tf.mul(tf.transpose(theta), input_data), 1))))

    # Cost function
    cost_function = tf.reduce_mean(tf.sub(tf.mul(tf.neg(y), tf.log(hypnosis)),
                                                 tf.mul(tf.sub(1.0, y), tf.log(tf.sub(1.0, hypnosis)))))

    # Find global minimum 
    # Note: Original exercise says to use fminunc, but TensorFlow doesn't support it yet.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
    train = optimizer.minimize(cost_function)

    # Predict student admittion probability
    predict = tf.div(1.0, tf.add(1.0, tf.exp(tf.reduce_sum(
        tf.mul(tf.transpose(theta), tf.transpose(tf.pack([1.0, tf.div(score1, MAX_SCORE), tf.div(score2, MAX_SCORE)])))
    ))))

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())

        feed_dict = {
            x1: [first_score for first_score, _, _ in exams],
            x2: [second_score for _, second_score, _ in exams],
            y: [is_admitted for _, _, is_admitted in exams]
        }

        print "Cost function before training:", session.run(cost_function, feed_dict=feed_dict)
        
        for i in xrange(ITERATIONS):
            session.run(train, feed_dict=feed_dict)

            if i % 100 == 0:
                print (len("Cost function before training:") - 2) * ' ' + '->', session.run(cost_function, feed_dict=feed_dict)

        # Predict student probability
        print "Admittion probability prediction of a student with scores 45, 85:", session.run(predict, feed_dict={ score1: 45, score2: 85 })

        final_theta = session.run(theta)

    # Graph data
    graph_exams(exams, final_theta)

if __name__ == '__main__':
    main()