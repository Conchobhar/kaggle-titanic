# OLD - remove if unused

################################################################################
# Tensorflow model:
################################################################################

#tf_data_train = data_train[['Age','SibSp']]
tf_data_train = data_train.drop("Survived", axis=1)
tf_data_test  = data_test[['Age','SibSp']]

# Converting to 2 classes to work with softmax
#   (Survived and Perished, logical opposites)
tf_data_train_labels = data_train[['Survived']]
perished = np.logical_not(data_train[['Survived']])
tf_data_train_labels = tf_data_train_labels.assign(perished=perished.astype(int))

# tf_data_test_labels = data_test[['Survived']]
# perished = np.logical_not(data_test[['Survived']])
# tf_data_test_labels = tf_data_test_labels.assign(perished=perished.astype(int))

# Using data from split instead
# tf_data_train = train_x
# perished = np.logical_not(data_train[['Survived']])
# tf_data_train_labels = tf_data_train_labels.assign(perished=perished.astype(int))

# tf_data_train_labels = train_y
# tf_data_test = valid_x
# tf_data_test_labels = valid_y



n_features = len(tf_data_train.columns)
n_classes = tf_data_train_labels.shape[1]
n_hidden = 4


# Reset tensorflow graph (incase I'm rerunning code):
tf.reset_default_graph()

# n_features dimensions for each pixel, None specifies dimensions of any length 
input_x = tf.placeholder(tf.float32, [None, n_features])
input_y = tf.placeholder(tf.int32, [None, n_classes])

x = tf.transpose(input_x)

# 2 output classes, Survived or Not Survived
W1 = tf.get_variable('W1', [n_classes, n_features], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
#W2 = tf.get_variable('W2', [n_classes, n_hidden], initializer = tf.contrib.layers.xavier_initializer(seed = 1))

b1 = tf.get_variable('b1', [n_classes, 1], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
#b2 = tf.get_variable('b2', [n_classes, 1], initializer = tf.contrib.layers.xavier_initializer(seed = 1))

#a1 = tf.nn.relu(tf.matmul(W1, x) + b1)
z1 = tf.matmul(W1, x) + b1

# placeholder for correct answers

# Preperation for cost function (require transpose)
logits = tf.transpose(z1)
labels = input_y

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#squared_error = tf.reduce_mean(tf.squared_difference(y, y_))

# Compute the cross entropy cost
# This function expects labels/logits of shape (batch size by number of classes)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( labels = labels, logits = logits ) )

# Optimization
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Begin session, initalize variables:
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # for _ in range(1000):
    #   batch_xs, batch_ys = tf_data_train.next_batch(100)
    #   sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    sess.run(init)
    num_epochs = 500
    for epoch in range(num_epochs):
        _, cost = sess.run([train_step, cross_entropy], feed_dict={input_x: tf_data_train, input_y: tf_data_train_labels})
        print(cost)

    model_predictions = tf.cast(tf.nn.softmax(logits, dim = 1) > 0.5, tf.int32)
    correct_prediction = tf.equal(model_predictions, labels)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print ("Train Accuracy:", accuracy.eval({input_x: tf_data_train, input_y: tf_data_train_labels}))
    #print ("Test Accuracy:", accuracy.eval({input_x: tf_data_test, input_y: tf_data_train_labels}))
################################################################################
# Results and Evaluation
################################################################################


# Eval on test set, if you split the initial titanic 'train' data into a further train test split
#print ("Test Accuracy:", accuracy.eval({input_x: tf_data_test, input_y: tf_data_test_labels}))

print(sess.run([labels], feed_dict={input_x: tf_data_train, input_y: tf_data_train_labels}))
print(sess.run([predictions], feed_dict={input_x: tf_data_train, input_y: tf_data_train_labels}))
print(sess.run([correct_prediction], feed_dict={input_x: tf_data_train, input_y: tf_data_train_labels}))
print(sess.run([accuracy], feed_dict={input_x: tf_data_train, input_y: tf_data_train_labels}))


mypred = np.array(sess.run(model_predictions, feed_dict={input_x: tf_data_train, input_y: tf_data_train_labels}))
mypredtotal = list(np.sum(mypred, axis=0).astype("int"))
actualtotal = list(np.sum(np.array(tf_data_train_labels),axis=0).astype("int"))
print("Model Predictions:")
"{:8} {:8}".format(*["Survived","Perished"])
"{:8d} {:8d}".format(*mypredtotal)
"{:8d} {:8d}".format(*actualtotal)

def softmax(x):
    return [np.e**x[0]/(np.e**x[0]+np.e**x[1]),np.e**x[1]/(np.e**x[0]+np.e**x[1])]

import matplotlib.pyplot as plt

plt_range = range(100)

plt.scatter(tf_data_train, tf_data_train_labels.Survived)
plt.show()

test = data_train[["Age", "Survived"]].groupby(['Age'], as_index=False).mean().sort_values(by='Age', ascending=False)

plt.scatter(test.Age,test.Survived)


#Notes
# with 2 vars, Age and SibSp, pred on training set:
# cost = 0.6697
# accu = 61.6%
# with all vars from inital data prep:
# cost = 0.46
# accu = 80.0%

# Assume everyone perished: (61.6% prediction success)
baseline = np.sum(np.equal(tf_data_train_labels["Survived"],0))/tf_data_train_labels.shape[0]