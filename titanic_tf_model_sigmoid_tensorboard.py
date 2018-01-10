################################################################################
# Tensorflow model:
################################################################################
'''
Modified version of the original TF model to utilize some tensorboard
features.
'''


def tf_model(train_x, train_y, test_x, test_y, learning_rate = 0.1, num_epochs = 600, n_hidden = 0):
    
    ################################################################################
    # Construction Phase:
    ################################################################################

    # Reset tensorflow graph:
    tf.reset_default_graph()

    # Setup logs
    from datetime import  datetime
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_titanic_logs"
    logdir = "{}/run-{}/".format(root_logdir, now)

    # Setting up model structure
    _, n_features = train_x.shape
    _, n_classes = train_y.shape

    # n_features dimensions for each pixel, None specifies dimensions of any length 
    input_x = tf.placeholder(tf.float32, [None, n_features], name = "input_x")
    input_y = tf.placeholder(tf.float32, [None, n_classes], name = "input_y")

    x = tf.transpose(input_x)
    
    if n_hidden == 0:
        W1 = tf.get_variable('W1', [n_classes, n_features], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b1 = tf.get_variable('b1', [n_classes, 1], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        z1 = tf.matmul(W1, x) + b1
        logits = tf.transpose(z1)
    else:
        W1 = tf.get_variable('W1', [n_hidden, n_features], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        W2 = tf.get_variable('W2', [n_classes, n_hidden], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b1 = tf.get_variable('b1', [n_hidden, 1], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b2 = tf.get_variable('b2', [n_classes, 1], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        
        z1 = tf.matmul(W1, x) + b1
        z2 = tf.matmul(W2, z1) + b2
        logits = tf.transpose(z2)

    labels = input_y

    # Compute the cross entropy cost
    # This function expects labels/logits of shape (batch size by number of classes)
    sigmoid_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits( labels = labels, logits = logits ) 
    sce_cost = tf.reduce_mean( sigmoid_cross_entropy )

    # Optimization
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(sce_cost)

    # Logging
    # creates node to evaluate cost, then writes it to a TensorBoard-compatible binary log string called a summary
    sce_summary = tf.summary.scalar('Sigmoid cross-entropy', sce_cost)
    # creates a FileWriter that you will use to write summaries to logfiles in the log directory
    file_writer = tf.summary.FileWriter(logdir,   tf.get_default_graph())

    # Saving graph state:
    # saver =   tf.train.Saver()

    ################################################################################
    # Execution Phase: Begin interactive session
    ################################################################################
    # init = tf.global_variables_initializer()
    # sess = tf.Session()
    # sess.run(init)
    # num_epochs = 10
    # _, cost = sess.run([train_step, ce_cost], feed_dict={input_x: tf_data_train, input_y: tf_data_train_labels})

    # for epoch in range(num_epochs):
    #     _, cost = sess.run([train_step, sce_cost], feed_dict={input_x: tf_data_train, input_y: tf_data_train_labels})
    #     print(cost)
    # sess.close()

    ################################################################################
    # Begin enclosed session, initalize variables:
    ################################################################################
    init = tf.global_variables_initializer()
    feed_dict = {input_x: train_x, input_y: train_y}
    with tf.Session() as sess:

        # If restoring from a saved session:
        # saver.restore(sess, "/tmp/my_model_final.ckpt")
        sess.run(init)
        costs_train = []
        costs_test = []
        for epoch in range(num_epochs):
            _, epoch_cost = sess.run([train_step, sce_cost], feed_dict = feed_dict)
            test_cost = sce_cost.eval({input_x: test_x, input_y: test_y})
            print(epoch_cost)
            costs_train.append(epoch_cost)
            costs_test.append(test_cost)

            summary_str = sce_summary.eval(feed_dict = feed_dict)
            step = epoch #* n_batches + batch_index # Make sure to consider batches here if implemented!
            file_writer.add_summary(summary_str, step)

        model_predictions = tf.cast(tf.nn.sigmoid(logits) > 0.5, tf.int32)
        correct_prediction = tf.equal(model_predictions, tf.cast(labels, tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Create checkpoint
        # save_path = saver.save(sess, "/tmp/my_model.ckpt")

        # Plot the cost
        plt.plot(np.squeeze(costs_train), label = "Train cost")
        plt.plot(np.squeeze(costs_test), label = "Test cost")
        plt.legend()
        plt.ylabel('Cost')
        plt.xlabel('Iterations')
        plt.title("Learning rate = " + str(learning_rate))
        plt.show()
        plt.close()

        print ("Train Accuracy:", accuracy.eval({input_x: train_x, input_y: train_y}))
        print ("Test Accuracy:", accuracy.eval({input_x: test_x, input_y: test_y}))

        test_Ypred = model_predictions.eval({input_x: test_x, input_y: test_y})
        file_writer.close()

        return test_Ypred

################################################################################
# Results and Evaluation
################################################################################

# THIS MIGHT NEED MODIFIYED AS DONE IN ORIGINAL FILE
test_Ypred = tf_model(train_x = train_x, train_y = train_y, test_x = valid_x, test_y = valid_y,
    learning_rate = 0.1, num_epochs = 600, n_hidden = 6)


################################################################################
# Plotting
################################################################################

import matplotlib.pyplot as plt

plt_range = range(100)
plt.scatter(tf_data_train, tf_data_train_labels.Survived)
plt.show()
test = data_train[["Age", "Survived"]].groupby(['Age'], as_index=False).mean().sort_values(by='Age', ascending=False)
plt.scatter(test.Age,test.Survived)

# Assume everyone perished: (61.6% prediction success)
baseline = np.sum(np.equal(tf_data_train_labels["Survived"],0))/tf_data_train_labels.shape[0]


################################################################################
# Evaluation on test set and Submission
################################################################################

submission = pd.DataFrame({
        "PassengerId": data_test_passenger_id,
        "Survived": np.ravel(test_Ypred)
    })
# submission.to_csv('./output/tfpy_submission.csv', index=False)