
def tf_model_btb(train_x, train_y, test_x, test_y = None, learning_rate = 0.1, n_epochs = 600):
    '''
    TF model for kaggle titanic dataset, using a DNN "ByTheBook" as outlined in
        O'Reily - Hand on with machine learning ...

    With test_y = None, model will not try to evaluate the cost on the test set,
    and only return the predictions for the test set.
    
    '''
    ################################################################################
    # Construction Phase:
    ################################################################################
    import tensorflow as tf
    from tensorflow.contrib.layers import batch_norm

    tf.reset_default_graph()
    
    saver_filename = "./by_the_book.ckpt"

    _, n_inputs = train_x.shape
    _, n_outputs = train_y.shape
    n_hidden1 = 25
    n_hidden2 = 10
    n_hidden3 = 10

    # For batch normalization:
    # Tells the batch_norm() function if it should use the current mini-batch’s
    # mean and standard deviation (during training) or the running averages that it
    # keeps track of (during testing).
    is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
    bn_params = {
        'is_training': is_training, 'decay': 0.99,
        'updates_collections': None,
        "scale": True  # Necessary for activation other than None, or ReLU
    }

    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.float32, shape=(None), name="y")
    
    from tensorflow.contrib.layers import fully_connected
    #from tensorflow.contrib.layers.python.layers import initializers
    # with tf.name_scope("dnn"):
    #     hidden1 = fully_connected(X, n_hidden1, scope="hidden1",activation_fn=tf.nn.elu,
    #         normalizer_fn=batch_norm, normalizer_params=bn_params)
    #     hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2",activation_fn=tf.nn.elu,
    #         normalizer_fn=batch_norm, normalizer_params=bn_params)
    #     hidden3 = fully_connected(hidden2, n_hidden3, scope="hidden3",activation_fn=tf.nn.elu,
    #         normalizer_fn=batch_norm, normalizer_params=bn_params)
    #     logits = fully_connected(hidden3, n_outputs, scope="outputs", activation_fn=None,
    #         normalizer_fn=batch_norm, normalizer_params=bn_params)
    

    # These layer definition lines can be tidied with an argument scope:
    with tf.contrib.framework.arg_scope(
        [fully_connected],
        normalizer_fn=batch_norm,
        normalizer_params=bn_params,
        weights_initializer=tf.contrib.layers.xavier_initializer(seed=1)):
        hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
        hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
        hidden3 = fully_connected(hidden2, n_hidden3, scope="hidden3")
        logits = fully_connected(hidden3, n_outputs, scope="outputs",
        activation_fn=None)

    # Define loss and train:
    with tf.name_scope("loss"):
        xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)

    # Evaluate:
    with tf.name_scope("eval"):
        model_predictions = tf.cast(tf.nn.sigmoid(logits) > 0.5, tf.int32)
        correct = tf.equal(model_predictions, tf.cast(y, tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    ################################################################################
    # Execution Phase:
    ################################################################################

    with tf.Session() as sess:
        init.run()
        accs_train = []
        accs_test = []
        for epoch in range(n_epochs):
            sess.run(training_op, feed_dict={is_training: True, X: train_x, y: train_y})
            acc_train = accuracy.eval(feed_dict={is_training: False, X: train_x, y: train_y})
            accs_train.append(acc_train)
            if test_y is not None:
                acc_test = accuracy.eval(feed_dict={is_training: False, X: test_x, y: test_y})
                accs_test.append(acc_test)
            print(epoch, "Train Acc:", acc_train)
        save_path = saver.save(sess, saver_filename)

    with tf.Session() as sess:
        saver.restore(sess, saver_filename)
        test_Ypred = model_predictions.eval({is_training: False, X: test_x})

    # Plot the accuracy
    plt.plot(np.squeeze(accs_train), label = "Train Acc")
    plt.plot(np.squeeze(accs_test), label = "Test Acc")
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()
    plt.close()

    if test_y is None:
        accs_test = None

    return test_Ypred, accs_test

Ypred, _ = tf_model_btb(train_x, train_y, valid_x,valid_y, learning_rate = 0.1, n_epochs = 600)
Ypred, _ = tf_model_btb(full_train_x, full_train_y, test, learning_rate = 0.1, n_epochs = 1500)

# itt 560
submission = pd.DataFrame({
        "PassengerId": test_passenger_id,
        "Survived": np.ravel(Ypred)
    })
# submission.to_csv('./output/tfpy_btb_submission.csv', index=False)