print("a")

import tensorflow as tf

print("b")

def square_loss(val, func):
    return tf.reduce_mean(tf.square(func(val)))


def apply_gradient_descent(x0, f_function, eta, num_steps):
    x = x0
    # TODO put several lines of code in here.
    # You might also want to create a helper function to perform the differentiation.

    for i in range(0, num_steps):
        with tf.GradientTape() as tape:
            tape.watch([x])

            dx = f_function(x)

        grads = tape.gradient(dx, [x])

        #optimizer.apply_gradients(zip(grads, [x]))
        x = (x - (eta * grads[0]))

        print("x=%.3f, f=%.3f" % (x.numpy(), f_function(x).numpy()))  # show progress

    return x

def f_function(x): # This is the function we are trying to find the minimum of
    return x*x
x0=tf.constant(2.0, tf.float32)
x=apply_gradient_descent(x0, f_function, eta=0.25, num_steps=15)
print("Final x",round(x.numpy(),2))