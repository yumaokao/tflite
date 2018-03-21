import tensorflow as tf

a = tf.Variable( [1,2,3,1] )    
start_op = tf.global_variables_initializer()    
comparison = tf.equal( a, tf.constant( 1 ) )    
conditional_assignment_op = a.assign( tf.where (comparison, tf.zeros_like(a), a) )

with tf.Session() as session:
    # Equivalent to: a = np.array( [1, 2, 3, 1] )
    session.run( start_op )
    print( a.eval() )    
    # Equivalent to: a[a==1] = 0
    session.run( conditional_assignment_op )
    print( a.eval() )

# Output is:
# [1 2 3 1]
# [0 2 3 0]
