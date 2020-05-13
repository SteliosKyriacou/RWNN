from tensorflow.keras.layers import Dense
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np
import tensorflow.keras.initializers as initializers


#custom weight constraint 
class removeSynapses (Constraint):
    def __init__(self, size,pc,rng):
        a,b=size
        msk=rng.rand(a,b)
        while ((msk>pc) == False).sum() < 1: # danger!
            msk=rng.rand(a,b)
        msk[msk > pc] = 0
        msk[msk != 0] = 1
        self.mask=tf.Variable(msk,dtype=tf.float32)
    def __call__(self, w):
        return tf.multiply(self.mask,w)
    

def PseudoRWNN(inputshape,layers,seed):
    #sigmoid activation friendly weight initializations
    tf.random.set_seed(seed)
    # init = initializers.glorot_uniform(seed=seed) 
    init = initializers.RandomUniform(minval=-1, maxval=1, seed=seed)
    rng=np.random.RandomState(seed)
    #construct model using sequential 
    model = Sequential()
    for n,l in enumerate(layers):
        nodes,act,synrem,bias = l
        # init = initializers.RandomUniform(minval=-1/float(n+1), maxval=1/float(n+1), seed=seed)
        model.add(Dense(nodes, 
                        activation=act,
                        input_shape=(inputshape,),
                        use_bias=bias,
                        kernel_initializer=init,
                        kernel_constraint=removeSynapses((inputshape,nodes),synrem,rng))
                )
    return model
