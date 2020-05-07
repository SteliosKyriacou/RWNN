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
    init=initializers.RandomUniform(minval=-1, maxval=1, seed=seed)
    rng=np.random.RandomState(seed)
    print ("1")
    #construct model using sequential 
    model = Sequential()
    for l in layers:
        nodes,act,synrem = l
        model.add(Dense(nodes, 
                        activation=act,
                        input_shape=(inputshape,),
                        use_bias=False,
                        kernel_initializer=init,
                        kernel_constraint=removeSynapses((inputshape,nodes),synrem,rng))
                 )
    return model
