from tensorflow.keras.layers import Dense
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.models import Sequential
import tensorflow as tf
import numpy as np
import tensorflow.keras.initializers as initializers
import tensorflow.keras as keras
from RWNN import PseudoRWNN

class EnsemblePsudoRWNN:
    def __init__(self,number_of_models,inputsize,layers,seed=42):
        self.seed=seed
        self.models = [PseudoRWNN.PseudoRWNN(inputsize,layers,seed=seed+i) for i in range(number_of_models)]

    def train(self,x,y,batch_size=8,epochs=80,learning_rate=0.1,perToTrain=0.9):
        rng=np.random.RandomState(self.seed)
        opt=keras.optimizers.Adagrad(learning_rate=learning_rate)
        self.loss=[]
        for n,model in enumerate(self.models):
            s=rng.randint(0, len(x), size=int(perToTrain*len(x))) #Randomly Sellect perToTrain % of the training patterns 
            model.compile(  loss='mean_squared_error',
                        optimizer=opt)
            history = model.fit(  x[s], y[s],
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0)
            self.loss.append(history.history["loss"][-1])
            print (n,history.history["loss"][-1])
    
    def use(self,x):
        y_hat=[model.predict(x) for model in self.models]
        return y_hat

    