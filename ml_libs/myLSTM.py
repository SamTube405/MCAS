from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

class MyLSTM:
    def __init__(self, lstm_layers, epochs, batch_size, verbose, shuffle, loss, opt):
        self.lstm_layers = lstm_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose=verbose
        self.shuffle=shuffle
        self.loss = loss
        self.opt = opt
    
    def lstm_train(self, X, y):
        
        return_sequences=True
        
        if len(self.lstm_layers)==1:
            return_sequences=False
        
        ### Build model
        model = Sequential()
        
        for i, hidden_neurons in enumerate(self.lstm_layers):
            if i == 0:
                model.add(LSTM(hidden_neurons, input_shape=(X.shape[1], X.shape[2]), return_sequences=return_sequences))
#                 model.add(LSTM(hidden_neurons, input_shape=(X.shape[1], X.shape[2]), return_sequences=return_sequences,
#                                dropout=0.25))
            elif i < len(self.lstm_layers)-1:
                model.add(LSTM(hidden_neurons), return_sequences=return_sequences)
            else:
                model.add(LSTM(hidden_neurons))
        
        model.add(Dense(y.shape[1]))
        
        ### compile and fit
        model.compile(loss=self.loss, optimizer=self.opt)
        print(model.summary())
        
        # fit network
        history=model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose,
                          shuffle=self.shuffle)
        
        return model   