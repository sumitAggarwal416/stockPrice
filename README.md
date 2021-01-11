# stockPrice

In this program, I used a machine learning technique called LSTM (Long Term Short Memory) to predit the price of stocks. The program is fairly simple and only considers one factor out of many to predict the price.

LSTM is an artificial Recurrent Neural Network (RNN) architecture, which has feedback connections. It is able to store past information that is important and discard the information that is not.

I used 80% of the data to train the model and the rest 20% to test it, following the old 80-20 rule.

My model has 2 LSTM layers (50 neurons each) and 2 Dense layers (one with 25 and the other (output layer) with 1 neurons respectively).  
