from keras.models import Model
from keras.layers import Input, LSTM, Dense, GRU

def LSTM1(n_channels):
    input_layer = Input(shape=(None, n_channels))
    lstm = LSTM(32)(input_layer)
    dense = Dense(16, activation='relu')(lstm)
    dense2 = Dense(4, activation='relu')(dense)
    out_layer = Dense(1, activation='sigmoid')(dense2)

    model = Model(inputs=input_layer, outputs=out_layer)
    model.compile(
        'adam', 
        'binary_crossentropy',
        metrics=['accuracy'])

    return model

def LSTM2(n_channels):
    input_layer = Input(shape=(None, n_channels))
    lstm = LSTM(16)(input_layer)
    dense = Dense(8, activation='relu')(lstm)
    out_layer = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=input_layer, outputs=out_layer)
    model.compile(
        'adam', 
        'binary_crossentropy',
        metrics=['accuracy'])

    return model

def GRU1(n_channels):
    input_layer = Input(shape=(None, n_channels))
    gru = GRU(32)(input_layer)
    dense = Dense(16, activation='relu')(gru)
    dense2 = Dense(4, activation='relu')(dense)
    out_layer = Dense(1, activation='sigmoid')(dense2)

    model = Model(inputs=input_layer, outputs=out_layer)
    model.compile(
        'adam', 
        'binary_crossentropy',
        metrics=['accuracy'])

    return model
