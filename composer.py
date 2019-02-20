from batch import BatchProcessor
from midiProcessor import MIDIProcessor
import keras 
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.layers import Dropout
import numpy as np

#while using keras we dont need to get batch out of data by ourself , keras has an ready syntax for that

# batch_processor = BatchProcessor(batch_size=2, dataset_size=2)
# batch=batch_processor.get_next_batch()

midiprocessor = MIDIProcessor(2)

midiprocessor.read_files(0,2)
encoded_data = []
for item in midiprocessor.all_songs_objects:
    encoded_data.append(midiprocessor.one_hot_encode(item))

encoded_data=np.array(encoded_data)    
print(encoded_data.shape)



#training model

#Encoder
encoder_inputs = Input(shape=(None,encoded_data.shape[2]))
endocer_lstm1=LSTM(1024,return_sequences=True)
encoder_output=endocer_lstm1(encoder_inputs)
#Dropout

encoder_output= Dropout(0.3)(encoder_output)

encoder_lstm2=LSTM(1024,return_state=True)

encoder_output,encoder_h,encoder_c=encoder_lstm2(encoder_output)

encoder_states=[encoder_h,encoder_c]
#Decoder
decoder_input=Input(shape=(None,encoded_data.shape[2]))
decoder_lstm1=LSTM(1024,return_sequences=True)
decoder_output=decoder_lstm1(decoder_input,initial_state=encoder_states)#seeting the decoder intial state to encoder states
decoder_lstm2=LSTM(1024,return_sequences=True,return_state=True)
decoder_out,_,_=decoder_lstm2(decoder_output)
decoder_dence=Dense(encoded_data.shape[2],activation='softmax')
decoder_out=decoder_dence(decoder_out)


# #define model
model=Model(inputs=[encoder_inputs,decoder_input],outputs=[decoder_out])

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

model.fit(x=[encoded_data,encoded_data],y=encoded_data,batch_size=2,epochs=50) #TO DO: add validation split while having more data



# inference model
encoder_model=Model(encoder_inputs,encoder_states)

decoder_state_input_h=Input(shape=(1024,))
decoder_state_input_c=Input(shape=(1024,))
decoder_states_input=[decoder_state_input_h,decoder_state_input_c]
decoder_output=decoder_lstm1(decoder_input,initial_state=decoder_states_input)
decoder_output, state_h, state_c=decoder_lstm2(decoder_output)
decoder_states=[state_h,state_c]
# dence - softmax layer
decoder_output=decoder_dence(decoder_output)

decoder_model=Model([decoder_input]+decoder_states_input, [decoder_output]+decoder_states)



start=np.random.randint(0,len(encoded_data))
states_value=encoder_model.predict(encoded_data[start].reshape(1,60,444))
#predict recursively
target_seq=np.zeros((1,1,encoded_data.shape[2]))
target_seq[0,0,0]=1 # set the first note

decoded=[]
for i in range (60):
    output_tkn,h,c=decoder_model.predict([target_seq]+states_value)
    sample=np.argmax(output_tkn[0,-1,:])
    decoded.append(sample)
    target_seq=np.zeros((1,1,encoded_data.shape[2]))
    target_seq[0,0,sample]=1 # get the next from the prev one in testing/infering
    states_value=[h,c]
print(decoded)

















