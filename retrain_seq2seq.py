# Now we are going to directly copy the code, licensed under the MIT license, from
# https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py.  Because we plan to
# use a sequence to sequence model to see if it will enable us to achieve our goals.

'''
#Sequence to sequence example in Keras (character-level).
This script demonstrates how to implement a basic character-level
sequence-to-sequence model. We apply it to translating
short English sentences into short French sentences,
character-by-character. Note that it is fairly unusual to
do character-level machine translation, as word-level
models are more common in this domain.
**Summary of the algorithm**
- We start with input sequences from a domain (e.g. English sentences)
    and corresponding target sequences from another domain
    (e.g. French sentences).
- An encoder LSTM turns input sequences to 2 state vectors
    (we keep the last LSTM state and discard the outputs).
- A decoder LSTM is trained to turn the target sequences into
    the same sequence but offset by one timestep in the future,
    a training process called "teacher forcing" in this context.
    It uses as initial state the state vectors from the encoder.
    Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.
- In inference mode, when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next character
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled character to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the character limit.
**Data download**
[English to French sentence pairs.
](http://www.manythings.org/anki/fra-eng.zip)
[Lots of neat sentence pairs datasets.
](http://www.manythings.org/anki/)
**References**
- [Sequence to Sequence Learning with Neural Networks
   ](https://arxiv.org/abs/1409.3215)
- [Learning Phrase Representations using
    RNN Encoder-Decoder for Statistical Machine Translation
    ](https://arxiv.org/abs/1406.1078)
'''
from __future__ import print_function
import pickle
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Bidirectional
from keras.layers.merge import Concatenate
import numpy as np

def retrain_network(data_set_filename):
    batch_size = 512  # Batch size for training.
    epochs = 1  # Number of epochs to train for.
    latent_dim = 256  # Latent dimensionality of the encoding space.
    num_samples = 50000  # Number of samples to train on.
    # Path to the data txt file on disk.  Changed from original (commented out) code.
    #data_path = 'fra-eng/fra.txt'
    #data_path = 'fra.txt'
    #data_path = 'logic_data_extended.tsv'
    data_path = data_set_filename
    # Vectorize the data.
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    line_index = 0
    for line in lines[: min(num_samples, len(lines) - 1)]:
        #input_text, target_text, _ = line.split('\t')
        line_array = line.split('\t')
        # By using line_array[0] as opposed to original line_array[1], we are now
        # allowing repetitions and random ordering, which is very important.
        input_text = line_array[0]
        target_text = line_array[3]
        if line_index < 20:
            print("input_text=%s\ntarget_text=%s\n" % (input_text, target_text))
        line_index = line_index+1
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)
    #
    # input_token_index = dict(
    #     [(char, i) for i, char in enumerate(input_characters)])
    # target_token_index = dict(
    #     [(char, i) for i, char in enumerate(target_characters)])


    fp = open('input_token_index.pickle', 'rb')
    input_token_index = pickle.load(fp)
    fp.close()

    fp = open('target_token_index.pickle', 'rb')
    target_token_index = pickle.load(fp)
    fp.close()

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.
        encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.
        decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
        decoder_target_data[i, t:, target_token_index[' ']] = 1.
    # # Define an input sequence and process it.
    # encoder_inputs = Input(shape=(None, num_encoder_tokens))
    # encoder = Bidirectional(LSTM(latent_dim, return_state=True))
    # encoder_outputs, state_h_forward, state_c_forward, state_h_backward, state_c_backward = encoder(encoder_inputs)
    # # We discard `encoder_outputs` and only keep the states.
    # state_h = Concatenate()([state_h_forward, state_h_backward])
    # state_c = Concatenate()([state_c_forward, state_c_backward])
    # encoder_states = [state_h, state_c]
    # #encoder = LSTM(latent_dim, return_state=True)
    # #encoder_outputs, state_h_forward, state_c_forward = encoder(encoder_inputs)
    # # We discard `encoder_outputs` and only keep the states.
    # #encoder_states = [state_h_forward, state_c_forward]
    #
    # # Set up the decoder, using `encoder_states` as initial state.
    # decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # # We set up our decoder to return full output sequences,
    # # and to return internal states as well. We don't use the
    # # return states in the training model, but we will use them in inference.
    # decoder_lstm = LSTM(2*latent_dim, return_sequences=True, return_state=True)
    # decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
    #                                      initial_state=encoder_states)
    # decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    # decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    # model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Run training
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = Bidirectional(LSTM(latent_dim, return_state=True))
    encoder_outputs, state_h_forward, state_c_forward, state_h_backward, state_c_backward = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    state_h = Concatenate()([state_h_forward, state_h_backward])
    state_c = Concatenate()([state_c_forward, state_c_backward])
    encoder_states = [state_h, state_c]
    #encoder = LSTM(latent_dim, return_state=True)
    #encoder_outputs, state_h_forward, state_c_forward = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    #encoder_states = [state_h_forward, state_c_forward]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(2*latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.load_weights('s2s_weights.h5')
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2)


    # Save model
    model.save('s2s.h5')
    model.save_weights('s2s_weights.h5')
    fp = open('training_history.pickle','wb')
    pickle.dump(history, fp)
    fp.close()

    # Next: inference mode (sampling).
    # Here's the drill:
    # 1) encode input and retrieve initial decoder state
    # 2) run one step of decoder with this initial state
    # and a "start of sequence" token as target.
    # Output will be the next target token
    # 3) Repeat with the current target token and current states

    # Define sampling models
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h_forward = Input(shape=(2*latent_dim,))
    decoder_state_input_c_forward = Input(shape=(2*latent_dim,))


    decoder_states_inputs = [decoder_state_input_h_forward, decoder_state_input_c_forward]

    decoder_outputs, state_h_forward, state_c_forward = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h_forward, state_c_forward]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    encoder_model.save('encoder_model.h5')
    decoder_model.save('decoder_model.h5')

    return history
# End function retrain_network

list_of_training_files = [
'logic_data_extended_00.tsv',
'logic_data_extended_01.tsv',
'logic_data_extended_02.tsv',
'logic_data_extended_03.tsv',
'logic_data_extended_04.tsv',
'logic_data_extended_05.tsv',
'logic_data_extended_06.tsv',
'logic_data_extended_07.tsv',
'logic_data_extended_08.tsv',
'logic_data_extended_09.tsv',
'logic_data_extended_10.tsv',
'logic_data_extended_11.tsv',
'logic_data_extended_12.tsv',
'logic_data_extended_13.tsv',
'logic_data_extended_14.tsv',
'logic_data_extended_15.tsv',
'logic_data_extended_16.tsv',
'logic_data_extended_17.tsv',
'logic_data_extended_18.tsv',
'logic_data_extended_19.tsv'
]
list_of_training_files = [
        'logic_data_extended_00.tsv',
        'logic_data_extended_01.tsv',
        'logic_data_extended_02.tsv',
        'logic_data_extended_03.tsv',
        'logic_data_extended_04.tsv',
        'logic_data_extended_05.tsv',
        'logic_data_extended_06.tsv',
        'logic_data_extended_07.tsv',
        'logic_data_extended_08.tsv',
        'logic_data_extended_09.tsv'
        ]
num_training_sets = len(list_of_training_files)

epochs = 128

fp = open('training_log.txt','w')

for current_epoch in range(epochs):

    fp.write("STARTING EPOCH %d\n" % (current_epoch,))

    current_set = np.random.randint(num_training_sets)

    fp.write("Using training set file %d\n" % (current_set,))

    filename = list_of_training_files[current_set]

    history = retrain_network(filename)

    loss_history = history.history['loss']

    fp.write("Loss history:\n%s\n" % (str(loss_history),))

# End epoch loop
