from __future__ import print_function
import pickle
import re
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.merge import Concatenate
import numpy as np
import traceback
from logic import *

from utils import *


batch_size = 64  # Batch size for training.
epochs = 250  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 50000  # Number of samples to train on.
# Path to the data txt file on disk.  Changed from original (commented out) code.
#data_path = 'fra-eng/fra.txt'
#data_path = 'fra.txt'
data_path = 'logic_data_extended.tsv'
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
    # Major addition: Replace the question with the word HELP
    # and make the answer equal to the sentences without repetition.

    input_array = input_text.split(" . ")
    target_array = line_array[1].split(" . ")

    input_text = input_array[0].strip() + " . " + "HELP"
    target_text = target_array[0].strip()

    # if line_index < 20:
    #     print("input_text=%s\ntarget_text=%s\n" % (input_text, target_text))
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
# End the loop

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

# print('Number of samples:', len(input_texts))
# print('Number of unique input tokens:', num_encoder_tokens)
# print('Number of unique output tokens:', num_decoder_tokens)
# print('Max sequence length for inputs:', max_encoder_seq_length)
# print('Max sequence length for outputs:', max_decoder_seq_length)

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


def encode_input_text(input_token_index, text_to_encode):
    encoder_input_data = np.zeros(
        (1, max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    for t, char in enumerate(text_to_encode):
        encoder_input_data[0, t, input_token_index[char]] = 1.
    encoder_input_data[0, t + 1:, input_token_index[' ']] = 1.
    return encoder_input_data
# End encode_input_text


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

    encoder_input_data[i, :, :] = encode_input_text(input_token_index, input_text)

    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
    decoder_target_data[i, t:, target_token_index[' ']] = 1.

def load_token_indices(input_token_index_filename, target_token_index_filename):
    fp = open(input_token_index_filename, 'rb')
    input_token_index = pickle.load(fp)
    fp.close()

    fp = open(target_token_index_filename, 'rb')
    target_token_index = pickle.load(fp)
    fp.close()

    return {
        'input_token_index' : input_token_index,
        'target_token_index' : target_token_index
    }

# End load_token_indices

def check_expression(input_string):

    try:

        input_expr = expr(input_string)
        return str(input_expr)
    except:
        # Print the stack trace
        print("Error: input %s is not a logical expression!\n" % (str(input_string),))
        traceback.print_exc()
        print("See stack trace above\n")
        return None

# End check_expression

def check_expression_list(list_of_strings):
    try:
        new_list = []
        for current_item in list_of_strings:
            check_result = check_expression(current_item)
            if check_result is not None:
                new_list.append(check_expression(current_item))
            else:
                print("WARNING: Invalid item in list of expression strings - see stack trace!\n")
        return new_list
    except:
        # Print stack trace
        print("Error: One or more expressions cannot be processed or input is not a valid list of expression strings!\n")
        traceback.print_exc()
        return None
# End check_expression_list

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

#model = load_model('s2s.h5')

# Define sampling models

# encoder_inputs = model.input[0]
# encoder_outputs, state_h_forward, state_c_forward, state_h_backward, state_c_backward = model.layers[1].output
# state_h_enc = Concatenate()([state_h_forward, state_h_backward])
# state_c_enc = Concatenate()([state_c_forward, state_c_backward])
# encoder_states = [state_h_enc, state_c_enc]
# encoder_model = Model(encoder_inputs, encoder_states)

encoder_model = load_model('encoder_model.h5')
#
# decoder_inputs = model.input[1]   # input_2
# decoder_state_input_h = Input(shape=(2*latent_dim,), name='input_3')
# decoder_state_input_c = Input(shape=(2*latent_dim,), name='input_4')
# decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
# decoder_lstm = model.layers[5]
# decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
#     decoder_inputs, initial_state=decoder_states_inputs)
# decoder_states = [state_h_dec, state_c_dec]
# decoder_dense = model.layers[6]
# decoder_outputs = decoder_dense(decoder_outputs)
# decoder_model = Model(
#     [decoder_inputs] + decoder_states_inputs,
#     [decoder_outputs] + decoder_states)

decoder_model = load_model('decoder_model.h5')

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h_forward, c_forward = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h_forward, c_forward]

    return decoded_sentence

# error_count = 0
# for seq_index in range(5000):
#     # Take one sequence (part of the training set)
#     # for trying out decoding.
#     input_seq = encoder_input_data[seq_index: seq_index + 1]
#     decoded_sentence = decode_sequence(input_seq)
#     if target_texts[seq_index].strip() != decoded_sentence.strip():
#         error_count = error_count + 1
#
#     if seq_index % 1000 == 0 or seq_index % 1000 == 1:
#         print('-')
#         print('Input sentence:', input_texts[seq_index])
#         print('Target sentence:', target_texts[seq_index])
#         print('Decoded sentence:', decoded_sentence)
#         if target_texts[seq_index].strip() != decoded_sentence.strip():
#
#             print("WRONG")
#         else:
#             print("CORRECT")
#         print("%d errors out of %d examples so far\n" % (error_count, seq_index+1))

class nn_entity:

    def __init__(self):

        self.encoder_model = load_model('encoder_model.h5')
        self.decoder_model = load_model('decoder_model.h5')

        fp = open('input_token_index.pickle', 'rb')
        self.input_token_index = pickle.load(fp)
        fp.close()

        fp = open('target_token_index.pickle', 'rb')
        self.target_token_index = pickle.load(fp)
        fp.close()

        self.knowledge_sentences = []

    # End initializer

    def run_network(self, input_sentence):

        self.encoder_input_data = np.zeros(
            (1, max_encoder_seq_length, num_encoder_tokens),
            dtype='float32')
        self.decoder_input_data = np.zeros(
            (1, max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')
        self.decoder_target_data = np.zeros(
            (1, max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')

        self.encoder_input_data[0, :, :] = encode_input_text(self.input_token_index, input_sentence)
        decoded_sentence = decode_sequence(self.encoder_input_data)
        return_sentence = decoded_sentence.strip()

        return return_sentence

    # End run_network

    def add_knowledge(self, knowledge_sentence):

        self.knowledge_sentences.append(knowledge_sentence.strip())

    # End add_knowledge

    def ask_question(self, question):

        query_string = ""

        n_sentences = len(self.knowledge_sentences)

        for index in range(n_sentences-1):

            query_string = query_string + "%s & " % self.knowledge_sentences[index]

        query_string = query_string + "%s . %s" % (self.knowledge_sentences[-1], question)

        network_answer = self.run_network(query_string)

        return network_answer

    # End ask_question

# End class declaration for nn_entity

def run_dual_agent_demo(agent_1_initial_knowledge_list = ['~A'],
                        agent_2_initial_knowledge_list = ['C ==> A'],
                        question_for_agent_1 = 'What is C ?'):

  # Test code for the entity below.
  # Agent 1 was called test_obj
  test_obj = nn_entity()
  # Agent 2 was called test_obj_2
  test_obj_2 = nn_entity()

  regex_help = re.compile('HELP')
  # Code below checks to be sure expressions are valid expressions and not
  # unusable nonsense.
  agent_1_knowledge_list = check_expression_list(agent_1_initial_knowledge_list)

  agent_2_knowledge_list = check_expression_list(agent_2_initial_knowledge_list)
  # Add knowledge expressions to each agent's knowledge base
  for item in agent_1_knowledge_list:
      test_obj.add_knowledge(item)

  for item in agent_2_knowledge_list:
      test_obj_2.add_knowledge(item)
  # Define the question we ask agent 1
  the_question = question_for_agent_1
  # Lines below are the interaction between the two agents
  print ("Agent 1 initial knowledge below:\n")

  for item in agent_1_knowledge_list:
      print("%s\n" % (item,))

  print ("\nAgent 2 initial knowledge below:\n")

  for item in agent_2_knowledge_list:
      print("%s\n" % (item,))

  print ("\nAsking agent 1 this question: '%s'\n" % (the_question,))

  result_1 = test_obj.ask_question(the_question).strip()

  print("Agent 1 response: %s\n" % (result_1,))
  # If agent 1 has to ask for help, then agent 2 will dump its knowledge base
  # to aid agent 2.  Otherwise agent 1 just answers the question.
  if regex_help.search(result_1) is None:

      print ("Agent 1 answer = %s\n" % (result_1,))

  else:

      print("Agent 1 is asking for help.\n")

      result_2 = test_obj_2.ask_question('HELP')

      agent_2_knowledge_list = result_2.split('.')

      print("Agent 2 has answered - adding knowledge to agent 1\n")

      for sentence in agent_2_knowledge_list:
          test_obj.add_knowledge(sentence.strip())
          print("Telling Agent 1: %s\n" % (sentence,))

      print("Asking agent 1 the same question again\n")

      result3 = test_obj.ask_question(the_question).strip()

      print("Agent 1 answer: %s\n" % (result3,))

  # End logic for handling possible help request
# End function run_dual_agent_demo
