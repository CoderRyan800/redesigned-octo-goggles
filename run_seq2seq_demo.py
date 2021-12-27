from __future__ import print_function
import pickle
import re
from keras.models import Model, load_model
import traceback

from utils import *

num_encoder_tokens = 30
num_decoder_tokens = 42
max_encoder_seq_length = 192
max_decoder_seq_length = 64

def encode_input_text(input_token_index, text_to_encode):
    encoder_input_data = np.zeros(
        (1, max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    for t, char in enumerate(text_to_encode):
        encoder_input_data[0, t, input_token_index[char]] = 1.
    encoder_input_data[0, t + 1:, input_token_index[' ']] = 1.
    return encoder_input_data
# End encode_input_text

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

        self.reverse_input_char_index = dict(
            (i, char) for char, i in self.input_token_index.items())
        self.reverse_target_char_index = dict(
            (i, char) for char, i in self.target_token_index.items())

        self.knowledge_sentences = []

    # End initializer

    def decode_sequence(self,input_seq):
        # Encode the input as state vectors.
        states_value = self.encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, self.target_token_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h_forward, c_forward = self.decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
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
        decoded_sentence = self.decode_sequence(self.encoder_input_data)
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

#run_dual_agent_demo()
