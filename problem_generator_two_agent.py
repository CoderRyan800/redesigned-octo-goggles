"""
problem_generator_extended.py

Initial version by R. Mukai 27 December 2019

This module is for generating logic problems to
be solved by a neural network.
"""

import re
import copy
import logging
import string
import pickle
import numpy as np

from logic import *

from utils import *

# We have to go from prototype templates containing expression placeholders
# (necessary given that neural networks require numerous examples of the same
# pattern with different atomic expressions to learn to generalize) to concrete
# templates with actual atomic expressions replacing the placeholders.  Once this
# is done, we can then generate the complex template, with concrete atomic expressions,
# that includes random repetitions in an effort to be able to train the X to Y1 mapping
# that teaches these networks to recognize sentences as units.


def create_problem_with_repetition(problem_def, max_rep=3):
    """
    create_problem_with_repetition:
    If we have a set of statements with a question that
    leads to a given answer, we have to be robust not only
    to the order of the statements but also to repetitions of
    the same true statement(s).  We define a problem in terms
    of a set of true statements and a question and answer.
    Then we create an instance by stating each of the statements
    at lest once and maybe many times in random order followed
    by the question.  The network ought to provide
    two answers:
    1. A single repeat of each statement in any order.
    2. The correct answer to the question.
    Note that we want the network to be capable of
    stating the reasons for its answer, and reciting the
    logical statements it used is a key step.  Note that
    this training would involve regurgitating irrelevant
    statements as well.
    Warning: Present training scheme expects statements
    to be recited in an order.  Future versions must accept
    any order.  This is an ADVANCED topic for LATER!
    :param problem_def: Problem definition.  Has
    list of statements, question, and answer.
    :param max_rep: Max number of repetitions of any one
    statement.
    :return: An instance consisting of a problem string,
    a first answer string consisting of the statements in
    order, and then the answer to the question.
    """

    statement_list = problem_def['statement_list']

    # Create list to hold problem statement copies, including
    # the repetitions.

    num_statements = len(problem_def['statement_list'])
    repetition_count = np.zeros((num_statements,))
    rep_choices = np.random.randint(1, max_rep + 1, size=(num_statements,))
    total_reps = np.sum(rep_choices)

    statement_choices = np.zeros((total_reps,))

    final_statement_list = [None] * total_reps

    working_statement_set = set()

    order_of_first_mention = []

    # Loop through the repetitions and put in statement choices.

    for index in range(total_reps):

        # Keep trying to choose a statement to insert until you
        # get one that still needs to be repeated.

        statement_choice = np.random.randint(num_statements)

        while repetition_count[statement_choice] >= rep_choices[statement_choice]:
            statement_choice = np.random.randint(num_statements)

        # Now increment its counter.

        repetition_count[statement_choice] = repetition_count[statement_choice] + 1

        statement_choices[index] = statement_choice

        final_statement_list[index] = statement_list[statement_choice]

        if statement_choice not in working_statement_set:
            order_of_first_mention.append(statement_choice)
            working_statement_set = working_statement_set | {statement_choice}

    # End loop

    ordered_statement_list = []

    for statement_choice in order_of_first_mention:
        ordered_statement_list.append(statement_list[statement_choice])

    # Generate period separated string for first answer

    # Run a sanity check.

    # Is each statement in the ordered list in the final list as well?

    list_errors = False

    for current_statement in final_statement_list:

        if current_statement not in ordered_statement_list:

            list_errors = True

    for current_statement in ordered_statement_list:

        if current_statement not in final_statement_list:

            list_errors = True

    # Verify ordered statement list has no repeats.

    if len(ordered_statement_list) != len(set(ordered_statement_list)):

        list_errors = True

    # Here, the "sentence_list" key corresponds to the ordered list
    # without any repetitions, but the "sentence_list_with_repetitions" key
    # corresponds to what we present to the neural network agent during training
    # with potential sentence repetition.

    problem_dictionary = {
        "sentence_list_with_repetition" : final_statement_list,
        "sentence_list" : ordered_statement_list
    }

    return problem_dictionary

# End function create_problem_with_repetition


def permutation_generator(n_vars):
    """
    permutation_generator: Create permutation dictionary to be
    used to instantiate a given logical sentence pattern.

    # Suppose you have the following very
    simple pattern:

    if x1 then x2
    x1 is true

    Then x2 is true

    Now we have specific variables x(n) above.  But
    we wish to have our network learn basic generalization
    by knowing how to do basic substitutions.  A pattern
    should contain placeholders, like this:

    if px1 then px2
    px1 is true
    Therefore px2 is true

    We can do this by creating a permutation that maps
    for example px1 => x7, px2=> x3, resulting an in
    instance:

    if x7 then x3
    x7 is true
    Therefore x3 is true

    This next function is meant to do just that.  Here,
    pp would mean pre-predicate so pp1 might, for a given
    instance, get mapped into p8 while pp2 might be mapped
    into p5.  Likewise, px means pre-x, and a specific x
    gets chosen.

    What is the reason for this?  Given the pattern

    if px1 then px2
    px1 is true
    Therefore px2 is true

    We want to create many instances using different x variables
    to teach our network to recognize simple modus ponens.

    :param n_vars: Number of possible variables, names, predicates,
    and constants.
    :return: Dictionaries to map from placeholders to actual names
    when instantiating a problem.
    """

    # Generate permutations for variables "x".

    permutation_x = np.random.permutation(n_vars)
    permutation_p = np.random.permutation(n_vars)
    permutation_a = np.random.permutation(n_vars)
    permutation_n = np.random.permutation(n_vars)

    var_x_dict = {}
    inv_var_x_dict = {}
    var_p_dict = {}
    inv_var_p_dict = {}
    var_a_dict = {}
    inv_var_a_dict = {}
    var_n_dict = {}
    inv_var_n_dict = {}

    for n in range(n_vars):

        var_p_dict['pp' + str(n)] = chr(ord('A') + permutation_x[n])
        inv_var_p_dict[chr(ord('A') + permutation_x[n])] = 'pp' + str(n)

    # End permutation loop

    permutation_instance = {
        'var_p_dict'        :       var_p_dict
    }

    main_dictionary = {}

    for sub_dictionary in permutation_instance:
        if sub_dictionary[:3] != 'inv':
            main_dictionary.update(permutation_instance[sub_dictionary])

    return main_dictionary

# End function permutation_generator

def instantiate_placeholders(permutation_dictionary, data_string):
    """
    instantiate_placeholders: Go through a data_string that may have placeholders
    and replace each one with an instance as defined by the permuation dictionary.
    :param permutation_dictionary: Dictionary mapping placeholders to values
    :param data_string: String that needs placeholders replaced
    :return: New string with placeholders replaced.
    """

    current_string = copy.deepcopy(data_string)

    for current_placeholder in permutation_dictionary:

        #current_regex = re.compile(current_placeholder)
        current_replacement = permutation_dictionary[current_placeholder]

        new_string = re.sub(current_placeholder, current_replacement, current_string)

        current_string = copy.deepcopy(new_string)

    # End loop over placeholders

    return current_string

# End instantiate_placeholders



#### PART TWO: LOGIC CODE FOR GENERATING LOGICAL DATA TRAINING SETS

def make_simple_template(sentence_list, question):

    """
    make_simple_template: Given a list of sentences and an expression to
    evaluate, create a template.
    :param sentence_list: A list of logical sentences, each as a string.
    :param question: A proposition or other statement for which we wish
    to know whether it is true, false, or unknown.
    :return: A template with sentences, question, and answer.  It is a
    dictionary with these keys and values.
        sentence_list: Original list of sentence strings.
        expr_list: List of expressions corresponding to sentences.
        question: Original question string
        question_expr: Expression corresponding to question.
        question_kb: The knowledge base made from the sentence
        expressions.
        answer: "True", "False", or "Unknown"
        kb_self_contradiction_warning_flag: This is True if the
        knowledge base is self contradictory, meaning that any
        possible query must return True because from False you can
        derive anything, including False itself, in logic!
    """

    # We begin by creating expressions out of each sentence.
    # And as expressions are generated we add them to the
    # knowledge base.

    expr_list = []

    question_kb = PropKB()

    for current_sentence in sentence_list:

        expr_list.append(expr(current_sentence))

        question_kb.tell(expr(current_sentence))

    # End expression generator loop

    question_expr = expr(question)

    resolution_result_positive_expr = pl_resolution(question_kb, question_expr)

    resolution_result_negative_expr = pl_resolution(question_kb, ~question_expr)

    kb_is_self_contradictory = False

    if resolution_result_positive_expr is False and resolution_result_negative_expr is False:

        # In this case, resolution provides neither the question expression or its
        # negation, so resolution has no answer.  Hence, answer is "Unknown"

        answer = "Unknown"

    elif resolution_result_positive_expr is True and resolution_result_negative_expr is True:

        # Here, there is a CONTRADICTION IN THE KNOWLEDGE BASE!!!!

        answer = "Contradictory"

        #print ("WARNING: KNOWLEDGE BASE IS SELF CONTRADICTORY - QUERY IS TRIVIALLY TRUE!\n")

        kb_is_self_contradictory = True

    elif resolution_result_positive_expr is True:

        answer = "True"

    else:

        answer = "False"

    # End logic to get the answer of "True", "False", or "Unknown"

    template_dictionary = {
        'sentence_list' : sentence_list,
        'expr_list' : expr_list,
        'question' : question,
        'question_expr' : question_expr,
        'answer' : answer,
        'kb_self_contradiction_warning_flag' : kb_is_self_contradictory
    }

    return template_dictionary

# End make_simple_template


def generate_problem_from_template(input_template, n_vars=10, max_reps=3):

    """
    generate_problem_from_template: Given an input template perform these steps:

    1. Create a substitution dictionary and replace the placeholders with real
    entities in the sentences and in the question.
    2. Then instantiate the sentences with repetitions.
    3. Then run the logic problem solver to obtain the answer.
    4. Set the contradiction flag.  Initially we will not try to train the
    networks with contradictions because we think this will be overly complex,
    although a contradictions means the answer to all queries is always True.

    :param input_template: A dictionary:
        "statement_list": with a value that is a list of prototype statements
            using placeholders
        "question": A query to give to the system.  This will be turned into
        a "what is ?" query so that the LSTM will learn the difference between
        a question and a statement.
    :param n_vars: Max number of variables, propositions, names, etc.
    See permutation generator for details.
    :param max_reps: Maximum times a given sentence can be repeated in a problem instance.
    :return: A problem dictionary that can be converted into neural net training data.
    """

    proto_statement_list = input_template["statement_list"]
    proto_question = input_template["question"]

    # Step 1: Create a permutation dictionary and instantiate the placeholders with
    # actual expressions.

    permutation_dictionary = permutation_generator(n_vars)

    statement_list = []

    for proto_statement in proto_statement_list:

        statement_list.append(instantiate_placeholders(permutation_dictionary,
                                                       proto_statement))

    question = instantiate_placeholders(permutation_dictionary,
                                        proto_question)

    # Step 2: Generate a dictionary that has the statements randomly ordered
    # and randomly repeated.  This dictionary also has an ordered statement list
    # where each statement appears in the same order but is given exactly once.

    initial_template = {
        'statement_list': statement_list
    }

    repetition_template = create_problem_with_repetition(initial_template, max_reps)

    # Step 3: Create a simple template using the non-repeated statement list.

    simple_problem = make_simple_template(repetition_template['sentence_list'], question)

    simple_problem['sentence_list_with_repetition'] = repetition_template['sentence_list_with_repetition']

    # We have a problem generated.  Return it.

    return simple_problem

# End function generate_problem_from_template

def combine_sentence_list(sentence_list):

    """
    combine_sentence_list: Given a list of sentences, combine into one conjunction
    for presentation to an LSTM.
    :param sentence_list: List of sentences
    :return: Combined conjunction.
    """

    num_sentences = len(sentence_list)

    if num_sentences == 0:
        return ""

    new_sentence_list = []

    for current_sentence in sentence_list:

        if len(current_sentence) <= 2:
            new_sentence_list.append(copy.deepcopy(current_sentence))
        else:
            new_sentence_list.append("(" + copy.deepcopy(current_sentence) + ")")

    # End loop

    new_string = " & ".join(new_sentence_list)

    return new_string

# End combine_sentence_list


def template_list_to_training_data(template_list, n_vars = 10, max_reps = 3, filename = None):

    """
    High level function that
    :param template_list: List of templates, each of which is a dictionary with a
    "statement_list" and a "question".
    :return: A dictionary with these keys and values:
        problem_list: A list of problems each created by generate_problem_from_template.
        X: A batch tensor that has dimensions (batch_size, time_steps, features)
        that encodes the input knowledge base with repeats.
        Y_Q: A batch tensor that encodes the knowledge base with repetitions removed.
        Y_A: A batch tensor that encodes the answers.
    """

    problem_list = []

    string_list_questions_with_repetition = []
    string_list_questions_without_repetition = []
    string_list_sentences_without_repetition = []
    string_list_answers = []

    loop_index = 0

    for current_template in template_list:

        loop_index = loop_index + 1

        #print ("DEBUG: Loop Index %d\n" % (loop_index,))

        current_problem = copy.deepcopy(generate_problem_from_template(current_template, n_vars, max_reps))

        problem_list.append(current_problem)

        string_list_questions_with_repetition.append(
            combine_sentence_list(current_problem['sentence_list_with_repetition']) +
            " . What is %s ?" % (current_problem['question']))

        string_list_questions_without_repetition.append(
            combine_sentence_list(current_problem['sentence_list']) +
            " . What is %s ?" % (current_problem['question']))

        string_list_sentences_without_repetition.append(
            combine_sentence_list(current_problem['sentence_list']))

        if current_problem['answer'] == 'Unknown':

            string_list_answers.append(current_problem['answer'] + " HELP!")


        else:

            string_list_answers.append(current_problem['answer'])

    # End loop over the input templates

    n_problems = len(string_list_questions_with_repetition)

    data_dictionary = {
        'list_of_questions_with_repetition' : string_list_questions_with_repetition,
        'list_of_questions_without_repetition' : string_list_questions_without_repetition,
        'list_of_sentences_without_repetition' : string_list_sentences_without_repetition,
        'list_of_answers' : string_list_answers
    }

    try:
        fp = open(filename,'w')
        for index in range(n_problems):
            fp.write("%s\t%s\t%s\t%s\n" % (string_list_questions_with_repetition[index],
                                           string_list_questions_without_repetition[index],
                                           string_list_sentences_without_repetition[index],
                                           string_list_answers[index]))
        fp.close()
    except:
        print ("ERROR: Unable to write to filename %s\n" % (str(filename),))
    return data_dictionary

# End template_list_to_training_data

#### PART 5: TEMPLATE CREATION
#### This part of the code is very mutable as we create various templates.

def basic_template_list():
    """
    This just returns a very simple template list, which is hard coded.
    :return: List of templates
    """

    template_list_0 = [
        # Core common sense - you can't state an answer if you know nothing and
        # if you're given an answer then use it!
        {
            "statement_list": [],
            "question": "pp1"
        },
        {
            "statement_list": ["pp1"],
            "question": "pp1"
        },
        {
            "statement_list": ["pp1"],
            "question": "~pp1"
        },
        {
            "statement_list": ["~pp1"],
            "question": "pp1"
        },
        {
            "statement_list": ["~pp1"],
            "question": "~pp1"
        },

        # Core common sense - if you're given an answer for an irrelevant variable
        # then you can't answer!

        {
            "statement_list": ["~pp1"],
            "question": "pp3"
        },
        {
            "statement_list": ["~pp1"],
            "question": "~pp3"
        },
        {
            "statement_list": ["pp1"],
            "question": "~pp3"
        },
        {
            "statement_list": ["pp1"],
            "question": "pp3"
        }
    ]

    # Next, create very simple if-then problems.
    # Start with only the if-then where you cannot draw
    # conclusions and work from there.
    template_list_1 = []
    for first_variable in ['~pp1', 'pp1']:
        for second_variable in ['~pp2', 'pp2']:
            for third_variable in ['pp1', 'pp2', 'pp3', 'pp4',
            '~pp1', '~pp2', '~pp3', '~pp4']:
                for fourth_variable in ['pp1', 'pp2', 'pp3', 'pp4',
                '~pp1', '~pp2', '~pp3', '~pp4']:
                    for first_operator in ['==>', '<=>', '&', '|', '^']:

                        dictionary_1 = {
                            'statement_list' : ["%s %s %s" % (first_variable, first_operator, second_variable)],
                            'question': "%s" % (fourth_variable,)
                        }

                        dictionary_2 = copy.deepcopy(dictionary_1)
                        dictionary_2['statement_list'].append(third_variable)

                        template_list_1.append(dictionary_1)
                        template_list_1.append(dictionary_2)

                        extraneous_var_list_1 = ['pp7', '~pp7', 'pp8', '~pp8', 'pp9', '~pp9']
                        extraneous_var_list_2 = ['pp8', '~pp8']
                        extraneous_var_list_3 = ['pp9', '~pp9']

                        extraneous_operator_list = ['==>', '<=>', '&', '|', '^']

                        extraneous_var_1 = extraneous_var_list_1[np.random.randint(len(extraneous_var_list_1))]
                        extraneous_var_2 = extraneous_var_list_2[np.random.randint(len(extraneous_var_list_2))]
                        extraneous_var_3 = extraneous_var_list_3[np.random.randint(len(extraneous_var_list_3))]
                        extraneous_operator = extraneous_operator_list[np.random.randint(len(extraneous_operator_list))]

                        extraneous_statement = "%s %s %s" % (extraneous_var_2, extraneous_operator, extraneous_var_3)

                        dictionary_3 = copy.deepcopy(dictionary_1)
                        dictionary_4 = copy.deepcopy(dictionary_2)

                        dictionary_3['statement_list'].append(extraneous_var_1)
                        dictionary_3['statement_list'].append(extraneous_statement)

                        dictionary_4['statement_list'].append(extraneous_var_1)
                        dictionary_4['statement_list'].append(extraneous_statement)

                        template_list_1.append(dictionary_3)
                        template_list_1.append(dictionary_4)

    # End loop for basic if-then

    # Next, consider two-stage if-then modus ponens.
    template_list_2 = []
    for first_variable in ['~pp1', 'pp1']:
        for second_variable in ['~pp2', 'pp2']:
            for third_variable in ['~pp3', 'pp3']:
                for fourth_variable in ['pp1', 'pp2', 'pp3', 'pp4', '~pp1', '~pp2', '~pp3', '~pp4']:
                    for fifth_variable in ['pp1', 'pp2', 'pp3', 'pp4', '~pp1', '~pp2', '~pp3', '~pp4']:
                        for first_operator in ['==>', '<=>', '&', '|', '^']:
                            for second_operator in ['==>', '<=>', '&', '|', '^']:
                                dictionary_1 = {
                                    'statement_list': ["%s %s %s" % (first_variable, first_operator, second_variable),
                                                       "%s %s %s" % (second_variable, second_operator, third_variable)],
                                    'question': "%s" % (fifth_variable,)
                                }

                                dictionary_2 = copy.deepcopy(dictionary_1)
                                dictionary_2['statement_list'].append(fourth_variable)

                                template_list_2.append(dictionary_1)
                                template_list_2.append(dictionary_2)

                                extraneous_var_list_1 = ['pp7', '~pp7', 'pp8', '~pp8', 'pp9', '~pp9']
                                extraneous_var_list_2 = ['pp8', '~pp8']
                                extraneous_var_list_3 = ['pp9', '~pp9']

                                extraneous_operator_list = ['==>', '<=>', '&', '|', '^']

                                extraneous_var_1 = extraneous_var_list_1[np.random.randint(len(extraneous_var_list_1))]
                                extraneous_var_2 = extraneous_var_list_2[np.random.randint(len(extraneous_var_list_2))]
                                extraneous_var_3 = extraneous_var_list_3[np.random.randint(len(extraneous_var_list_3))]
                                extraneous_operator = extraneous_operator_list[np.random.randint(len(extraneous_operator_list))]

                                extraneous_statement = "%s %s %s" % (extraneous_var_2, extraneous_operator, extraneous_var_3)

                                dictionary_3 = copy.deepcopy(dictionary_1)
                                dictionary_4 = copy.deepcopy(dictionary_2)

                                dictionary_3['statement_list'].append(extraneous_var_1)
                                dictionary_3['statement_list'].append(extraneous_statement)

                                dictionary_4['statement_list'].append(extraneous_var_1)
                                dictionary_4['statement_list'].append(extraneous_statement)

                                template_list_2.append(dictionary_3)
                                template_list_2.append(dictionary_4)

    #End loop for two-stage if-then

    # Next, consider three-stage if-then modus ponens.
    template_list_3 = []
    for first_variable in ['~pp1', 'pp1']:
        for second_variable in ['~pp2', 'pp2']:
            for third_variable in ['~pp3', 'pp3']:
                for fourth_variable in ['~pp4', 'pp4']:
                    for fifth_variable in ['pp1', 'pp2', 'pp3', 'pp4', 'pp5', '~pp1', '~pp2', '~pp3', '~pp4', '~pp5']:
                        for sixth_variable in ['pp1', 'pp2', 'pp3', 'pp4', 'pp5', '~pp1', '~pp2', '~pp3', '~pp4',
                                               '~pp5']:
                            for first_operator in ['==>', '<=>', '&', '|', '^']:
                                for second_operator in ['==>', '<=>', '&', '|', '^']:
                                    for third_operator in ['==>', '<=>', '&', '|', '^']:
                                        dictionary_1 = {
                                            'statement_list': ["%s %s %s" % (first_variable, first_operator, second_variable),
                                                               "%s %s %s" % (second_variable, second_operator, third_variable),
                                                               "%s %s %s" % (third_variable, third_operator, fourth_variable)],
                                            'question': "%s" % (sixth_variable,)
                                        }

                                        dictionary_2 = copy.deepcopy(dictionary_1)
                                        dictionary_2['statement_list'].append(fifth_variable)

                                        template_list_3.append(dictionary_1)
                                        template_list_3.append(dictionary_2)


                                        extraneous_var_list_1 = ['pp7', '~pp7', 'pp8', '~pp8', 'pp9', '~pp9']
                                        extraneous_var_list_2 = ['pp8', '~pp8']
                                        extraneous_var_list_3 = ['pp9', '~pp9']

                                        extraneous_operator_list = ['==>', '<=>', '&', '|', '^']

                                        extraneous_var_1 = extraneous_var_list_1[np.random.randint(len(extraneous_var_list_1))]
                                        extraneous_var_2 = extraneous_var_list_2[np.random.randint(len(extraneous_var_list_2))]
                                        extraneous_var_3 = extraneous_var_list_3[np.random.randint(len(extraneous_var_list_3))]
                                        extraneous_operator = extraneous_operator_list[np.random.randint(len(extraneous_operator_list))]

                                        extraneous_statement = "%s %s %s" % (extraneous_var_2, extraneous_operator, extraneous_var_3)

                                        dictionary_3 = copy.deepcopy(dictionary_1)
                                        dictionary_4 = copy.deepcopy(dictionary_2)

                                        dictionary_3['statement_list'].append(extraneous_var_1)
                                        dictionary_3['statement_list'].append(extraneous_statement)

                                        dictionary_4['statement_list'].append(extraneous_var_1)
                                        dictionary_4['statement_list'].append(extraneous_statement)

                                        template_list_3.append(dictionary_3)
                                        template_list_3.append(dictionary_4)

    # End loop for three-stage if-then

    template_dictionary = {
        'template_list_0' : template_list_0,
        'template_list_1' : template_list_1,
        'template_list_2' : template_list_2,
        'template_list_3' : template_list_3
    }

    return template_dictionary

# End basic_template_list

#### TODO: Create a two-agent or N-agent template generator.  Have agent 1's list contain some
#### protovariables, and have agent 2's list contain another set.  Have the question ask about a
#### protovariable that is "disjointed" from the sentences in agent 1's list.  For example,
#### agent 1 might have ['pp1 => pp2', 'pp3 => pp4'] and the question could be what is pp1?
#### Agent 1 can't solve it without knowing agent 2's list ['pp2 => pp3'], so if pp4 is false
#### then pp1 is also false, but agent 1 must collaborate to find it.  Have two answers.  First
#### verify agent 1's initial response, then verify agent 2's knowledge dump, then verify
#### agent 1's final response.  Also, design the agent so if agent 2 has contradictory knowledge
#### then verify agent 1's final response is unknown and contradiction.


def two_agent_template_list():
    """
    This just returns a very simple template list, which is hard coded.
    :return: List of templates
    """

    template_list_0 = [
        # Core common sense - you can't state an answer if you know nothing and
        # if you're given an answer then use it!
        {
            "statement_list_1": [],
            "statement_list_2": [],
            "question": "pp1"
        },
        {
            "statement_list_1": ["pp1"],
            "statement_list_2": [],
            "question": "pp1"
        },
        {
            "statement_list_1": ["pp1"],
            "statement_list_2": [],
            "question": "~pp1"
        },
        {
            "statement_list_1": ["~pp1"],
            "statement_list_2": [],
            "question": "pp1"
        },
        {
            "statement_list_1": ["~pp1"],
            "statement_list_2": [],
            "question": "~pp1"
        },
        # Also, if current agent knows nothing or knows something irrelevant,
        # it needs to ask.
        {
            "statement_list_1": [],
            "statement_list_2": ["pp1"],
            "question": "pp1"
        },
        {
            "statement_list_1": [],
            "statement_list_2": ["pp1"],
            "question": "~pp1"
        },
        {
            "statement_list_1": [],
            "statement_list_2": ["~pp1"],
            "question": "pp1"
        },
        {
            "statement_list_1": ["pp2"],
            "statement_list_2": ["~pp1"],
            "question": "~pp1"
        },

        {
            "statement_list_1": ["pp2"],
            "statement_list_2": ["pp1"],
            "question": "pp1"
        },
        {
            "statement_list_1": ["pp2"],
            "statement_list_2": ["pp1"],
            "question": "~pp1"
        },
        {
            "statement_list_1": ["pp2"],
            "statement_list_2": ["~pp1"],
            "question": "pp1"
        },
        {
            "statement_list_1": ["pp2"],
            "statement_list_2": ["~pp1"],
            "question": "~pp1"
        },

        {
            "statement_list_1": ["~pp2"],
            "statement_list_2": ["pp1"],
            "question": "pp1"
        },
        {
            "statement_list_1": ["~pp2"],
            "statement_list_2": ["pp1"],
            "question": "~pp1"
        },
        {
            "statement_list_1": ["~pp2"],
            "statement_list_2": ["~pp1"],
            "question": "pp1"
        },
        {
            "statement_list_1": ["~pp2"],
            "statement_list_2": ["~pp1"],
            "question": "~pp1"
        },

        # Core common sense - if you're given an answer for an irrelevant variable
        # then you can't answer!

        {
            "statement_list_1": ["~pp1"],
            "statement_list_2": [],
            "question": "pp3"
        },
        {
            "statement_list_1": ["~pp1"],
            "statement_list_2": [],
            "question": "~pp3"
        },
        {
            "statement_list_1": ["pp1"],
            "statement_list_2": [],
            "question": "~pp3"
        },
        {
            "statement_list_1": ["pp1"],
            "statement_list_2": [],
            "question": "pp3"
        }


        {
            "statement_list_1": ["~pp1"],
            "statement_list_2": ["~pp2"],
            "question": "pp3"
        },
        {
            "statement_list_1": ["~pp1"],
            "statement_list_2": ["~pp2"],
            "question": "~pp3"
        },
        {
            "statement_list_1": ["pp1"],
            "statement_list_2": ["~pp2"],
            "question": "~pp3"
        },
        {
            "statement_list_1": ["pp1"],
            "statement_list_2": ["~pp2"],
            "question": "pp3"
        }

        {
            "statement_list_1": ["~pp1"],
            "statement_list_2": ["pp2"],
            "question": "pp3"
        },
        {
            "statement_list_1": ["~pp1"],
            "statement_list_2": ["pp2"],
            "question": "~pp3"
        },
        {
            "statement_list_1": ["pp1"],
            "statement_list_2": ["pp2"],
            "question": "~pp3"
        },
        {
            "statement_list_1": ["pp1"],
            "statement_list_2": ["pp2"],
            "question": "pp3"
        }

    ]

    # Next, create very simple if-then problems.
    # Start with only the if-then where you cannot draw
    # conclusions and work from there.
    template_list_1 = []
    for first_variable in ['~pp1', 'pp1']:
        for second_variable in ['~pp2', 'pp2']:
            for third_variable in ['pp1', 'pp2', 'pp3', 'pp4',
            '~pp1', '~pp2', '~pp3', '~pp4']:
                for fourth_variable in ['pp1', 'pp2', 'pp3', 'pp4',
                '~pp1', '~pp2', '~pp3', '~pp4']:
                    for first_operator in ['==>', '<=>', '&', '|', '^']:

                        dictionary_1 = {
                            'statement_list_1' : ["%s %s %s" % (first_variable, first_operator, second_variable)],
                            'statement_list_2' : [],
                            'question': "%s" % (fourth_variable,)
                        }

                        dictionary_2 = copy.deepcopy(dictionary_1)
                        dictionary_2['statement_list_2'].append(third_variable)

                        template_list_1.append(dictionary_1)
                        template_list_1.append(dictionary_2)

                        extraneous_var_list_1 = ['pp7', '~pp7', 'pp8', '~pp8', 'pp9', '~pp9']
                        extraneous_var_list_2 = ['pp8', '~pp8']
                        extraneous_var_list_3 = ['pp9', '~pp9']

                        extraneous_operator_list = ['==>', '<=>', '&', '|', '^']

                        extraneous_var_1 = extraneous_var_list_1[np.random.randint(len(extraneous_var_list_1))]
                        extraneous_var_2 = extraneous_var_list_2[np.random.randint(len(extraneous_var_list_2))]
                        extraneous_var_3 = extraneous_var_list_3[np.random.randint(len(extraneous_var_list_3))]
                        extraneous_operator = extraneous_operator_list[np.random.randint(len(extraneous_operator_list))]

                        extraneous_statement = "%s %s %s" % (extraneous_var_2, extraneous_operator, extraneous_var_3)

                        dictionary_3 = copy.deepcopy(dictionary_1)
                        dictionary_4 = copy.deepcopy(dictionary_2)

                        dictionary_3['statement_list_2'].append(extraneous_var_1)
                        dictionary_3['statement_list_2'].append(extraneous_statement)

                        dictionary_4['statement_list_2'].append(extraneous_var_1)
                        dictionary_4['statement_list_2'].append(extraneous_statement)

                        template_list_1.append(dictionary_3)
                        template_list_1.append(dictionary_4)

    # End loop for basic if-then

    # Next, consider two-stage if-then modus ponens.
    template_list_2 = []
    for first_variable in ['~pp1', 'pp1']:
        for second_variable in ['~pp2', 'pp2']:
            for third_variable in ['~pp3', 'pp3']:
                for fourth_variable in ['pp1', 'pp2', 'pp3', 'pp4', '~pp1', '~pp2', '~pp3', '~pp4']:
                    for fifth_variable in ['pp1', 'pp2', 'pp3', 'pp4', '~pp1', '~pp2', '~pp3', '~pp4']:
                        for first_operator in ['==>', '<=>', '&', '|', '^']:
                            for second_operator in ['==>', '<=>', '&', '|', '^']:
                                dictionary_1 = {
                                    'statement_list_1': ["%s %s %s" % (first_variable, first_operator, second_variable)],
                                    'statement_list_2': ["%s %s %s" % (second_variable, second_operator, third_variable)],
                                    'question': "%s" % (fifth_variable,)
                                }

                                dictionary_2 = copy.deepcopy(dictionary_1)
                                dictionary_2['statement_list_2'].append(fourth_variable)

                                template_list_2.append(dictionary_1)
                                template_list_2.append(dictionary_2)

                                extraneous_var_list_1 = ['pp7', '~pp7', 'pp8', '~pp8', 'pp9', '~pp9']
                                extraneous_var_list_2 = ['pp8', '~pp8']
                                extraneous_var_list_3 = ['pp9', '~pp9']

                                extraneous_operator_list = ['==>', '<=>', '&', '|', '^']

                                extraneous_var_1 = extraneous_var_list_1[np.random.randint(len(extraneous_var_list_1))]
                                extraneous_var_2 = extraneous_var_list_2[np.random.randint(len(extraneous_var_list_2))]
                                extraneous_var_3 = extraneous_var_list_3[np.random.randint(len(extraneous_var_list_3))]
                                extraneous_operator = extraneous_operator_list[np.random.randint(len(extraneous_operator_list))]

                                extraneous_statement = "%s %s %s" % (extraneous_var_2, extraneous_operator, extraneous_var_3)

                                dictionary_3 = copy.deepcopy(dictionary_1)
                                dictionary_4 = copy.deepcopy(dictionary_2)

                                dictionary_3['statement_list_2'].append(extraneous_var_1)
                                dictionary_3['statement_list_2'].append(extraneous_statement)

                                dictionary_4['statement_list_2'].append(extraneous_var_1)
                                dictionary_4['statement_list_2'].append(extraneous_statement)

                                template_list_2.append(dictionary_3)
                                template_list_2.append(dictionary_4)

    #End loop for two-stage if-then

    # Next, consider three-stage if-then modus ponens.
    template_list_3 = []
    for first_variable in ['~pp1', 'pp1']:
        for second_variable in ['~pp2', 'pp2']:
            for third_variable in ['~pp3', 'pp3']:
                for fourth_variable in ['~pp4', 'pp4']:
                    for fifth_variable in ['pp1', 'pp2', 'pp3', 'pp4', 'pp5', '~pp1', '~pp2', '~pp3', '~pp4', '~pp5']:
                        for sixth_variable in ['pp1', 'pp2', 'pp3', 'pp4', 'pp5', '~pp1', '~pp2', '~pp3', '~pp4',
                                               '~pp5']:
                            for first_operator in ['==>', '<=>', '&', '|', '^']:
                                for second_operator in ['==>', '<=>', '&', '|', '^']:
                                    for third_operator in ['==>', '<=>', '&', '|', '^']:
                                        dictionary_1 = {
                                            'statement_list_1': ["%s %s %s" % (first_variable, first_operator, second_variable),
                                                                 "%s %s %s" % (third_variable, third_operator, fourth_variable)],
                                            'statement_list_2': ["%s %s %s" % (second_variable, second_operator, third_variable)],

                                            'question': "%s" % (sixth_variable,)
                                        }

                                        dictionary_2 = copy.deepcopy(dictionary_1)
                                        dictionary_2['statement_list_2'].append(fifth_variable)

                                        template_list_3.append(dictionary_1)
                                        template_list_3.append(dictionary_2)


                                        extraneous_var_list_1 = ['pp7', '~pp7', 'pp8', '~pp8', 'pp9', '~pp9']
                                        extraneous_var_list_2 = ['pp8', '~pp8']
                                        extraneous_var_list_3 = ['pp9', '~pp9']

                                        extraneous_operator_list = ['==>', '<=>', '&', '|', '^']

                                        extraneous_var_1 = extraneous_var_list_1[np.random.randint(len(extraneous_var_list_1))]
                                        extraneous_var_2 = extraneous_var_list_2[np.random.randint(len(extraneous_var_list_2))]
                                        extraneous_var_3 = extraneous_var_list_3[np.random.randint(len(extraneous_var_list_3))]
                                        extraneous_operator = extraneous_operator_list[np.random.randint(len(extraneous_operator_list))]

                                        extraneous_statement = "%s %s %s" % (extraneous_var_2, extraneous_operator, extraneous_var_3)

                                        dictionary_3 = copy.deepcopy(dictionary_1)
                                        dictionary_4 = copy.deepcopy(dictionary_2)

                                        dictionary_3['statement_list_2'].append(extraneous_var_1)
                                        dictionary_3['statement_list_2'].append(extraneous_statement)

                                        dictionary_4['statement_list_2'].append(extraneous_var_1)
                                        dictionary_4['statement_list_2'].append(extraneous_statement)

                                        template_list_3.append(dictionary_3)
                                        template_list_3.append(dictionary_4)

    # End loop for three-stage if-then

    template_dictionary = {
        'template_list_0' : template_list_0,
        'template_list_1' : template_list_1,
        'template_list_2' : template_list_2,
        'template_list_3' : template_list_3
    }

    return template_dictionary

# End two_agent_template_list


#### MAIN CODE BELOW FOR GENERTING PROBLEM SETS

#### TODO AS OF 29 AUGUST 2022: We must copy make_problem_tsv_file, which is based on
#### basic_template_list, and create a new function make_problem_two_agent_tsv_file.
#### This function will work with the output of two_agent_template list in a way that is
#### analogous to how make_problem_tsv_file works.  This is because we must gather test
#### statistics for two agent cooperation for various types of problems, including
#### problems the agent solves on its own, problems where things are found to be contradictory,
#### and problems where the agent asks for help in order to learn whether the answer is
#### True, False, not enough information, or contradictory.  We must get accuracy statistics
#### for those cases, and especially those cases where the agent must demonstrate that it
#### knows its knowledge is insufficient or contradictory, to demonstrate the core of
#### self-awareness as we have defined it.  The paper would have to contain a table
#### describing these problems and the overall accuracy solving these problems.  Also
#### show accuracy when the agent is able to determine True, False, or contradictory
#### on its own without a second agent.

def make_problem_tsv_file(filename):

    long_template_dictionary = basic_template_list()

    result_list = []

    num_problems = 25000

    percentage_targets = {
        'True' : 25.0,
        'False' : 25.0,
        'Unknown' : 25.0,
        'Contradictory' : 25.0
    }

    problem_index = 0

    for problem_index in range(num_problems):

        # Randomly select a number.

        random_number = np.random.rand()
        # Decide on answer target.
        if random_number < 0.400:
            answer_target = 'True'
        elif random_number < 0.800:
            answer_target = 'False'
        elif random_number < 0.900:
            answer_target = 'Unknown'
        else:
            answer_target = 'Contradictory'

        problem_answer = None
        # Select another random number to decide on template complexity.
        random_number = np.random.rand()

        if random_number < 0.250:
            long_template_list = long_template_dictionary['template_list_0']
        elif random_number < 0.500:
            long_template_list = long_template_dictionary['template_list_1']
        elif random_number < 0.750:
            long_template_list = long_template_dictionary['template_list_2']
        else:
            long_template_list = long_template_dictionary['template_list_3']

        num_templates = len(long_template_list)

        attempt_index = 0
        while ((problem_answer is None or problem_answer != answer_target) and
            attempt_index < 50):
            random_choice = np.random.randint(num_templates)
            current_template = long_template_list[random_choice]
            problem_instance = generate_problem_from_template(current_template)
            problem_answer = problem_instance['answer']
            attempt_index = attempt_index + 1
        # End loop to keep selecting randomly until we get what we need.

        result_list.append(current_template)

        if problem_index % 1000 == 0:
            print("problem_index = %d" % (problem_index,))

            print (current_template)
            print (problem_instance)
            print ("%s = %s" % (answer_target, problem_answer))

            print("\n\n")

    # Test in earnest - must be able to generator data!

    long_template_list = basic_template_list()

    data_dictionary = template_list_to_training_data(template_list = result_list,
                                                     filename = filename)
# End function make_problem_tsv_file
#
# list_of_files = [
# 'logic_data_extended_00.tsv',
# 'logic_data_extended_01.tsv',
# 'logic_data_extended_02.tsv',
# 'logic_data_extended_03.tsv',
# 'logic_data_extended_04.tsv',
# 'logic_data_extended_05.tsv',
# 'logic_data_extended_06.tsv',
# 'logic_data_extended_07.tsv',
# 'logic_data_extended_08.tsv',
# 'logic_data_extended_09.tsv',
# 'logic_data_extended_10.tsv',
# 'logic_data_extended_11.tsv',
# 'logic_data_extended_12.tsv',
# 'logic_data_extended_13.tsv',
# 'logic_data_extended_14.tsv',
# 'logic_data_extended_15.tsv',
# 'logic_data_extended_16.tsv',
# 'logic_data_extended_17.tsv',
# 'logic_data_extended_18.tsv',
# 'logic_data_extended_19.tsv',
# 'logic_data_extended_20.tsv',
# 'logic_data_extended_21.tsv',
# 'logic_data_extended_22.tsv',
# 'logic_data_extended_23.tsv',
# 'logic_data_extended_24.tsv',
# ]
# list_of_files = [
# 'logic_data_extended_25.tsv',
# 'logic_data_extended_26.tsv',
# 'logic_data_extended_27.tsv',
# 'logic_data_extended_28.tsv',
# 'logic_data_extended_29.tsv',
# 'logic_data_extended_30.tsv',
# 'logic_data_extended_31.tsv',
# 'logic_data_extended_32.tsv',
# 'logic_data_extended_33.tsv',
# 'logic_data_extended_34.tsv',
# 'logic_data_extended_35.tsv',
# 'logic_data_extended_36.tsv',
# 'logic_data_extended_37.tsv',
# 'logic_data_extended_38.tsv',
# 'logic_data_extended_39.tsv',
# 'logic_data_extended_40.tsv',
# 'logic_data_extended_41.tsv',
# 'logic_data_extended_42.tsv',
# 'logic_data_extended_43.tsv',
# 'logic_data_extended_44.tsv',
# 'logic_data_extended_45.tsv',
# 'logic_data_extended_46.tsv',
# 'logic_data_extended_47.tsv',
# 'logic_data_extended_48.tsv',
# 'logic_data_extended_49.tsv',
# ]

list_of_files = []

#for index in range(50,200):
for index in range(10,250):
    list_of_files.append('logic_data_extended_%02d.tsv' % (index,))

for filename in list_of_files:
    make_problem_tsv_file(filename)
