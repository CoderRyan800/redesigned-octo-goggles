# redesigned-octo-goggles
Second stage of neural networks and reasoning project

This project represents an advance over the early agent architecture at https://github.com/CoderRyan800/alpha
which served as the basis for our original Medium article at 
https://medium.com/@ryan20083437/simple-reasoning-and-knowledge-states-in-a-lstm-based-agent-4e603780cc08
in several key ways:

1. The new agent handles handles multi-stage logical reasoning.
2. The new agent can recognize situations where the knowledge presented is contradictory.

In order to generate training and validation data for the new agent, we have borrowed
important code from the Python repository for "Artificial Intelligence: A Modern Approach", and
the repo from which we borrowed is available at
https://github.com/aimacode/aima-python.git.  The files we are using are:

1. agents.py
2. csp.py
3. utils.py
4. search.py
5. logic.py

The last file above is particularly important for generating training and test data for the agent.

Our code is based  heavily upon the seq2seq example used in the Keras documentation
at https://keras.io/examples/lstm_seq2seq/ and readers can immediately recognize the similarities.
The example at keras.io focuses on language translation, but we are focusing on propositional
reasoning instead, using a sequence-to-sequence model of the type used for translation.
