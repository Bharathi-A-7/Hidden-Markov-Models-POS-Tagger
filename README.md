# Hidden-Markov-Models-POS-Tagger

A Parts of Speech Tagger deployed on heroku . 

Access the app [here](https://pos-tagger-app.herokuapp.com/)

A Parts of Speech Tagger trained on Wall Street Journal 's vocabulary using Hidden Markov Models . Predictions are made possible by passing the Transition and Emission probabilities of the Hidden Markov Model to the Viterbi Algorithm that makes forward and backward propagation on the test corpus to generate the best sequence of POS tags for a given sentence . 

The model achieved an accuracy of 95% on the test dataset .


Future Improvements : 
1. Training on a larger vocabulary 
2. Using a Sequence model
