
import pickle
import import_ipynb
import POS_tagger as pos
from flask import Flask,request,render_template


#Import necessary data structures
states = pickle.load((open('states.pickle','rb')))
tag_counts = pickle.load((open('tag_counts.pickle','rb')))

states_mapping = pickle.load((open('states_mapping.pickle','rb')))
vocab = pickle.load((open('vocab.pickle','rb')))
TP = pickle.load((open('Transition_probabilities.pickle','rb')))
EP = pickle.load((open('Emission probabilities.pickle','rb')))


app = Flask(__name__,static_url_path='/static')

@app.route("/")
def index_page():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict_POS():
    text = request.form['text']
    
    text = text.rstrip('\n')
    
    words = text.split(' ')
    
    
    preprocessed_words = pos.preprocess_test_sample(words,vocab)
    
    #Initialize Viterbi Matrices
    test_best_probs , test_best_paths = pos.initialize(states,tag_counts,TP,EP,preprocessed_words,vocab)
    
    #Forward propagate to populate matrices
    test_best_probs , test_best_paths = pos.forward(TP,EP,preprocessed_words,test_best_probs,test_best_paths,vocab)
    
    #Back propagate to make predictions
    
    test_predictions = pos.backward(test_best_probs,test_best_paths,preprocessed_words,states)
    
    results = {}
    
    for i , pred in enumerate(test_predictions):
        results[preprocessed_words[i]] = states_mapping[pred]
        
    
    return render_template('result.html',results = results)
    
    
    
    return message

if __name__ == '__main__':
    app.run()

#https://bootstrap-menu.com/detail-animation.html
#https://bootsnipp.com/snippets/yN681




