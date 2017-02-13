#!/usr/bin/env python
# -*- coding: utf-8 -*-

# jarvis.py
# Tarun Gupta, University of Vermont, 2016

import websocket # for creating a connection with slackbot's RTM API
import pickle    # Allows saving python objects to disk.
import json
import urllib
import requests
import sqlite3 # database
import sklearn 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

import botsettings # local file with API token. Use your own file and do not share.
TOKEN = botsettings.API_TOKEN
DEBUG = True

def debug_print(*args):
    if DEBUG:
        print(*args)


try:
    conn = sqlite3.connect("jarvis.db") # connection object for SQLite embedded database
    c_ = conn.cursor()
    if conn:
        print ("Connection succesful!")
        #c_.execute("CREATE TABLE training_data (id INTEGER PRIMARY KEY ASC, txt text, action text)")
except:
    debug_print("Can't connect to sqlite3 database...")


def post_message(message_text, channel_id):
    requests.post("https://slack.com/api/chat.postMessage?token={}&channel={}&text={}&as_user=true".format(TOKEN,channel_id,message_text))


class Jarvis():
    
    def __init__(self): # initialize Jarvis
        self.JARVIS_MODE = None
        self.ACTION_NAME = None
        self.previous_question = None
        self.msg_count = 0
        training_data = []
        training_labels = []
        
        for row in c_.execute("SELECT * from training_data"):
            training_data.append(' '.join(map(str, (row[1:2]))))
            training_labels.append(' '.join(map(str, (row[2:3]))).upper())
        
        #encode the text labels/classes to numerical form and fit them:
        self.label_encode = preprocessing.LabelEncoder()
        self.labels_dtm = self.label_encode.fit_transform(training_labels) 
        
        # Instantiate vectorizers 
        self.count_vect = CountVectorizer(analyzer=u'word', lowercase=True, stop_words='english')
        self.tfidf_transformer = TfidfTransformer()
        
        # Transform and extract features from training text
        training_data_dtm = self.count_vect.fit_transform(training_data)
        training_tfidf = self.tfidf_transformer.fit_transform(training_data_dtm)
                
        # Instantiate the classifier & fit training data
        self.BRAIN = MultinomialNB().fit(training_tfidf, self.labels_dtm)
        
        # Dump the trained classifier to a pickle file
        pkclf = open('jarvis_brain.pkl', 'wb')
        pickle.dump(self.BRAIN, pkclf)
        pkclf.close()
    
    def on_message(self, ws, message):
        print ("\n message is :", message)
        m = json.loads(message)
        debug_print(m, self.JARVIS_MODE, self.ACTION_NAME)
        
        # only react to Slack "messages" not from bots (me):
        if m['type'] == 'message' and 'bot_id' not in m:
            if self.msg_count == 0:                 # Post welcome message for the very first message.
                post_message("Hi. My name is `Jarvis`. I was designed for *CS287* by Tarun Gupta.\
                \n\n I use a multinomial NaiveBayes algorithm to predict `LABELS`.\
                \nYou can use me in either of the following modes: \n\t`Training` or `Testing`", m['channel'])
            
            # Count the number of user messages
            self.msg_count = self.msg_count + 1
            
            # Create a lowercase copy of all incoming text for condition testing
            mtext_l =  m['text'].lower()
            
            print ("\n\n message = ", m['text'])
            if self.JARVIS_MODE == None:
                if ('training' in mtext_l):
                    post_message("Should I enter Training mode? Type `YES` or `NO` to continue...\n", m['channel'])
                    self.previous_question = "training_confirm"
                    
                if (self.previous_question == "training_confirm") and (mtext_l == "yes"):
                    self.JARVIS_MODE = 'Training'
                    post_message("OK! I am in training mode now... \n", m['channel'])
                    self.previous_question = "training_mode"
                    
                if ('testing' in mtext_l):
                    post_message("Should I enter Testing mode? Type `YES` or `NO` to continue...\n", m['channel'])
                    self.previous_question = "testing_confirm"
                    
                if (self.previous_question == "testing_confirm") and (mtext_l == "yes"):
                    self.JARVIS_MODE = 'Testing'
                    #c_.execute("SELECT * FROM list WHERE action=?", (Variable,))
                    post_message("OK! I am in Testing mode now.....\
                    \nAsk me something and I'll try to predict what you are talking about.", m['channel'])
                    self.previous_question = "testing_mode"
                
            
            if self.JARVIS_MODE != None:   
                if self.JARVIS_MODE == 'Training':                   
                    if (self.ACTION_NAME != None) and (self.previous_question == "data_requested_for_action"):
                        if ('done' in m['text'].lower()):                            
                            self.previous_question = "done_training"
                            post_message(("Done training for label `{0}`.".format(self.ACTION_NAME)), m['channel'])
                            self.ACTION_NAME = None
                            self.JARVIS_MODE = None
                            conn.close()
                            post_message("`Jarvis` is out of Training mode. \
                            \n What would you like to do?: `Training` or `Testing`.", m['channel'])
                            
                        else:
                            #self.training_data.setdefault(self.ACTION_NAME, []).append(m['text'])
                            c_.execute("INSERT INTO training_data (txt,action) VALUES (?, ?)", (m['text'], self.ACTION_NAME,))
                            conn.commit()                                
                            self.previous_question = "data_requested_for_action"                            
                            post_message(("*Training:* label `{0}`: Got it! Give me some Moe!".format(self.ACTION_NAME)), m['channel'])
                        
                                                                                                
                    if (self.ACTION_NAME != None) and (self.previous_question == "action_assigned"):
                        post_message(("*Training mode:* Great. Give me some data for: {0}".format(self.ACTION_NAME)), m['channel'])
                        self.previous_question = "data_requested_for_action"
                    
                    if (self.ACTION_NAME == None) and (self.previous_question == "label_requested"):
                        self.ACTION_NAME = m['text'].upper()
                        post_message(("*Training mode:* OK! The *Action* `LABEL` is: {0}. \
                        \nEnter `<Y>` to continue...".format(self.ACTION_NAME)), m['channel'])
                        self.previous_question = "action_assigned"
                        
                    if (self.ACTION_NAME == None) and (self.previous_question == "training_mode"):   
                        post_message("*Training mode:* What `LABEL` or name should this *Action* be?", m['channel'])
                        self.previous_question = "label_requested"

                    if (self.ACTION_NAME != None) and (self.previous_question == "done_training_confirm") and (m['text'] == "no"):
                        self.previous_question = "data_requested_for_action"
                        post_message(("Ah ok! Sorry about that. Please continue traning for `{0}`".format(self.ACTION_NAME)), m['channel'])                                
                        
                    else:
                        pass 

                if self.JARVIS_MODE == 'Testing':
                    if (self.ACTION_NAME == None) and (self.previous_question == "get_test_data"):
                        if ('done' in mtext_l):                            
                            self.previous_question = "done_testing"
                            self.JARVIS_MODE = None
                            post_message("`Jarvis` is out of Testing mode. \
                            \n What would you like to do?: `Training` or `Testing`.", m['channel'])                            
                        else:                                                        
                            sample_text = [m['text'].lower()]
                            sample_vect = self.count_vect.transform(sample_text)
                            sample_tfifd_vect = self.tfidf_transformer.transform(sample_vect)
                            
                            label_predict = self.BRAIN.predict(sample_tfifd_vect)
                            label_predict_decoded = self.label_encode.inverse_transform(label_predict)
                            post_message(("I think you are talking about: `{0}`".format(label_predict_decoded)), m['channel'])
                            self.previous_question == "get_test_data"
                                               
                    if (self.ACTION_NAME == None) and (self.previous_question == "testing_mode"):
                        self.previous_question = "get_test_data"
                pass
            
            pass
        pass



def start_rtm():
    """Connect to Slack and initiate websocket handshake"""
    r = requests.get("https://slack.com/api/rtm.start?token={}".format(TOKEN), verify=False)
    r = r.json()
    r = r["url"]
    return r


def on_error(ws, error):
    print("SOME ERROR HAS HAPPENED", error)


def on_close(ws):
    conn.close()
    print("Web and Database connections closed")


def on_open(ws):
    print("Connection Started - Ready to have fun on Slack!")



r = start_rtm()
jarvis = Jarvis()
ws = websocket.WebSocketApp(r, on_message=jarvis.on_message, on_error=on_error, on_close=on_close)
ws.run_forever()


