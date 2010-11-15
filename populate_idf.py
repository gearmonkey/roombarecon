#!/usr/bin/env python
# encoding: utf-8
"""
populate_idf.py

Quick and dirty tf/idf from soundcloud comments seeder.
Grabs soundcloud comments for random public tracks.  
Treats all comments for one track as a single document in the VSM.
Continues grabbing track comment until MAXDOCS is hit.
Writes out corpus every STASHAFTER tracks and at finish.
MAXTRACKNUMBER is the top usable track number.

Created by Benjamin Fields on 2010-09-04.
"""

import sys
import os

import scapi
import sc_auth

from gensim import corpora, models, similarities

from random import randrange

API_HOST = "api.soundcloud.com"

token = 'TOKEN HERE'
token_secret = 'SECRET HERE'

MAXDOCS = 50000
MAXTRACKNUMBER = 5000000
STASHAFTER = 500

stoplist = set('for a of the and to in'.split())#words to ignore
striplist = '.,:@#!_-"\''#chars to remove for the word->token process

def init_scope():
	'''
	This returns a my access root.  I may go back and add a log in apparatus, but for now, it will be public track driven.
	'''
	return scapi.Scope(scapi.ApiConnector(API_HOST, authenticator = \
		scapi.authentication.OAuthAuthenticator(sc_auth.sc_key, sc_auth.sc_secret, token, token_secret)))
		
		
class grabRandomComments(object):
	"""a simple generator that grabs random tracks, and gives the random track's comments, never give the same track twice in a given instance."""
	def __init__(self):
		self.root = init_scope()
		self.visited = []
	
	def __iter__(self):
		return self
	
	def next(self):
		this_track = None
		while this_track == None or (this_track.comment_count == 0 and this_track.tag_list == ''):
			if len(self.visited) == MAXTRACKNUMBER-1:raise StopIteration
			idx = randrange(0,MAXTRACKNUMBER)
			while idx in self.visited: 
				self.visited.append(idx)
				if len(self.visited) == MAXTRACKNUMBER-1:raise StopIteration
				idx = int(randrange(0,MAXTRACKNUMBER))
			self.visited.append(idx)
			this_track = self.root.tracks(idx)
		
		# if this_track.comment_count == 0: return []
		return this_track
		
def comments_into_tokenized_doc(track):
	'''
	takes in the comment structure for a track, spits out a tokenized document suitable for training a corpus or other vector space fun
	'''
	bag_of_words = []
	if track.comment_count != 0:
		for document in map(lambda x:x.body, track.comments()):
			bag_of_words += map(lambda x:x.strip(striplist), [word for word in document.lower().split() if (word.strip(striplist) not in stoplist) and (len(word.strip(striplist)) > 1)])
	bag_of_words += map(lambda x:x.strip(striplist), [word for word in track.tag_list.lower().split() if (word.strip(striplist) not in stoplist) and (len(word.strip(striplist)) > 1)])
	return bag_of_words

def main():
	#if you want to force a start from stratch, move the saved files or remove the trys
	try:
		a_dict = corpora.Dictionary.load('current_dictionary.dict')
	except:
		a_dict = corpora.Dictionary()
	try:
		a_corps = list(corpora.MmCorpus('current_corpus.mm'))
	except:
		a_corps = []
	commentGrabber = grabRandomComments()
	for idx in xrange(MAXDOCS):
		try:
			old = a_dict.numDocs
			a_corps.append(a_dict.doc2bow(comments_into_tokenized_doc(commentGrabber.next()), allowUpdate=True))
			if a_dict.numDocs != old + 1 :print "=======dictionary didn't update===="
		except Exception, err:
			print "+++++++++++++++++ Call just fell over here's why:+++++++++++\n\t\t\t" + str(err)
			continue
		if len(a_corps)%STASHAFTER == 0:
			print "******corpus contains {0} documents, writing out".format(len(a_corps))
			corpora.MmCorpus.saveCorpus('current_corpus.mm', a_corps)
			a_dict.save('current_dictionary.dict')
	corpora.MmCorpus.saveCorpus('current_corpus.mm', a_corps)
	a_dict.save('current_dictionary.dict')

if __name__ == '__main__':
	main()

