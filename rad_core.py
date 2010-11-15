#!/usr/bin/env python
# encoding: utf-8
"""
rad_core.py

the core radio player functionality.  

Created by Benjamin Fields on 2010-09-02.
"""

import os.path
current_dir = os.path.dirname(os.path.abspath(__file__))

import cherrypy
import scapi
import sc_auth
import igraph
import random
from  pyechonest import track, config
from urllib2 import HTTPError
from simplejson import dumps

from gensim import corpora, models, similarities, matutils
from populate_idf import comments_into_tokenized_doc

config.ECHO_NEST_API_KEY = "EN KEY HERE"

import numpy as np
from scipy import spatial

API_HOST = "api.soundcloud.com"
genres = ["Alternative", "Ambient", "Bass", "Dance", "Deep", "Drum & bass", "Dub", "Dubstep", "Electronic", "Experimental", "Funk", "Hardcore", "Hiphop", "House", "Independent", "Instrumental", "Minimal", "Music", "Pop", "Progressive", "Rap", "Remix", "Rnb", "Rock", "Tech", "Techno", "Trance"]
token = 'k4Tt2rJlZM0r92IHpROIIQ'
token_secret = 'msDqMeLboio56vYu5TJDB1FGuXs7gEfpB04ZHFhE84'

# FRIENDCAP = 10 #if a user follows/is followed by more than this many people, ignore (speed hack) no hard c

try:
	a_dict = corpora.Dictionary.load('current_dictionary.dict')
except:
	a_dict = corpora.Dictionary()
try:
	a_corps = list(corpora.MmCorpus('current_corpus.mm'))
	print "properly loaded a_corps"
except:
	a_corps = []

def init_scope():
	'''
	This returns a my access root.  I may go back and add a log in apparatus, but for now, it will be public track driven.
	'''
	return scapi.Scope(scapi.ApiConnector(sc_auth.API_HOST, authenticator = \
		scapi.authentication.OAuthAuthenticator(sc_auth.sc_key, sc_auth.sc_secret, token, token_secret)))
		
def en_timbre(a_track, b_track, distance='cos'):
	'''
	computes timbral distance using EN features
	distance = ['cos'|'euc'|'man'|'all'] 
	default is cos distance, can do all
	'''
	a_bits = track.track_from_url(a_track["streaming"])
	b_bits = track.track_from_url(b_track["streaming"])
	
	a_timbre = np.zeros((len(a_bits.segments), 12))
	for idx, seg in enumerate(a_bits.segments):
		a_timbre[idx] = seg['timbre']
	a_stack = np.hstack((a_timbre.mean(axis=0),a_timbre.std(axis=0)))
	
	b_timbre = np.zeros((len(b_bits.segments), 12))
	for idx, seg in enumerate(b_bits.segments):
		b_timbre[idx] = seg['timbre']
	b_stack = np.hstack((b_timbre.mean(axis=0),b_timbre.std(axis=0)))
	if distance == 'cos':
		return spatial.distance.cosine(a_stack, b_stack)		
	elif distance == 'euc':
		return spatial.distance.euclidean(a_stack, b_stack)		
	elif distance == 'man':
		return spatial.distance.cityblock(a_stack, b_stack)#as in manhatten distance
	elif distance == 'all':
		euc = spatial.distance.euclidean(a_stack, b_stack)
		cos = spatial.distance.cosine(a_stack, b_stack)
		man = spatial.distance.cityblock(a_stack, b_stack)#as in manhatten distance
		return euc, cos, man
		
def vsm_dist(song_A, song_B):
	# try:
	tif = models.TfidfModel(a_corps)
	a_tif = tif[song_A['tokenized_comments']]
	b_tif = tif[song_B['tokenized_comments']]
	dist =  matutils.cossim(a_tif, b_tif)
	if dist == 0: 
		dist = 0.0000001#avoid the div by 0
	return dist
	# except Exception, err:
	# 	print "Distance fail. Reason: " + str(err) 
	# 	return 1

def get_distance(song_A, song_B, method="social_only"):
	"""
	this is the distance picker.
	the value of <method> will determine which distance metric is used
	 "social only" unweighted, which is basically just a pure random walk.
	 "timbre sim" en timbre sim but it's sloooooow
	 "tfidf" tfidf
	"""
	if method == "social_only":
		return 1
	elif method == "timbre_sim":
		return en_timbre(song_A, song_B)
	elif method == "tfidf":
		return vsm_dist(song_A, song_B)
	raise NotImplementedError("that distance metric is not available")
	
def fill_node(node, track):
	"""add attributes from track (and it's user) to the node in the graph"""
	global a_corps
	node['title'] = track.title
	node['track_id'] = track.id
	node['perm_url'] = track.permalink_url
	try:
		node['artwork_url'] = str(track.artwork_url)
	except:
		node['artwork_url'] = ''
	try:
		node['streaming'] = str(track.stream_url)
	except:
		node['streaming'] = ''
	node['artist'] = track.user.username
	node['artist_url'] = track.user.permalink_url
	node['artist_id'] = track.user.id
	try:
		node['tokenized_comments'] = a_dict.doc2bow(comments_into_tokenized_doc(track), allowUpdate=True)
		a_corps += [node['tokenized_comments']]
	except HTTPError,err:
		print "ran into an HTTPerror: "+ str(err) + ":: retrying..."
		node['tokenized_comments'] = a_dict.doc2bow(comments_into_tokenized_doc(track), allowUpdate=True)
	except Exception, err:
		print "fail token"

class Recon:
	
	def index(self):
		"""
		shouldn't ever really end up here just a stop over
		"""
		return """
		<h2>Hello and welcome to SC-EN similarity calculator</h2>  
		Uses the echonest timbre data to find a rough timbral distance between any two streamable soundcloud songs<br>
		add two soundcloud track ids to create the url to the similarty output:<br>
		[base url]/[track id one]/[track id two]<br>
		<br><br>
		Here are some examples to get you started:
		<ul>
		<li><a href='./4817938/4877079'>4817938 to 4877079</a></li>
		<li><a href='./4817938/228976'>4817938 to 228976</a></li>
		<li><a href='./4817938/401877'>4817938 to 401877</a></li>
		<li><a href='./228976/401877'> 228976 to 401877</a></li>
		</ul>
		<br>Note, for a number of reasons, loading the similarity score may take a little while (~1 minute).
		<br>
		<br>direct question to Ben at the hack or b.fields@gold.ac.uk or @alsothing (twitter) or beqqn (on irc)
		"""
	index.exposed = True
	
	def playlist(self, start_id=None, end_id=None, half_length=4, distance="social_only", friendcap = 5, trackcap=5,fmt="html"):
		"""
		make some playlists.  using a bilateral beam searchish thing.
		start song is vertex 0
		end song is vertex 1
		
		"""
		root = init_scope()
		if start_id == None or end_id == None:
			tracks = []
			select_text = ""
			for genre in genres:
				try:
					a_track = random.sample(list(root.tracks(params={'genres':[genre], 'order':'hotness', 'limit':5})),1)[0]
					select_text+="""\t\t\t<option value="{0}">{1}</option>\n""".format(a_track.id, genre.upper()+": "+a_track.title+' - '+a_track.user.username)
				except ValueError, err:
					print "Encountered ValueError with genre "+genre+" Reason: "+ str(err)
				except Exception, err:
					print "Encountered unknown error with genre "+genre+" Reason: "+ str(err)
			return"""<head>
		<title>Roomba Recon :: Finding a path through SoundCloud's Jukebox</title>
		<script type="text/javascript" src="/js/jquery-1.4.1.min.js"></script>
		<script type="text/javascript" src="/js/core.js"></script>
	</head>
	<body>
		<div style="width:800px;margin:25px auto;font-family:helvetica;">
		<img src="../images/roomba.jpg" style="text-align:center" alt="Roomba Recon banner"\>
		<h3 style="text-align:center">Finding a path through SoundCloud's Jukebox</h3>
		<br><br><div class="contents">
		Listening your way through the Soundcloud has never been easier!<br>
		Simply select a starting and ending track to begin.<br>
		Note, playlist will take a couple minutes to render.<br>
		Here are some static examples (some may be cached) that should work:
		<ul>
			<li><a href='./4817938/4877079'>Noisia & Spor - Falling Through (VSN009) to Malfunction - NiT GriT</a></li>
			<li><a href='./1962293/2224495'>John Legend vs. Lady Gaga - Used To Love You (Dance Mix\Mashup) Dj S.I.R. Rremix - Mixes and Mashups #2 to Kesha Vs Kevin Rudolf - Tik Tok _ Let It Rock (Rock Dance Mix) Dj MutantMixes - Mixes and Mashups #3</a></li>
			<li><a href='./4817938/401877'>Noisia & Spor - Falling Through (VSN009) to Time - Bad Mood Mix (addicted to junk) - kurtjx</a></li>
			<li><a href='./228976/401877'> EnLaSelvaMvt2 - GMO's Crusty Funk Mix - bfields to Time - Bad Mood Mix (addicted to junk) - kurtjx</a></li>
		</ul>
		
		or select a track from these two boxes (dynamicly generated, might fall over)
		<div class='form'>
			<h6 style='display:inline'>Start Song:</h6>
			<select name="start_id" id="start">
			{0}
			</select><br>
			<h6 style='display:inline'>End Song:</h6>
			<select name="end_id" id="end">
			{0}
			</select>
			<p style="font-sizs:0.7em"><h6 style="display:inline">Select a cost function:</h6><br>
			<input type="radio" name="distance" id="distance" checked="checked" value="social_only"  /><span style="font-size:0.7em">only social connections (1.5 min gen time)
			<input type="radio" name="distance" id="distance" value="tfidf"/>cheap tag-based similarities (2-4 min gen time)
			<input type="radio" name="distance" id="distance" value="timbre_sim"/>expensive audio-based similarities (sloooowwww...)</span>
			<input type="submit" value="generate playlist" id="submit"/>
		</div>
		</div>
		<p style='font-size:small'>created by <a href='http://benfields.net'>Ben Fields</a> as part of the <a href="http://http://london.musichackday.org/2010/">2010 London Music Hackday</a>. <a href="../pages/about.html">about</a>. <a href="../">compute echonest similarity for soundcloud tracks</a></p>
		</div>
	</body>
			""".format(select_text)
		#indicies of the start and end nodes
		start_idx = 0
		start_track = root.tracks(start_id)
		front_playlist = [0]
		front_vert_list = [0]
		end_idx = 1
		end_track = root.tracks(end_id)
		end_playlist = [1]
		end_vert_list = [1]
		#indicies of the current endpoints
		front_idx = 0 
		back_idx = 1
		G = igraph.Graph(n=2)#, directed=True |not just yet...
		fill_node(G.vs[start_idx], start_track)
		fill_node(G.vs[end_idx], end_track)
		for i in xrange(half_length):
			################
			#push out the front
			friends = root.users(G.vs[front_idx]['artist_id']).followings()
			new_edges = 0
			print "computing level {0} front...".format(i)
			for friend in friends[:friendcap]:
				track_count = 0
				if friend['id'] in G.vs['artist_id']:
					print "found a cycle, skiping..."
					continue
				try:
					tracks = root.users(friend['id']).tracks()
				except HTTPError,err:
					print "ran into an HTTPerror: "+ str(err) + ":: retrying..."
					tracks = root.users(friend['id']).tracks()
				for track in tracks:
					if track_count >= trackcap: break
					G.add_vertices(1)
					fill_node(G.vs[len(G.vs)-1], track)
					front_vert_list.append(len(G.vs)-1)
					G.add_edges((front_idx, len(G.vs)-1))
					new_edges += 1
					try:
						G.es[len(G.es)-1]['cost'] = float(get_distance(G.vs[start_idx], G.vs[len(G.vs)-1], method=distance)
							) /float(get_distance(G.vs[len(G.vs)-1], G.vs[end_idx], method=distance))
					except Exception, err:
						print "unable to compute cost.\nnumerator: {0}\ndenomenator: {1}\nerr msg:{2}".format(get_distance(G.vs[start_idx], G.vs[len(G.vs)-1], method=distance),get_distance(G.vs[len(G.vs)-1], G.vs[end_idx], method=distance), err)
					track_count += 1
				print "added {0} new edges".format(new_edges)
			try:
				new_es = G.es[(len(G.es)-new_edges):]
				print "in the {0} sized es".format(new_edges)
				best_edges = new_es.select(cost=sorted(set(new_es['cost']))[0])
				if len(best_edges) != 1:
					winner = random.sample(best_edges, 1)[0]
				else:
					winner = best_edges[0]
				front_playlist.append(winner.target)
				front_idx = winner.target
			except (IndexError, KeyError):
				print "ran into an Index or key error on front edge add.  Was going to add {0} edges".format(new_edges)
			################
			#push out the back
			friends = root.users(G.vs[end_idx]['artist_id']).followers()
			new_edges = 0
			print "computing level {0} back...".format(i)
			for friend in friends[:friendcap]:
				if friend['id'] in G.vs['artist_id']:
					print "found a cycle, skiping..."
					continue
				try:
					tracks = root.users(friend['id']).tracks()
				except HTTPError,err:
					print "ran into an HTTPerror: "+ str(err) + ":: retrying..."
					tracks = root.users(friend['id']).tracks()
				for track in tracks:
					G.add_vertices(1)
					fill_node(G.vs[len(G.vs)-1], track)
					end_vert_list.append(len(G.vs)-1)
					G.add_edges((back_idx, len(G.vs)-1))
					new_edges += 1
					try:
						G.es[len(G.es)-1]['cost'] = float(get_distance(G.vs[len(G.vs)-1], G.vs[end_idx], method=distance)
							) /float(get_distance(G.vs[start_idx], G.vs[len(G.vs)-1], method=distance))
					except Exception, err:
						print "unable to compute cost.\nnumerator: {0}\ndenomenator: {1}\nerr msg:{2}".format(get_distance(G.vs[len(G.vs)-1], G.vs[end_idx], method=distance),get_distance(G.vs[start_idx], G.vs[len(G.vs)-1], method=distance), err)
			print "added {0} new edges".format(new_edges)
			try:
				new_es = G.es[(len(G.es)-new_edges):]
				print "in the {0} sized es".format(new_edges)
				best_edges = new_es.select(cost=sorted(set(new_es['cost']))[0])
				if len(best_edges) != 1:
					winner = random.sample(best_edges, 1)[0]
				else:
					winner = best_edges[0]
				end_playlist.append(winner.target)
				back_idx = winner.target
			except (IndexError, KeyError):
				print "ran into an Index Attribute error on front edge add.  Was going to add {0} edges".format(new_edges)
		full_path = G.get_shortest_paths(start_idx)[end_idx]
		if full_path == []:
			# not connected yet so paste.
			if distance in ["social_only", "tfidf", "timbre_sim"]:
				full_path = front_playlist +list(reversed(end_playlist))
			else:
				raise NotImplementedError("don't know how to make a proper path with this metric")
		if fmt=="json":
			#returns the playlist as a json list of dicts, for ajaxy fun and data getting
			output = {"status":'ok',
				"start_song":G.vs[start_idx]['track_id'],
				"end_song":G.vs[end_idx]['track_id'],
				"method":distance,
				"playlist":[]}
			for idx, a_track in enumerate(full_path):
				output['playlist'].append({'position':idx,
					'title':G.vs[a_track]['title'],
					'track_id':G.vs[a_track]['track_id'],
					'perm_url':G.vs[a_track]['perm_url'],
					'artwork_url':G.vs[a_track]['artwork_url'],
					'streaming_url': G.vs[a_track]['streaming'],
					'artist_name': G.vs[a_track]['artist'],
					'artist_url': G.vs[a_track]['artist_url'],
					'artist_id': G.vs[a_track]['artist_id']})
			return dumps(output)
						
		out= """<head>
	<title>Roomba Recon :: Finding a path through SoundCloud's Jukebox</title>
</head>
<body>
	<script type="text/javascript" src="http://mediaplayer.yahoo.com/js"></script>
	<div style="width:800px;margin:25px auto;font-family:helvetica;">
	<img src="../images/roomba.jpg" style="text-align:center" alt="Roomba Recon banner"\>
	<h3 style="text-align:center">Finding a path through SoundCloud's Jukebox</h3>
	<br><br>
	<h3>playlist from {0} to {1}:</h3>
	<ol>{2}</ol>
	<br><br>
	(<a href="../playlist" style="font-size:small">back</a>)
	</div>
</body>
		"""
		listbits = ""
		for song in full_path:
			listbits += "<li><a href=\"{0}\">{1}</a><a href=\"{2}\" alt='for the player' type=\"audio/mpeg\" title=\"{3}\"><img src=\"{4}\" alt=\"album art\" style=\"display:none\" /></a></li>\n".format(
				G.vs[song]['perm_url'],
				G.vs[song]['title']+' - '+G.vs[song]['artist'],
				G.vs[song]['streaming'],
				G.vs[song]['title'],
				G.vs[song]['artwork_url'])
		return out.format(start_track.title+' - '+start_track.user.username, end_track.title+' - '+end_track.user.username, listbits)
	playlist.exposed = True
	
	def default(self, track_a, track_b):
		"""
		takes in two track ids from sc and grabs some analyze from EN.  Uses this to make a similarity assertion between the two tracks.
		Note, doing everything in approximately the stupidist way possible, so ymmv.
		
		"""
		root = init_scope()
		a_track = root.tracks(int(track_a))
		b_track = root.tracks(int(track_b))
		a_bits = track.track_from_url(a_track.stream_url)
		b_bits = track.track_from_url(b_track.stream_url)

		a_timbre = np.zeros((len(a_bits.segments), 12))
		for idx, seg in enumerate(a_bits.segments):
			a_timbre[idx] = seg['timbre']
		a_stack = np.hstack((a_timbre.mean(axis=0),a_timbre.std(axis=0)))

		b_timbre = np.zeros((len(b_bits.segments), 12))
		for idx, seg in enumerate(b_bits.segments):
			b_timbre[idx] = seg['timbre']
		b_stack = np.hstack((b_timbre.mean(axis=0),b_timbre.std(axis=0)))
		euc = spatial.distance.euclidean(a_stack, b_stack)
		cos = spatial.distance.cosine(a_stack, b_stack)
		man = spatial.distance.cityblock(a_stack, b_stack)#as in manhatten distance
		return """
		timbral distance between <a href='{0}'>{1}</a> and <a href='{2}'>{3}</a> in various ways
		<ul>
		<li> Euclidean distance: {4}</li>
		<li> Cosine distance: {5}</li>
		<li> City block distance {6}</li>
		</ul>
		(<a href="../">back</a>)
		""".format(
			a_track.permalink_url,
			"'"+a_track.title+"' by "+a_track.user.username, 
			b_track.permalink_url,
			"'"+b_track.title+"' by "+b_track.user.username, 
			euc,
			cos,
			man)
	default.exposed = True
	
cherrypy.tree.mount(Recon())


if __name__ == '__main__':
	import os.path
	thisdir = os.path.dirname(__file__)
	current_dir = os.path.dirname(os.path.abspath(__file__))
    # Set up site-wide config first so we get a log if errors occur.
	cherrypy.config.update({#'environment': 'development',
							'log.error_file': 'site.log',
							'log.screen': True})
        
	conf = {'global': {'server.socket_host': "127.0.0.1",
						'server.socket_port': 9000,
						'server.thread_pool': 10},
			'/': {'tools.caching.on': True},
			'/js': {'tools.staticdir.on': True,
				'tools.staticdir.dir': os.path.join(current_dir, 'js'),
				'tools.staticdir.content_types': {'js': 'text/javascript'}},
			'/css': {'tools.staticdir.on': True,
					'tools.staticdir.dir': os.path.join(current_dir, 'css'),
					'tools.staticdir.content_types': {'css': 'text/css'}},
			'/images':{'tools.staticdir.on': True,
					'tools.staticdir.dir': os.path.join(current_dir, 'images'),
					'tools.staticdir.content_types': {'jpg': 'image/jpeg', 'png':'image/png', 'gif':'image/gif'}},
			'/pages':{'tools.staticdir.on': True,
					'tools.staticdir.dir': os.path.join(current_dir, 'pages'),
					'tools.staticdir.content_types': {'html': 'text/html;charset: utf-8'}
					}}
					
	cherrypy.quickstart(Recon(), '/', config=conf)
	# cherrypy.quickstart(config=os.path.join(thisdir, 'recon.conf'))