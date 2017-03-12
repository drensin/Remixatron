""" Classes for remixing audio files.

(c) 2017 - Dave Rensin - dave@rensin.com 

This module contains classes for remixing audio files. It started
as an attempt to re-create the amazing Infinite Jukebox (http://www.infinitejuke.com)
created by Paul Lamere of Echo Nest.

The InfiniteJukebox class can do it's processing in a background thread and 
reports progress via the progress_callback arg. To run in a thread, pass async=True
to the constructor. In that case, it exposes an Event named play_ready -- which will
be signaled when the processing is complete. The default mode is to run synchronously.

  Async example:

      def MyCallback(percentage_complete_as_float, string_message):
        print "I am now %f percent complete with message: %s" % (percentage_complete_as_float * 100, string_message)

      jukebox = InfiniteJukebox(filename='some_file.mp3', progress_callback=MyCallback, async=True)
      jukebox.play_ready.wait()

      <some work here...>
  
  Non-async example:

      def MyCallback(percentage_complete_as_float, string_message):
        print "I am now %f percent complete with message: %s" % (percentage_complete_as_float * 100, string_message)

      jukebox = InfiniteJukebox(filename='some_file.mp3', progress_callback=MyCallback, async=False)

      <blocks until completion... some work here...>
  
"""

import collections
import librosa
import math
import pygame
import random
import scipy
import sys
import threading

from operator import itemgetter, attrgetter
from pygame import mixer

import numpy as np
import sklearn.cluster

class InfiniteJukebox(object):
    
    """ Class to "infinitely" remix a song.

    This class will take an audio file (wav, mp3, ogg, etc) and
    (a) decompose it into individual beats, (b) find the tempo
    of the track, and (c) create a play path that you can use
    to play the song approx infinitely.
    
    The idea is that it will find and cluster beats that are
    musically similar and return them to you so you can automatically
    'remix' the song.
    
    Attributes:
    
     play_ready: an Event that triggers when the processing/clustering is complete and
                 playback can begin. This is only defined if you pass async=True in the 
                 constructor.
    
       duration: the duration (in seconds) of the track after the leading and trailing silences
                 have been removed.
              
      raw_audio: an array of numpy.Int16 that is suitable for using for playback via pygame
                 or similar modules. If the audio is mono then the shape of the array will
                 be (bytes,). If it's stereo, then the shape will be (2,bytes).
               
    sample_rate: the sample rate from the audio file. Usually 44100
    
       clusters: the number of clusters used to group the beats. If you pass in a value, then 
                 this will be reflected here. If you let the algorithm decide, then auto-generated
                 value will be reflected here.
    
          beats: a dictionary containing the individual beats of the song in normal order. Each
                 beat will have the following keys:
               
                         id: the ordinal position of the beat in the song
                      start: the time (in seconds) in the song where this beat occurs
                   duration: the duration (in seconds) of the beat
                     buffer: an array of audio bytes for this beat. it is just raw_audio[start:start+duration]
                    cluster: the cluster that this beat most closely belongs. Beats in the same cluster
                             have similar harmonic (timbre) and chromatic (pitch) characteristics. They
                             will "sound similar"
                    segment: the segment to which this beat belongs. A 'segment' is a contiguous block of
                             beats that belong to the same cluster.
                  amplitude: the loudness of the beat
                       next: the next beat to play after this one, if playing sequentially
            jump_candidates: a list of the other beats in the song to which it is reasonable to jump. Those beats
                             (a) are in the same cluster as the NEXT oridnal beat, (b) are of the same segment position
                             as the next ordinal beat, (c) are in the same place in the measure as the NEXT beat, 
                             (d) but AREN'T the next beat.
              
                 An example of playing the first 32 beats of a song:

                    from Remixatron import InfiniteJukebox
                    from pygame import mixer
                    import time

                    jukebox = InfiniteJukebox('some_file.mp3')

                    pygame.mixer.init(frequency=jukebox.sample_rate)
                    channel = pygame.mixer.Channel(0)

                    for beat in jukebox.beats[0:32]:
                        snd = pygame.Sound(buffer=beat['buffer'])
                        channel.queue(snd)
                        time.sleep(beat['duration'])
                    
    play_vector: a beat play list of 1024^2 items. This represents a pre-computed
                 remix of this song that will last beat['duration'] * 1024 * 1024
                 seconds long. A song that is 120bpm will have a beat duration of .5 sec,
                 so this playlist will last .5 * 1024 * 1024 seconds -- or 145.67 hours.
                 
                 Each item contains:
    
                    beat: an index into the beats array of the beat to play
                 seq_len: the length of the musical sequence being played
                          in this part of play_vector.
                 seq_pos: this beat's position in seq_len. When
                          seq_len - seq_pos == 0 the song will "jump"
    
    """

    def __init__(self, filename, start_beat=1, clusters=0, progress_callback=None, async=False):
        
        """ The constructor for the class. Also starts the processing thread.
        
            Args:
            
                filename: the path to the audio file to process
              start_beat: the first beat to play in the file. Should almost always be 1,
                          but you can override it to skip into a specific part of the song.
                clusters: the number of similarity clusters to compute. The DEFAULT value 
                          of 0 means that the code will try to automatically find an optimal 
                          cluster. If you specify your own value, it MUST be non-negative. Lower
                          values will create more promiscuous jumps. Larger values will create higher quality
                          matches, but run the risk of jumps->0 -- which will just loop the 
                          audio sequentially ~forever. 
       progress_callback: a callback function that will get periodic satatus updates as
                          the audio file is processed. MUST be a function that takes 2 args:
            
                             percent_complete: FLOAT between 0.0 and 1.0
                                      message: STRING with the progress message
        """
        self.__progress_callback = progress_callback
        self.__filename = filename
        self.__start_beat = start_beat        
        self.clusters = clusters
        
        if async == True:
            self.play_ready = threading.Event()
            self.__thread = threading.Thread(target=self.__process_audio)
            self.__thread.start()
        else:
            self.play_ready = None
            self.__process_audio()
            
    def __process_audio(self):
    
        """ The main audio processing routine for the thread.
        
        This routine uses Laplacian Segmentation to find and 
        group similar beats in the song. 
        
        This code has been adapted from the sample created by Brian McFee at
        https://librosa.github.io/librosa_gallery/auto_examples/plot_segmentation.html#sphx-glr-auto-examples-plot-segmentation-py 
        and is based on his 2014 paper published at http://bmcfee.github.io/papers/ismir2014_spectral.pdf
        
        I have made some performance improvements, but the basic parts remain (mostly) unchanged
        """
        
        self.__report_progress( .1, "loading file and extracting raw audio")
        
        #
        # load the file as stereo with a high sample rate and 
        # trim the silences from each end
        #
        
        y, sr = librosa.core.load(self.__filename, mono=False, sr=44100)
        y, index = librosa.effects.trim(y)

        self.duration = librosa.core.get_duration(y,sr)
        self.raw_audio = (y * np.iinfo(np.int16).max).astype(np.int16).T.copy(order='C')
        self.sample_rate = sr
        
        # after the raw audio bytes are saved, convert the samples to mono
        # because the beat detection algorithm in librosa requires it.
        
        y = librosa.core.to_mono(y)
    
        self.__report_progress( .2, "computing pitch data..." )
        
        # Compute the constant-q chromagram for the samples. 
        
        BINS_PER_OCTAVE = 12 * 3
        N_OCTAVES = 7

        cqt = librosa.cqt(y=y, sr=sr, bins_per_octave=BINS_PER_OCTAVE, n_bins=N_OCTAVES * BINS_PER_OCTAVE)
        C = librosa.amplitude_to_db( cqt, ref=np.max)
        
        self.__report_progress( .3, "Finding beats..." )
        
        ##########################################################
        # To reduce dimensionality, we'll beat-synchronous the CQT
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
        Csync = librosa.util.sync(C, beats, aggregate=np.median)

        self.tempo = tempo
        
        # For alignment purposes, we'll need the timing of the beats
        # we fix_frames to include non-beat frames 0 and C.shape[1] (final frame)
        beat_times = librosa.frames_to_time(librosa.util.fix_frames(beats,
                                                                    x_min=0,
                                                                    x_max=C.shape[1]),
                                            sr=sr)

        self.__report_progress( .4, "building recurrence matrix..." )
        #####################################################################
        # Let's build a weighted recurrence matrix using beat-synchronous CQT
        # (Equation 1)
        # width=3 prevents links within the same bar
        # mode='affinity' here implements S_rep (after Eq. 8)
        R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity',
                                              sym=True)

        # Enhance diagonals with a median filter (Equation 2)
        df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
        Rf = df(R, size=(1, 7))


        ###################################################################
        # Now let's build the sequence matrix (S_loc) using mfcc-similarity
        #
        #   :math:`R_\text{path}[i, i\pm 1] = \exp(-\|C_i - C_{i\pm 1}\|^2 / \sigma^2)`
        #
        # Here, we take :math:`\sigma` to be the median distance between successive beats.
        #
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        Msync = librosa.util.sync(mfcc, beats)

        path_distance = np.sum(np.diff(Msync, axis=1)**2, axis=0)
        sigma = np.median(path_distance)
        path_sim = np.exp(-path_distance / sigma)

        R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)


        ##########################################################
        # And compute the balanced combination (Equations 6, 7, 9)

        deg_path = np.sum(R_path, axis=1)
        deg_rec = np.sum(Rf, axis=1)

        mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)

        A = mu * Rf + (1 - mu) * R_path

        #####################################################
        # Now let's compute the normalized Laplacian (Eq. 10)
        L = scipy.sparse.csgraph.laplacian(A, normed=True)


        # and its spectral decomposition
        evals, evecs = scipy.linalg.eigh(L)


        # We can clean this up further with a median filter.
        # This can help smooth over small discontinuities
        evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))


        # cumulative normalization is needed for symmetric normalize laplacian eigenvectors
        Cnorm = np.cumsum(evecs**2, axis=1)**0.5

        # If we want k clusters, use the first k normalized eigenvectors.
        # Fun exercise: see how the segmentation changes as you vary k

        self.__report_progress( .5, "clustering..." )

        if self.clusters == 0:
            self.clusters, seg_ids = self.__compute_best_cluster(evecs, Cnorm)
    
        else:
            k = self.clusters

            X = evecs[:, :k] / Cnorm[:, k-1:k]

            #############################################################
            # Let's use these k components to cluster beats into segments
            # (Algorithm 1)
            KM = sklearn.cluster.KMeans(n_clusters=k)

            seg_ids = KM.fit_predict(X)

        self.__report_progress( .51, "using %d clusters" % self.clusters )
            
        ###############################################################
        # Locate segment boundaries from the label sequence
        bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])

        # Count beat 0 as a boundary
        bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)

        # Compute the segment label for each boundary
        bound_segs = list(seg_ids[bound_beats])

        # Convert beat indices to frames
        bound_frames = beats[bound_beats]

        # Make sure we cover to the end of the track
        bound_frames = librosa.util.fix_frames(bound_frames,
                                               x_min=None,
                                               x_max=C.shape[1]-1)

        bound_times = librosa.frames_to_time(bound_frames)

        # Get the amplitudes and beat-align them
        self.__report_progress( .6, "getting amplitudes" )
        amplitudes = librosa.feature.rmse(y=y)
        ampSync = librosa.util.sync(amplitudes, beats)

        # create a list of tuples that include the ordinal position, the start time of the beat,
        # the cluster to which the beat belongs and the mean amplitude of the beat
        
        beat_tuples = zip(range(0,len(beats)), beat_times, seg_ids, ampSync[0].tolist())

        info = []

        bytes_per_second = int(round(len(self.raw_audio) / self.duration))
        
        last_cluster = -1
        current_segment = -1
        segment_beat = 0
        
        for i in range(0, len(beat_tuples)):
            final_beat = {}
            final_beat['start'] = float(beat_tuples[i][1])
            final_beat['cluster'] = int(beat_tuples[i][2])
            final_beat['amplitude'] = float(beat_tuples[i][3])
            
            if final_beat['cluster'] != last_cluster:
                current_segment += 1
                segment_beat = 0
            else:
                segment_beat += 1
                
            final_beat['segment'] = current_segment
            final_beat['is'] = segment_beat
            
            last_cluster = final_beat['cluster']
            
            if i == len(beat_tuples) - 1:
                final_beat['duration'] = self.duration - final_beat['start']
            else:
                final_beat['duration'] = beat_tuples[i+1][1] - beat_tuples[i][1]

            if ( (final_beat['start'] * bytes_per_second) % 2 > 1.5 ):
                final_beat['start_index'] = int(math.ceil(final_beat['start'] * bytes_per_second))
            else:
                final_beat['start_index'] = int(final_beat['start'] * bytes_per_second)

            final_beat['stop_index'] = int(math.ceil((final_beat['start'] + final_beat['duration']) * bytes_per_second))

            # save pointers to the raw bytes for each beat with each beat.
            final_beat['buffer'] = self.raw_audio[ final_beat['start_index'] : final_beat['stop_index'] ]
            
            info.append(final_beat)

        self.__report_progress( .7, "truncating to fade point..." )
        
        # get the mean amplitude of the beats
        avg_amplitude = np.mean([b['amplitude'] for b in info])
        
        # assume that the fade point of the song is the beat that (a) is after 90% of the song and (b) has
        # an amplitude of <= 70% of the mean. For songs that have 'button' endings, just return the last
        # beat
        fade = next( (b for b in info[int(len(info) * .9):] if b['amplitude'] <= (.7 * avg_amplitude)), info[-1] )

        # truncate the beats to [start:fade]
        beats = info[self.__start_beat:info.index(fade)]
        self.fade = info[info.index(fade):]
        
        # truncate the beats so that they are a multiple of 4. The vast majority of songs will
        # have 4 beats per measure and doing this will make looping from the end of the song
        # into some other place sound more natural
        
        if ( fade != info[-1] and len(beats) % 4 != 0 ):
            beats = beats[:(len(beats) % 4) * -1]

        # nearly all songs have an intro that should be discarded during the jump calculations because
        # landing there will sound stilted. This line finds the first beat of the 2nd cluster in the song
        loop_bounds_begin = beats.index(next(b for b in beats if b['cluster'] != beats[0]['cluster']))
        
        # if start_beat has been passed in, then use the max(start_beat, loop_bounds_begin) as the earliest
        # allowed jump point in the song
        loop_bounds_begin = max(loop_bounds_begin, self.__start_beat)

        self.__report_progress( .8, "computing final beat array..." )

        # assign final beat ids        
        for beat in beats:
            beat['id'] = beats.index(beat)

        # compute a coherent 'next' beat to play. This is always just the next ordinal beat
        # unless we're at the end of the song. Then it gets a little trickier.
        
        for beat in beats:
            if beat == beats[-1]:
                
                # if we're at the last beat, then we want to find a reasonable 'next' beat to play. It should (a) share the
                # same cluster, (b) be not closer than 64 beats to the current beat, and (c) be after the computed loop_bounds_begin.
                # If we can't find such an animal, then just return the beat at loop_bounds_begin
                
                beat['next'] = next( (b['id'] for b in beats if b['cluster'] == beat['cluster'] and 
                                      b['id'] <= (beat['id'] - 64) and 
                                      b['id'] >= loop_bounds_begin), loop_bounds_begin )
            else:
                beat['next'] = beats.index(beat) + 1

            # find all the beats that (a) are in the same cluster as the NEXT oridnal beat, (b) are of the same
            # cluster position as the next ordinal beat, (c) are in the same place in the measure as the NEXT beat,
            # (d) but AREN'T the next beat.
            #
            # THAT collection of beats contains our jump candidates

            jump_candidates = [bx['id'] for bx in beats[loop_bounds_begin:] if 
                               (bx['cluster'] == beats[beat['next']]['cluster']) and 
                               (bx['is'] == beats[beat['next']]['is']) and 
                               (bx['id'] % 4 == beats[beat['next']]['id'] % 4) and
                               (bx['id'] != beat['next']) and
                               (abs(bx['id'] - beat['id']) >= 16)]
            
            beat['jump_candidates'] = jump_candidates

        #
        # This section of the code computes the play_vector -- a 1024*1024 beat length
        # remix of the current song.
        #
        
        random.seed()
        min_sequence = random.randrange(32,49,8)
        current_sequence = 0
        beat = beats[0]
        
        self.__report_progress( .9, "creating play vector" )

        play_vector = []

        play_vector.append( {'beat':0, 'seq_len':min_sequence, 'seq_pos':current_sequence} )

        self.segments = max([b['segment'] for b in beats])
        
        # we want to keep a list of recent jump segments so we don't accidentally wind up in a local loop
        #
        # the number of segments in a song will vary so we want to set the number of recents to keep 
        # at 10% of the total number of segments. Eg: if there are 34 segments, then the depth will
        # be set at 3.
        #
        # On the off chance that the # of segments < 10 we set a floor queue depth of 1
            
        recent_depth = max( int(self.segments * .1), 1 )
        recent = collections.deque(maxlen=recent_depth)

        for i in range(0, 1024 * 1024):

            current_sequence += 1

            will_jump = current_sequence >= min_sequence

            # if it's time to jump, then assign the next beat, and create
            # a new play sequence between 8 and 32 beats -- making sure
            # that the new sequence is always modulo 4.

            if ( will_jump ):

                # randomly pick from the beat jump candidates that aren't in recently jumped segments
                non_recent_candidates = [c for c in beat['jump_candidates'] if beats[c]['segment'] not in recent]

                # if there aren't any good jump candidates then just target the next ordinal beat. This is
                # a failsafe that in practice should very rarely be needed. Otherwise, just pick a random beat from
                # the candidates

                if len(non_recent_candidates) == 0:
                    beat = beats[ beat['next'] ]
                else:
                    beat = beats[ random.choice(non_recent_candidates) ]
                    recent.append(beat['segment'])

                current_sequence = 0
                min_sequence = random.randrange(8,33,4)

                play_vector.append({'beat':beat['id'], 'seq_len': min_sequence, 'seq_pos': current_sequence})
            else:                    
                play_vector.append({'beat':beat['next'], 'seq_len': min_sequence, 'seq_pos': current_sequence})
                beat = beats[beat['next']]

        self.__report_progress(1.0, "ready")

        self.beats = beats
        self.play_vector = play_vector
        
        if self.play_ready:
            self.play_ready.set()
            
    def __report_progress(self, pct_done, message):
        
        """ If a reporting callback was passed, call it in oder
            to mark progress.
        """
        if self.__progress_callback:
            self.__progress_callback( pct_done, message )
            
    def __compute_best_cluster(self, evecs, Cnorm):
        
        ''' Attempts to compute optimum clustering
        
            [TODO] Implelent a proper RMSE-based algorithm. This is kind
            of a hack right now..
            
            PARAMETERS:
                evecs: Eigen-vectors computed from the clusteration algorithm
                Cnorm: Cumulative normalization of evecs. Easier to pass it in than
                       compute it from scratch here.
                
            KEY DEFINITIONS:
                
                 Cluster: buckets of musical similarity
                Segments: contiguous blocks of beats belonging to the same cluster
                 Orphans: clusters that only belong to one cluster
                    Stub: a cluster with less than N beats. Stubs are a sign of 
                          overfitting
                 
            SUMMARY:
                
                Group the beats in [2..48] clusters. Compute the average number of
                orphans in each of the 47 computed clusterings. We want to find the
                right cluster size (2..48) that will give us the highest possiblity 
                of smoothly jumping around but without being too promiscuous. The way 
                we do that is to find the highest cluster value (2..48) that has an 
                average orphan count <= the global average orphan count AND has no stub clusters.
                
                Basically, we're looking for the highest possible cluster # that doesn't 
                obviously overfit.
                
                Someday I'll implement a proper RMSE algorithm...
        '''
        
        self._clusters_list = []
        
        for ki in range(2,49):

            # compute a matrix of the Eigen-vectors / their normalized values
            X = evecs[:, :ki] / Cnorm[:, ki-1:ki]

            # cluster with candidate ki
            labels = sklearn.cluster.KMeans(n_clusters=ki, max_iter=1000, n_init=10).fit_predict(X)
            
            entry = {'clusters':ki, 'labels':labels}
            
            # create an array of dictionary entries containing (a) the cluster label, 
            # (b) the number of total beats that belong to that cluster, and 
            # (c) the number of clusters in which that cluster appears.
            
            lst = []
            
            for i in range(0,ki):
                lst.append( {'label':i, 'beats':0, 'segs':0} )

            last_label = -1
            
            for l in labels:
                            
                if l != last_label:
                    lst[l]['segs'] += 1
                    last_label = l
                    
                lst[l]['beats'] += 1
            
            entry['cluster_map'] = lst
            
            # get a list of clusters that only appear in 1 cluster. Those are orphans.
            entry['orphans'] = [l['label'] for l in entry['cluster_map'] if l['segs'] == 1]
            
            # across all the clusters, get the avg number of orphans per cluster
            # ie: the % of clusters that appear in only 1 cluster
            entry['avg_orphans'] = len(entry['orphans']) / float(entry['clusters'])
            
            # get the list of clusters that have less than 6 beats. Those are stubs
            entry['stubs'] = len( [l for l in entry['cluster_map'] if l['beats'] < 6] )

            self._clusters_list.append(entry)

        # compute the average number of orphan clusters across all candidate clusterings
        avg_orphan_ratio = sum([cl['avg_orphans'] for cl in self._clusters_list]) / len(self._clusters_list)
        
        # find the candidates that have an average orphan count <= the global average AND have no stubs
        candidates = [cl['clusters'] for cl in self._clusters_list if cl['avg_orphans'] <= avg_orphan_ratio and cl['stubs'] == 0]
        
        # the winner is the highest cluster size among the candidates
        final_cluster_size = max(candidates)
        
        # return a tuple of (winning cluster size, [array of cluster labels for the beats])
        return (final_cluster_size, next(c['labels'] for c in self._clusters_list if c['clusters'] == final_cluster_size))
        
        