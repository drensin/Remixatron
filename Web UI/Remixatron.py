""" Classes for remixing audio files.
(c) 2017 - Dave Rensin - dave@rensin.com

This module contains classes for remixing audio files. It started
as an attempt to re-create the amazing Infinite Jukebox (http://www.infinitejuke.com)
created by Paul Lamere of Echo Nest.

The InfiniteJukebox class can do it's processing in a background thread and
reports progress via the progress_callback arg. To run in a thread, pass do_async=True
to the constructor. In that case, it exposes an Event named play_ready -- which will
be signaled when the processing is complete. The default mode is to run synchronously.

  Async example:

      def MyCallback(percentage_complete_as_float, string_message):
        print "I am now %f percent complete with message: %s" % (percentage_complete_as_float * 100, string_message)

      jukebox = InfiniteJukebox(filename='some_file.mp3', progress_callback=MyCallback, do_async=True)
      jukebox.play_ready.wait()

      <some work here...>

  Non-async example:

      def MyCallback(percentage_complete_as_float, string_message):
        print "I am now %f percent complete with message: %s" % (percentage_complete_as_float * 100, string_message)

      jukebox = InfiniteJukebox(filename='some_file.mp3', progress_callback=MyCallback, do_async=False)

      <blocks until completion... some work here...>

"""

import collections
import librosa
import madmom
import random
import scipy
import threading

import numpy as np
import sklearn.cluster
import sklearn.metrics

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
                 playback can begin. This is only defined if you pass do_async=True in the
                 constructor.

       duration: the duration (in seconds) of the track after the leading and trailing silences
                 have been removed.

      raw_audio: an array of numpy.Int16 that is suitable for using for playback via pygame
                 or similar modules. If the audio is mono then the shape of the array will
                 be (bytes,). If it's stereo, then the shape will be (2,bytes).

    sample_rate: the sample rate from the audio file. Usually 44100 or 48000

       clusters: the number of clusters used to group the beats. If you pass in a value, then
                 this will be reflected here. If you let the algorithm decide, then auto-generated
                 value will be reflected here.

          beats: a dictionary containing the individual beats of the song in normal order. Each
                 beat will have the following keys:

                         id: the ordinal position of the beat in the song
                      start: the time (in seconds) in the song where this beat occurs
                   duration: the duration (in seconds) of the beat
               bar_position: where in the musical bar this beat lies
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

    def __init__(self, filename, start_beat=0, clusters=0, progress_callback=None,
                 do_async=False, use_v1=False, starting_beat_cache=None):

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
                  use_v1: set to True if you want to use the original auto clustering algorithm.
                          Otherwise, it will use the newer silhouette-based scheme.
     starting_beat_cache: the process to pick out the beats in the audio is very compute
                          intense. You can shortcut it by passing in an already populated beat
                          dictionary in the form of self.beats
        """
        self.__progress_callback = progress_callback
        self.__filename = filename
        self.__start_beat = start_beat
        self.clusters = clusters
        self._extra_diag = ""
        self._use_v1 = use_v1
        self._starting_beat_cache = starting_beat_cache

        if do_async == True:
            self.play_ready = threading.Event()
            self.__thread = threading.Thread(target=self.__process_audio_madmom)
            self.__thread.start()
        else:
            self.play_ready = None
            self.__process_audio_madmom()

    #############################################
    # NOTE: This code is dead and probably will
    # be deleted pretty soon. Please proceed to 
    # __process_audio_madmom() for the most
    # current work
    ############################################

    # def __process_audio(self):

    #     """ The main audio processing routine for the thread.

    #     This routine uses Laplacian Segmentation to find and
    #     group similar beats in the song.

    #     This code has been adapted from the sample created by Brian McFee at
    #     https://librosa.github.io/librosa_gallery/auto_examples/plot_segmentation.html#sphx-glr-auto-examples-plot-segmentation-py
    #     and is based on his 2014 paper published at http://bmcfee.github.io/papers/ismir2014_spectral.pdf

    #     I have made some performance improvements, but the basic parts remain (mostly) unchanged
    #     """

    #     self.__report_progress( .1, "loading file and extracting raw audio")

    #     #
    #     # load the file as stereo with a high sample rate and
    #     # trim the silences from each end
    #     #

    #     y, sr = librosa.core.load(self.__filename, mono=False, sr=None)
    #     # y, _ = librosa.effects.trim(y)

    #     self.duration = librosa.core.get_duration(y,sr)
    #     self.raw_audio = (y * np.iinfo(np.int16).max).astype(np.int16).T.copy(order='C')
    #     self.sample_rate = sr

    #     # after the raw audio bytes are saved, convert the samples to mono
    #     # because the beat detection algorithm in librosa requires it.

    #     y = librosa.core.to_mono(y)

    #     self.__report_progress( .2, "computing pitch data..." )

    #     # Compute the constant-q chromagram for the samples.

    #     BINS_PER_OCTAVE = 12 * 3
    #     N_OCTAVES = 7

    #     cqt = librosa.cqt(y=y, sr=sr, bins_per_octave=BINS_PER_OCTAVE, n_bins=N_OCTAVES * BINS_PER_OCTAVE)
    #     C = librosa.amplitude_to_db( np.abs(cqt), ref=np.max)

    #     self.__report_progress( .3, "Finding beats..." )

    #     ##########################################################
    #     # To reduce dimensionality, we'll beat-synchronous the CQT
    #     # tempo, btz = librosa.beat.beat_track(y=y, sr=sr, trim=False)

    #     # onset_env = librosa.onset.onset_strength(y=y, sr=sr,
    #     #                                  aggregate=np.median)

    #     # tempo, btz = librosa.beat.beat_track( onset_envelope=onset_env, sr=sr)        

    #     self.__report_progress( .33, "Separating precussion from harmonics..." )
    #     D = librosa.stft(y)
    #     _, P = librosa.decompose.hpss(D, margin=(1.0,5.0))

    #     y_perc = librosa.istft(P)

    #     self.__report_progress( .35, "Finding beats on just the percussive bits..." )
    #     tempo, btz = librosa.beat.beat_track(y=y_perc, sr=sr, trim=False)
        
    #     Csync = librosa.util.sync(C, btz, aggregate=np.median)

    #     self.tempo = tempo

    #     # For alignment purposes, we'll need the timing of the beats
    #     # we fix_frames to include non-beat frames 0 and C.shape[1] (final frame)
    #     beat_times = librosa.frames_to_time(librosa.util.fix_frames(btz,
    #                                                                 x_min=0,
    #                                                                 x_max=C.shape[1]),
    #                                                                 sr=sr)

    #     self.__report_progress( .4, "building recurrence matrix..." )
    #     #####################################################################
    #     # Let's build a weighted recurrence matrix using beat-synchronous CQT
    #     # (Equation 1)
    #     # width=3 prevents links within the same bar
    #     # mode='affinity' here implements S_rep (after Eq. 8)
    #     R = librosa.segment.recurrence_matrix(Csync, width=3, mode='affinity',
    #                                           sym=True)

    #     # Enhance diagonals with a median filter (Equation 2)
    #     df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    #     Rf = df(R, size=(1, 7))


    #     ###################################################################
    #     # Now let's build the sequence matrix (S_loc) using mfcc-similarity
    #     #
    #     #   :math:`R_\text{path}[i, i\pm 1] = \exp(-\|C_i - C_{i\pm 1}\|^2 / \sigma^2)`
    #     #
    #     # Here, we take :math:`\sigma` to be the median distance between successive beats.
    #     #
    #     mfcc = librosa.feature.mfcc(y=y, sr=sr)
    #     Msync = librosa.util.sync(mfcc, btz)

    #     path_distance = np.sum(np.diff(Msync, axis=1)**2, axis=0)
    #     sigma = np.median(path_distance)
    #     path_sim = np.exp(-path_distance / sigma)

    #     R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)


    #     ##########################################################
    #     # And compute the balanced combination (Equations 6, 7, 9)

    #     deg_path = np.sum(R_path, axis=1)
    #     deg_rec = np.sum(Rf, axis=1)

    #     mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)

    #     A = mu * Rf + (1 - mu) * R_path

    #     #####################################################
    #     # Now let's compute the normalized Laplacian (Eq. 10)
    #     L = scipy.sparse.csgraph.laplacian(A, normed=True)


    #     # and its spectral decomposition
    #     _, evecs = scipy.linalg.eigh(L)


    #     # We can clean this up further with a median filter.
    #     # This can help smooth over small discontinuities
    #     evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))


    #     # cumulative normalization is needed for symmetric normalize laplacian eigenvectors
    #     Cnorm = np.cumsum(evecs**2, axis=1)**0.5

    #     # If we want k clusters, use the first k normalized eigenvectors.
    #     # Fun exercise: see how the segmentation changes as you vary k

    #     self.__report_progress( .5, "clustering..." )

    #     # if a value for clusters wasn't passed in, then we need to auto-cluster

    #     if self.clusters == 0:

    #         # if we've been asked to use the original auto clustering alogrithm, otherwise
    #         # use the new and improved one that accounts for silhouette scores.

    #         if self._use_v1:
    #             self.clusters, seg_ids = self.__compute_best_cluster(evecs, Cnorm)
    #         else:
    #             self.clusters, seg_ids = self.__compute_best_cluster_with_sil(evecs, Cnorm)

    #     else: # otherwise, just use the cluster value passed in
    #         k = self.clusters

    #         self.__report_progress( .51, "using " + str(self.clusters) + " clusters..." )

    #         X = evecs[:, :k] / Cnorm[:, k-1:k]

    #         # seg_ids = sklearn.cluster.KMeans(n_clusters=k, max_iter=300,
    #         #                                    random_state=0, n_init=20).fit_predict(X)

    #         seg_ids = sklearn.cluster.MiniBatchKMeans(n_clusters=k).fit_predict(X)

    #     # Get the amplitudes and beat-align them
    #     self.__report_progress( .93, "getting amplitudes" )

    #     # newer versions of librosa have renamed the rmse function

    #     if hasattr(librosa.feature,'rms'):
    #         amplitudes = librosa.feature.rms(y=y)
    #     else:
    #         amplitudes = librosa.feature.rmse(y=y)

    #     ampSync = librosa.util.sync(amplitudes, btz)

    #     # create a list of tuples that include the ordinal position, the start time of the beat,
    #     # the cluster to which the beat belongs and the mean amplitude of the beat

    #     zbeat_tuples = zip(range(0,len(btz)), beat_times, seg_ids, ampSync[0].tolist())
    #     beat_tuples =tuple(zbeat_tuples)

    #     info = []

    #     # bytes_per_second = int(round(len(self.raw_audio) / self.duration))
    #     bytes_per_second = len(self.raw_audio) / self.duration

    #     last_cluster = -1
    #     current_segment = -1
    #     segment_beat = 0

    #     for i in range(0, len(beat_tuples)):
    #         final_beat = {}
    #         final_beat['start'] = float(beat_tuples[i][1])
    #         final_beat['cluster'] = int(beat_tuples[i][2])
    #         final_beat['amplitude'] = float(beat_tuples[i][3])

    #         if final_beat['cluster'] != last_cluster:
    #             current_segment += 1
    #             segment_beat = 0
    #         else:
    #             segment_beat += 1

    #         final_beat['segment'] = current_segment
    #         final_beat['is'] = segment_beat

    #         last_cluster = final_beat['cluster']

    #         if i == len(beat_tuples) - 1:
    #             final_beat['duration'] = self.duration - final_beat['start']
    #         else:
    #             final_beat['duration'] = beat_tuples[i+1][1] - beat_tuples[i][1]

    #         # if ( (final_beat['start'] * bytes_per_second) % 2 > 1.5 ):
    #         #     final_beat['start_index'] = int(math.ceil(final_beat['start'] * bytes_per_second))
    #         # else:
    #         #     final_beat['start_index'] = int(final_beat['start'] * bytes_per_second)

    #         final_beat['start_index'] = int(final_beat['start'] * bytes_per_second)
    #         # final_beat['stop_index'] = int(math.ceil((final_beat['start'] + final_beat['duration']) * bytes_per_second))
    #         final_beat['stop_index'] = int( (final_beat['start'] + final_beat['duration']) * bytes_per_second )

    #         # save pointers to the raw bytes for each beat with each beat.
    #         final_beat['buffer'] = self.raw_audio[ final_beat['start_index'] : final_beat['stop_index'] ]

    #         info.append(final_beat)

    #     self.__report_progress( .93, "truncating to fade point..." )

    #     # get the max amplitude of the beats
    #     # max_amplitude = max([float(b['amplitude']) for b in info])
    #     max_amplitude = sum([float(b['amplitude']) for b in info]) / len(info)

    #     # assume that the fade point of the song is the last beat of the song that is >= 75% of
    #     # the max amplitude.

    #     self.max_amplitude = max_amplitude

    #     fade = len(info) - 1

    #     for b in reversed(info):
    #         if b['amplitude'] >= (.75 * max_amplitude):
    #             fade = info.index(b)
    #             break

    #     # truncate the beats to [start:fade + 1]
    #     beats = info[self.__start_beat:fade + 1]

    #     loop_bounds_begin = self.__start_beat

    #     self.__report_progress( .93, "computing final beat array..." )

    #     # assign final beat ids
    #     for beat in beats:
    #         beat['id'] = beats.index(beat)
    #         beat['quartile'] = beat['id'] // (len(beats) / 4.0)

    #     # compute a coherent 'next' beat to play. This is always just the next ordinal beat
    #     # unless we're at the end of the song. Then it gets a little trickier.

    #     for beat in beats:
    #         if beat == beats[-1]:

    #             # if we're at the last beat, then we want to find a reasonable 'next' beat to play. It should (a) share the
    #             # same cluster, (b) be in a logical place in its measure, (c) be after the computed loop_bounds_begin, and
    #             # is in the first half of the song. If we can't find such an animal, then just return the beat
    #             # at loop_bounds_begin

    #             beat['next'] = next( (b['id'] for b in beats if b['cluster'] == beat['cluster'] and
    #                                   b['id'] % 4 == (beat['id'] + 1) % 4 and
    #                                   b['id'] <= (.5 * len(beats)) and
    #                                   b['id'] >= loop_bounds_begin), loop_bounds_begin )
    #         else:
    #             beat['next'] = beat['id'] + 1

    #         # find all the beats that (a) are in the same cluster as the NEXT oridnal beat, (b) are of the same
    #         # cluster position as the next ordinal beat, (c) are in the same place in the measure as the NEXT beat,
    #         # (d) but AREN'T the next beat, (e) AREN'T in the same cluster as the current beat, AND
    #         # (f) are more than 7 beats away (to minimize weird local loops)
    #         #
    #         # THAT collection of beats contains our jump candidates

    #         jump_candidates = [bx['id'] for bx in beats[loop_bounds_begin:] if
    #                            (bx['cluster'] == beats[beat['next']]['cluster']) and
    #                            (bx['is'] == beats[beat['next']]['is']) and
    #                            (bx['id'] % 4 == beats[beat['next']]['id'] % 4) and
    #                            (bx['segment'] != beat['segment']) and
    #                            (bx['id'] != beat['next']) and
    #                            (abs(bx['id'] - beats[beat['next']]['id']) > 7)]

    #         if jump_candidates:
    #             beat['jump_candidates'] = jump_candidates
    #         else:
    #             beat['jump_candidates'] = []

    #     # save off the segment count

    #     self.segments = max([b['segment'] for b in beats]) + 1

    #     # we don't want to ever play past the point where it's impossible to loop,
    #     # so let's find the latest point in the song where there are still jump
    #     # candidates and make sure that we can't play past it.

    #     last_chance = len(beats) - 1

    #     for b in reversed(beats):
    #         if len(b['jump_candidates']) > 0:
    #             last_chance = beats.index(b)
    #             break

    #     # if we play our way to the last beat that has jump candidates, then just skip
    #     # to the earliest jump candidate rather than enter a section from which no
    #     # jumping is possible.

    #     beats[last_chance]['next'] = min(beats[last_chance]['jump_candidates'])

    #     # store the beats that start after the last jumpable point. That's
    #     # the outro to the song. We can use these
    #     # beasts to create a sane ending for a fixed-length remix

    #     outro_start = last_chance + 1 + self.__start_beat

    #     if outro_start >= len(info):
    #         self.outro = []
    #     else:
    #         self.outro = info[outro_start:]

    #     #
    #     # This section of the code computes the play_vector -- a 1024*1024 beat length
    #     # remix of the current song.
    #     #

    #     self.__report_progress(0.95, "creating play vector")

    #     play_vector = InfiniteJukebox.CreatePlayVectorFromBeats(beats, start_beat = loop_bounds_begin)

    #     # save off the beats array and play_vector. Signal
    #     # the play_ready event (if it's been set)

    #     self.beats = beats
    #     self.play_vector = play_vector

    #     self.__report_progress(1.0, "finished processing")

    #     if self.play_ready:
    #         self.play_ready.set()

    def __process_audio_madmom(self):

        """ The main audio processing routine for the thread.

        This routine uses Laplacian Segmentation to find and
        group similar beats in the song.

        This code has been adapted from the sample created by Brian McFee at
        https://librosa.github.io/librosa_gallery/auto_examples/plot_segmentation.html#sphx-glr-auto-examples-plot-segmentation-py
        and is based on his 2014 paper published at http://bmcfee.github.io/papers/ismir2014_spectral.pdf

        Additionally, this code performs downbeat detection via the madmom library.

        I have made some performance improvements, but the basic parts remain (mostly) unchanged
        """

        self.__report_progress( .1, "loading file and extracting raw audio")

        #
        # load the file as stereo with a high sample rate
        #

        y, sr = librosa.core.load(self.__filename, mono=False, sr=44100)

        self.duration = librosa.core.get_duration(y,sr)
        self.raw_audio = (y * np.iinfo(np.int16).max).astype(np.int16).T.copy(order='C')
        self.sample_rate = sr

        # after the raw audio bytes are saved, convert the samples to mono
        # because the beat detection algorithm in madmom performs better
        # if there's only one channel.

        y = librosa.core.to_mono(y)

        self.__report_progress( .2, "computing pitch data..." )

        # Compute the constant-q chromagram for the samples.

        BINS_PER_OCTAVE = 12 * 3
        N_OCTAVES = 7

        cqt = librosa.cqt(y=y, sr=sr, bins_per_octave=BINS_PER_OCTAVE, n_bins=N_OCTAVES * BINS_PER_OCTAVE)
        C = librosa.amplitude_to_db( np.abs(cqt), ref=np.max)

        ##########################################################
        # To reduce dimensionality, we'll beat-synchronous the CQT

        downbeats = []

        # if we didn't pass in a beat cache then we'll need to do downbeat
        # detection via madmom

        if self._starting_beat_cache == None:
    
            self.__report_progress( .3, "Running a high precision beat finding algorithm. This could take up to 2 minutes..." )
    
            proc = madmom.features.DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
            act = madmom.features.RNNDownBeatProcessor()(y)
            downbeats = proc(act)
        else:
            # the rest of this code expects downbeats to be a 2d numpy array of
            # [beat_time_in_sec, bar_position]

           self.__report_progress( .3, "Using local beat cache for this file..." )
           downbeats = np.array( [[beat['start'], beat['bar_position']] for beat in self._starting_beat_cache] )

        btz = librosa.time_to_frames(downbeats[:,0], sr=sr)

        Csync = librosa.util.sync(C, btz, aggregate=np.median)

        self.tempo = librosa.beat.tempo(y, sr)

        # For alignment purposes, we'll need the timing of the beats
        # we fix_frames to include non-beat frames 0 and C.shape[1] (final frame)
        beat_times = librosa.frames_to_time(librosa.util.fix_frames(btz,
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
        Msync = librosa.util.sync(mfcc, btz)

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
        _, evecs = scipy.linalg.eigh(L)


        # We can clean this up further with a median filter.
        # This can help smooth over small discontinuities
        evecs = scipy.ndimage.median_filter(evecs, size=(9, 1))


        # cumulative normalization is needed for symmetric normalize laplacian eigenvectors
        Cnorm = np.cumsum(evecs**2, axis=1)**0.5

        self.__report_progress( .5, "clustering..." )

        # if a value for clusters wasn't passed in, then we need to auto-cluster

        if self.clusters == 0:
            self.clusters, seg_ids = self.__compute_best_cluster_with_sil(evecs, Cnorm)

        else: # otherwise, just use the cluster value passed in
            k = self.clusters

            self.__report_progress( .51, "using " + str(self.clusters) + " clusters..." )

            X = evecs[:, :k] / Cnorm[:, k-1:k]

            seg_ids = sklearn.cluster.KMeans(n_clusters=k, max_iter=300,
                                               random_state=0, n_init=20).fit_predict(X)

        # Get the amplitudes and beat-align them
        self.__report_progress( .93, "getting amplitudes" )

        # newer versions of librosa have renamed the rmse function

        if hasattr(librosa.feature,'rms'):
            amplitudes = librosa.feature.rms(y=y)
        else:
            amplitudes = librosa.feature.rmse(y=y)

        ampSync = librosa.util.sync(amplitudes, btz)

        # create a list of tuples that include the ordinal position, the start time of the beat,
        # the cluster to which the beat belongs, the mean amplitude of the beat, and the position
        # of the beat in its bar.

        zbeat_tuples = zip(range(0,len(btz)), beat_times, 
                           seg_ids, ampSync[0].tolist(), 
                           downbeats[:,1])

        beat_tuples =tuple(zbeat_tuples)

        info = []

        bytes_per_second = len(self.raw_audio) / self.duration

        last_cluster = -1
        current_segment = -1
        segment_beat = 0

        for i in range(0, len(beat_tuples)):
            final_beat = {}
            final_beat['start'] = float(beat_tuples[i][1])
            final_beat['cluster'] = int(beat_tuples[i][2])
            final_beat['amplitude'] = float(beat_tuples[i][3])
            final_beat['bar_position'] = int(beat_tuples[i][4])

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

            final_beat['start_index'] = int(final_beat['start'] * bytes_per_second)
            final_beat['stop_index'] = int( (final_beat['start'] + final_beat['duration']) * bytes_per_second )

            # save pointers to the raw bytes for each beat with each beat.
            final_beat['buffer'] = self.raw_audio[ final_beat['start_index'] : final_beat['stop_index'] ]

            info.append(final_beat)

        self.__report_progress( .93, "truncating to fade point..." )

        # get the max amplitude of the beats
        # max_amplitude = max([float(b['amplitude']) for b in info])
        max_amplitude = sum([float(b['amplitude']) for b in info]) / len(info)

        # assume that the fade point of the song is the last beat of the song that is >= 75% of
        # the max amplitude.

        self.max_amplitude = max_amplitude

        fade = len(info) - 1

        for b in reversed(info):
            if b['amplitude'] >= (.75 * max_amplitude):
                fade = info.index(b)
                break

        # truncate the beats to [start:fade + 1]
        beats = info[self.__start_beat:fade + 1]

        loop_bounds_begin = self.__start_beat

        self.__report_progress( .93, "computing final beat array..." )

        # assign final beat ids
        for beat in beats:
            beat['id'] = beats.index(beat)
            beat['quartile'] = beat['id'] // (len(beats) / 4.0)

        # compute a coherent 'next' beat to play. This is always just the next ordinal beat
        # unless we're at the end of the song. Then it gets a little trickier.

        for beat in beats:
            if beat == beats[-1]:

                # if we're at the last beat, then we want to find a reasonable 'next' beat to play. It should (a) share the
                # same cluster, (b) be in a logical place in its measure, (c) be after the computed loop_bounds_begin, and
                # is in the first half of the song. If we can't find such an animal, then just return the beat
                # at loop_bounds_begin

                beat['next'] = next( (b['id'] for b in beats if b['cluster'] == beat['cluster'] and
                                      b['id'] % 4 == (beat['id'] + 1) % 4 and
                                      b['id'] <= (.5 * len(beats)) and
                                      b['id'] >= loop_bounds_begin), loop_bounds_begin )
            else:
                beat['next'] = beat['id'] + 1

            # find all the beats that (a) are in the same cluster as the NEXT oridnal beat, (b) are of the same
            # cluster position as the next ordinal beat, (c) are in the same place in the measure as the NEXT beat,
            # (d) but AREN'T the next beat, (e) AREN'T in the same cluster as the current beat, AND
            # (f) are more than 7 beats away (to minimize weird local loops)
            #
            # THAT collection of beats contains our jump candidates

            jump_candidates = [bx['id'] for bx in beats[loop_bounds_begin:] if
                               (bx['cluster'] == beats[beat['next']]['cluster']) and
                               (bx['is'] == beats[beat['next']]['is']) and
                               (bx['bar_position'] == beats[beat['next']]['bar_position']) and
                               (bx['segment'] != beat['segment']) and
                               (bx['id'] != beat['next']) and
                               (abs(bx['id'] - beats[beat['next']]['id']) > 7)]

            if jump_candidates:
                beat['jump_candidates'] = jump_candidates
            else:
                beat['jump_candidates'] = []

        # save off the segment count

        self.segments = max([b['segment'] for b in beats]) + 1

        # we don't want to ever play past the point where it's impossible to loop,
        # so let's find the latest point in the song where there are still jump
        # candidates and make sure that we can't play past it.

        last_chance = len(beats) - 1

        for b in reversed(beats):
            if len(b['jump_candidates']) > 0:
                last_chance = beats.index(b)
                break

        # if we play our way to the last beat that has jump candidates, then just skip
        # to the earliest jump candidate rather than enter a section from which no
        # jumping is possible.

        beats[last_chance]['next'] = min(beats[last_chance]['jump_candidates'])

        # store the beats that start after the last jumpable point. That's
        # the outro to the song. We can use these
        # beasts to create a sane ending for a fixed-length remix

        outro_start = last_chance + 1 + self.__start_beat

        if outro_start >= len(info):
            self.outro = []
        else:
            self.outro = info[outro_start:]

        #
        # This section of the code computes the play_vector -- a 1024*1024 beat length
        # remix of the current song.
        #

        self.__report_progress(0.95, "creating play vector")

        play_vector = InfiniteJukebox.CreatePlayVectorFromBeatsMadmom(beats, start_beat = loop_bounds_begin)

        # save off the beats array and play_vector. Signal
        # the play_ready event (if it's been set)

        self.beats = beats
        self.play_vector = play_vector

        self.__report_progress(1.0, "finished processing")

        if self.play_ready:
            self.play_ready.set()

    def __report_progress(self, pct_done, message):

        """ If a reporting callback was passed, call it in order
            to mark progress.
        """
        if self.__progress_callback:
            self.__progress_callback( pct_done, message )

    def __compute_best_cluster_with_sil(self, evecs, Cnorm):

        ''' Attempts to compute optimum clustering

            Uses the the silhouette score to pick the best number of clusters.
            See: https://en.wikipedia.org/wiki/Silhouette_(clustering)

            PARAMETERS:
                evecs: Eigen-vectors computed from the segmentation algorithm
                Cnorm: Cumulative normalization of evecs. Easier to pass it in than
                       compute it from scratch here.

            KEY DEFINITIONS:

                  Clusters: buckets of musical similarity
                  Segments: contiguous blocks of beats belonging to the same cluster
                Silhouette: A score given to a cluster that measures how well the cluster
                            members fit together. The value is from -1 to +1. Higher values
                            indicate higher quality.
                   Orphans: Segments with only one beat. The presence of orphans is a potential
                            sign of overfitting.

            SUMMARY:

                There are lots of things that might indicate one cluster count is better than another.
                High silhouette scores for the candidate clusters mean that the jumps will be higher
                quality.

                On the other hand, we could easily choose so many clusters that everyone has a great
                silhouette score but none of the beats have other segments into which they can jump.
                That will be a pretty boring result!

                So, the cluster/segment ratio matters, too The higher the number, the more places (on average)
                a beat can jump. However, if the beats aren't very similar (low silhouette scores) then
                the jumps won't make any musical sense.

                So, we can't just choose the cluster count with the highest average silhouette score or the
                highest cluster/segment ratio.

                Instead, we comput a simple fitness score of:
                        cluster_count * ratio * average_silhouette

                Finally, segments with only one beat are a potential (but not definite) sign of overfitting.
                We call these one-beat segments 'orphans'. We want to keep an eye out for those and slightly
                penalize any candidate cluster count that contains orphans.

                If we find an orphan, we scale the fitness score by .8 (ie. penalize it 20%). That's
                enough to push any candidate cluster count down the stack rank if orphans aren't
                otherwise very common across most of the other cluster count choices.

        '''

        self._clusters_list = []

        best_cluster_size = 0
        best_labels = None
        best_cluster_score = 0

        # we need at least 3 clusters for any song and shouldn't need to calculate more than
        # 48 clusters for even a really complicated peice of music.

        cluster_ratio_map = []
        cluster_ratio_map.append(['Clusters', 'AVG(sil)', 'MIN(seg_len)', 'Ratio', 'Cluster Score'])

        for n_clusters in range(48, 3, -1):

            report_pct = .95 - (n_clusters/100.00)
            self.__report_progress(round(report_pct,2), "Testing a cluster value of %d..." % n_clusters)

            # compute a matrix of the Eigen-vectors / their normalized values
            X = evecs[:, :n_clusters] / Cnorm[:, n_clusters-1:n_clusters]

            # create the candidate clusters and fit them

            cluster_labels = sklearn.cluster.KMeans(n_clusters=n_clusters, 
                                                    max_iter=600,
                                                    random_state=10, 
                                                    n_init=10).fit_predict(X)

            # get some key statistics, including how well each beat in the cluster resemble
            # each other (the silhouette average), the ratio of segments to clusters, and the
            # length of the smallest segment in this cluster configuration

            silhouette_avg = sklearn.metrics.silhouette_score(X, cluster_labels)

            ratio, min_segment_len = self.__segment_stats_from_labels(cluster_labels.tolist())

            # We need to grade each cluster according to how likely it is to produce a good
            # result. There are a few factors to look at.
            #
            # First, we can look at how similar the beats in each cluster (on average) are for
            # this candidate cluster size. This is known as the silhouette score. It ranges
            # from -1 (very bad) to 1 (very good).
            #
            # Another thing we can look at is the ratio of clusters to segments. Higher ratios
            # are preferred because they afford each beat in a cluster the opportunity to jump
            # around to meaningful places in the song.
            #
            # All other things being equal, we prefer a higher cluster count to a lower one
            # because it will tend to make the jumps more selective -- and therefore higher
            # quality.
            #
            # Lastly, if we see that we have segments equal to just one beat, that might be
            # a sign of overfitting. We call these one beat segments 'orphans'. Some songs,
            # however, will have orphans no matter what cluster count you use. So, we don't
            # want to throw out a cluster count just because it has orphans. Instead, we
            # just de-rate its fitness score. If most of the cluster candidates have orphans
            # then this won't matter in the overall scheme because everyone will be de-rated
            # by the same scaler.
            #
            # Putting this all together, we muliply the cluster count * the average
            # silhouette score for the clusters in this candidate * the ratio of clusters to
            # segments. Then we scale (or de-rate) the fitness score by whether or not is has
            # orphans in it.

            orphan_scaler = .5 if min_segment_len == 1 else 1

            #cluster_score = n_clusters * silhouette_avg * ratio * orphan_scaler
            
            # NOTE: Removing the effects of the orphan_scaler for now because it seems
            #       to be doing more harm than good.

            # NOTE: I'm playing with different fitness functions. As in all machine learning,
            # this has become the hardest bit.

            # My first thought was simply to choose the largest clustering that had a 
            # ratio > 3, an average silhouette score > .5, and no orphans. That didn't
            # quite produce the outcome I wanted

            # cluster_score = n_clusters if (ratio >= 3.0 and silhouette_avg > .5 and min_segment_len > 1) else 0

            # My next attempt is to compute a composite score of the number of clusters 
            # plus a scaled value of the silhouette average plus the minimum segment length.
            # So far this is producing a better result, but I think I need to compute and 
            # account for the maximum segment lengths, too.

            cluster_score = 0.0

            if ( ratio >= 3.0 and silhouette_avg > .5 ):
                cluster_score = n_clusters + \
                                (10.0 * silhouette_avg) + \
                                min(min_segment_len, 8) + \
                                ratio

            # I'm keeping track of the basic statistics per cluster value tested so I can
            # print them at the end of the evaluation in the hopes of discovering the optimal
            # general fitness function.

            cluster_ratio_map.append([n_clusters, round(silhouette_avg * 100, 2), min_segment_len, round(ratio,4), round(cluster_score,4)])

            # if this cluster count has a score that's better than the best score so far, store
            # it for later.

            if cluster_score >= best_cluster_score:
                best_cluster_score = cluster_score
                best_cluster_size = n_clusters
                best_labels = cluster_labels

        self.cluster_ratio_log = {'best_cluster_size': best_cluster_size, 'cluster_scores': cluster_ratio_map}

        # print a nice table of the cluster metrics I stored in each iteration
        for cr in cluster_ratio_map:
            c, sa, msl, r, cs = cr
            msg = "{:<10} {:<10} {:<15} {:<8} {:<15}".format(c,sa,msl,r,cs)
            print(msg)
   
        msg = "Selected best cluster size of: {}".format(best_cluster_size)
        print (msg)

        # return the best results
        return (best_cluster_size, best_labels)

    @staticmethod
    def __segment_count_from_labels(labels):

        ''' Computes the number of unique segments from a set of ordered labels. Segements are
            contiguous beats that belong to the same cluster. '''

        segment_count = 0
        previous_label = -1

        for label in labels:
            if label != previous_label:
                previous_label = label
                segment_count += 1

        return segment_count

    def __segment_stats_from_labels(self, labels):
        ''' Computes the segment/cluster ratio and min segment size value given an array
            of labels. '''

        segment_count = 0.0
        segment_length = 0
        clusters = max(labels) + 1

        previous_label = -1

        segment_lengths = []

        for label in labels:
            if label != previous_label:
                previous_label = label
                segment_count += 1.0

                if segment_length > 0:
                    segment_lengths.append(segment_length)

                segment_length = 1
            else:
                segment_length +=1

        return float(segment_count) / float(clusters), min(segment_lengths)

    def __add_log(self, line):
        """Convenience method to add debug logging info for later"""

        self._extra_diag += line + "\n"

    def CreatePlayVectorFromBeatsMadmom(beats, start_beat = 0):

        #
        # This section of the code computes the play_vector -- a 1024*1024 beat length
        # remix of the current song.
        #

        print("CreatePlayVectorFromBeatsMadmom")

        random.seed()
        duration = beats[-1]['start'] + beats[-1]['duration']
        tempo = (len(beats)/duration) * 60

        # how long should our longest contiguous playback blocks be? One way to
        # consider it is that higher bpm songs need longer blocks because
        # each beat takes less time. A simple way to estimate a good value
        # is to scale it by it's distance from 120bpm -- the canonical bpm
        # for popular music. Find that value and round down to the nearest
        # multiple of 4. (There almost always are 4 beats per measure in Western music).

        beats_per_bar = max([ b['bar_position'] for b in beats ])

        max_sequence_len = int(round((tempo / 120.0) * 48.0))
        max_sequence_len = max_sequence_len - (max_sequence_len % 4)

        acceptable_jump_amounts = [x for x in [16, 24, 32, 48, 64, 72, 96, 128] if x <= max_sequence_len]

        if beats_per_bar == 3:
            acceptable_jump_amounts = [(a * .75) for a in acceptable_jump_amounts]

        # min_sequence = max(random.randrange(16, max_sequence_len, 4), start_beat) + 1

        min_sequence = random.choice(acceptable_jump_amounts) - (beats[1]['bar_position'] + 2)

        current_sequence = 0
        beat = beats[0]

        play_vector = []

        play_vector.append( {'beat':0, 'seq_len':min_sequence, 'seq_pos':current_sequence} )

        # we want to keep a list of recently played segments so we don't accidentally wind up in a local loop
        #
        # the number of segments in a song will vary so we want to set the number of recents to keep
        # at 25% of the total number of segments. Eg: if there are 34 segments, then the depth will
        # be set at round(8.5) == 9.
        #
        # On the off chance that the (# of segments) *.25 < 1 we set a floor queue depth of 1

        segments = max([b['segment'] for b in beats]) + 1

        recent_depth = int(round(segments * .25))
        recent_depth = max( recent_depth, 1 )

        recent = collections.deque(maxlen=recent_depth)

        # keep track of the time since the last successful jump. If we go more than
        # 20% of the song length since our last jump, then we will prioritize an
        # immediate jump to a not recently played segment. Otherwise playback will
        # be boring for the listener. This also has the advantage of busting out of
        # local loops.

        max_beats_between_jumps = int(round(len(beats) * .1))
        acceptable_jump_amounts = [x for x in acceptable_jump_amounts if x <= max_beats_between_jumps]

        beats_since_jump = 0
        failed_jumps = 0

        for i in range(0, 1024 * 1024):

            if beat['segment'] not in recent:
                recent.append(beat['segment'])

            current_sequence += 1

            # it's time to attempt a jump if we've played all the beats we wanted in the
            # current sequence. Also, if we've gone more than 10% of the length of the song
            # without jumping we need to immediately prioritze jumping to a non-recent segment.

            will_jump = (current_sequence == min_sequence) or \
                        (beats_since_jump >= max_beats_between_jumps)

            # since it's time to jump, let's find the most musically pleasing place
            # to go

            if ( will_jump ):

                # find the jump candidates that haven't been recently played

                non_recent_candidates = []

                # if beat['bar_position'] == beats_per_bar:
                non_recent_candidates = [c for c in beat['jump_candidates'] if beats[c]['segment'] not in recent]

                # if there aren't any good jump candidates, then we need to fall back
                # to another selection scheme.

                if len(non_recent_candidates) == 0:

                    beats_since_jump += 1
                    failed_jumps += 1

                    # suppose we've been trying to jump but couldn't find a good non-recent candidate. If
                    # the length of time we've been trying (and failing) is >= 10% of the song length
                    # then it's time to relax our criteria. Let's find the jump candidate that's furthest
                    # from the current beat (irrespective if it's been played recently) and go there. Ideally
                    # we'd like to jump to a beat that is not in the same quartile of the song as the currently
                    # playing section. That way we maximize our chances of avoiding a long local loop -- such as
                    # might be found in the section preceeding the outro of a song.

                    non_quartile_candidates = [c for c in beat['jump_candidates'] if beats[c]['quartile'] != beat['quartile']]

                    if (failed_jumps >= (.1 * len(beats))) and (len(non_quartile_candidates) > 0):

                        furthest_distance = max([abs(beat['id'] - c) for c in non_quartile_candidates])

                        jump_to = next(c for c in non_quartile_candidates
                                       if abs(beat['id'] - c) == furthest_distance)

                        beat = beats[jump_to]
                        beats_since_jump = 0
                        failed_jumps = 0

                    # uh oh! That fallback hasn't worked for yet ANOTHER 30%
                    # of the song length. Something is seriously broken. Time
                    # to punt and just start again from the first beat.

                    elif failed_jumps >= (.3 * len(beats)):
                        beats_since_jump = 0
                        failed_jumps = 0
                        beat = beats[start_beat]

                    # asuuming we're not in one of the failure modes but haven't found a good
                    # candidate that hasn't been recently played, just play the next beat in the
                    # sequence

                    else:
                        beat = beats[beat['next']]

                else:

                    # if it's time to jump and we have at least one good non-recent
                    # candidate, let's just pick randomly from the list and go there

                    beats_since_jump = 0
                    failed_jumps = 0
                    beat = beats[ random.choice(non_recent_candidates) ]

                # reset our sequence position counter and pick a new target length
                # between 16 and max_sequence_len, making sure it's evenly divisible by
                # 4 beats

                current_sequence = 0

                min_sequence = random.choice(acceptable_jump_amounts) - (beat['bar_position'] + 2)

                # if the beats we'd like to play would make us go longer than
                # the max number of beats we should go between jumps, then let's
                # only try to play as many beats as it takes to get to
                # max_beats_between_jumps. We might be in a local loop so there's
                # no point hanging around!

                if min_sequence > (max_beats_between_jumps - beats_since_jump):
                    min_sequence = (max_beats_between_jumps - beats_since_jump)

                # if we're in the place where we want to jump but can't because
                # we haven't found any good candidates, then set current_sequence equal to
                # min_sequence. During playback this will show up as having 00 beats remaining
                # until we next jump. That's the signal that we'll jump as soon as we possibly can.
                #
                # Code that reads play_vector and sees this value can choose to visualize this in some
                # interesting way.

                if beats_since_jump >= max_beats_between_jumps:
                    current_sequence = min_sequence

                # add an entry to the play_vector
                play_vector.append({'beat':beat['id'], 'seq_len': min_sequence, 'seq_pos': current_sequence})
            else:

                # if we're not trying to jump then just add the next item to the play_vector
                play_vector.append({'beat':beat['next'], 'seq_len': min_sequence, 'seq_pos': current_sequence})
                beat = beats[beat['next']]
                beats_since_jump += 1

        return play_vector