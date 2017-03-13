# Remixatron
(c) 2017 - Dave Rensin - dave@rensin.com

This program attempts to recreate the wonderful Infinite Jukebox (http://www.infinitejuke.com) on the command line in Python. It groups musically similar beats of a song into clusters and then plays a random path through the song that makes musical sense, but not does not repeat. It will do this infinitely.  

***
# Installation  

pip install --upgrade pip  
pip install --user pygame pyparsing numpy  
pip install --user librosa  
***
# Usage  

usage: infinite_jukebox.py [-h] [-clusters N] [-start start_beat]
                           [-save label] [-duration seconds] [-verbose]
                           filename

Creates an infinite remix of an audio file by finding musically similar beats and computing a randomized play path through them. The default choices should be suitable for a variety of musical styles. This work is inspired by the Infinite Jukebox (http://www.infinitejuke.com) project creaeted by Paul Lamere (paul@spotify.com)

    positional arguments:
      filename           the name of the audio file to play. Most common audio
                         types should work. (mp3, wav, ogg, etc..)

    optional arguments:
      -h, --help         show this help message and exit
      -clusters N        set the number of clusters into which we want to bucket
                         the audio. Deafult: 0 (automatically try to find the
                         optimal cluster value.)
      -start start_beat  start on a specific beat. Deafult: 1
      -save label        Save the remix to a file, rather than play it. Will
                         create file named [label].wav
      -duration seconds  length (in seconds) to save. Must use with -save.
                         Deafult: 180
      -verbose           print extra info about the track and play vector
  
**Example 1:**  

Play a song infinitely.

    $ python infinite_jukebox.py i_cant_go_for_that.mp3 

    [##########] ready                                                                                                
  
       filename: i_cant_go_for_that.mp3  
       duration: 224.095782 seconds  
          beats: 396  
          tempo: 109.956782 beats per minute  
       clusters: 14  
     samplerate: 44100  
     
    .........................................................[07]..............................

The dots represent the beats of the song. The number is a countdown of how many beats until the playback will attempt to jump to a random place in the song that is musically similar. The **position** of the countdown is the part of the song that is now playing. In the above example, the song is playing at (roughly) 70% and will attempt a musically sensible jump in 7 beats.

**Example 2:**

Play with verbose info.

    $ python infinite_jukebox.py test_audio_files/i_got_bills.mp3 -verbose

    [##########] ready                                                                                                
    
       filename: i_got_bills.mp3
       duration: 203.035283 seconds
          beats: 424
          tempo: 126.048018 beats per minute
       clusters: 23
       segments: 82
     samplerate: 44100
    

    Segmemt Map:
    ###----------##----------###----######-----#---------########---############################----###---------####
    ---------#####--##----###-----###---##-##-##--################################--###--------####----------######-
    ###---###-----###--##----####----###----###########---######----#########-------########----############------##
    ###-##########-------###--####---#####---##---###----####----#####---------##-------####

    Cluster Map:
    8882222222222LL5555555555333FFFF00000044444I000000000IIIIIIIIbbbAAAAAAAAAAAAAAAAAAAAAAAAAAAAMMMM888222222222LLLL
    55555555533333FFcc7777111eeeee999ccc771hh1hh66AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAMM88822222222LLLL5555555555333333F
    ccc777111eeeee999cc771111eeee9999cccGGGGkkkkkkkkkkkGGGDDDDDD6666222222222LLLLLLL555555553333FFFFFFFFFFFF00000044
    444I0000000000IIIIIIIbbbcc7777111eeeee999cc777111eeee9999cccchhhhhjjjjjjjjjcchhhhhhh6666

    ........[16]....................................................................................

**Example 3:**

Create a 4 minute remix named *myRemix.wav*

    $ python infinite_jukebox.py i_cant_go_for_that.mp3 -save myRemix -duration 240 

    [##########] ready                                                                                                
  
       filename: i_cant_go_for_that.mp3  
       duration: 224.095782 seconds  
          beats: 396  
          tempo: 109.956782 beats per minute  
       clusters: 14  
     samplerate: 44100  


***
  
# Some notes about the code  

The core work is done in the InfiniteJukebox class in the Remixatron module. *infinite_jukebox.py* is just a simple demonstration on how to use that class.  

The InfiniteJukebox class can do its processing in a background thread and reports progress via the progress_callback arg. To run in a thread, pass *async=True* to the constructor. In that case, it exposes an Event named *play_ready* -- which will be signaled when the processing is complete. The default mode is to run synchronously.  

Simple async example:

      def MyCallback(percentage_complete_as_float, string_message):
        print "I am now %f percent complete with message: %s" % (percentage_complete_as_float * 100, string_message)

      jukebox = InfiniteJukebox(filename='some_file.mp3', progress_callback=MyCallback, async=True)
      jukebox.play_ready.wait()

      <some work here...>
  
Simple Non-async example:

      def MyCallback(percentage_complete_as_float, string_message):
        print "I am now %f percent complete with message: %s" % (percentage_complete_as_float * 100, string_message)

      jukebox = InfiniteJukebox(filename='some_file.mp3', progress_callback=MyCallback, async=False)

      <blocks until completion... some work here...>
      
Example: Playing the first 32 beats of a song:  

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
