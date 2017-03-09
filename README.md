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

python infinite_jukebox.py [-h] [-clusters CLUSTERS] [-start START] filename

Creates an infinite remix of an audio file by finding musically similar beats and computing a randomized play path through them. The default choices should be suitable for a variety of musical styles. This work is inspired by the Infinite Jukebox (http://www.infinitejuke.com) project created by Paul Lemere (paul@echonest.com)  
  
    positional arguments:  
      filename            the name of the audio file to play. Most common audio  
                          types should work. (mp3, wav, ogg, etc..)  
  
    optional arguments:  
      -h, --help          show this help message and exit  
      -clusters CLUSTERS  set the number of clusters into which we want to bucket  
                          the audio. Deafult: 0 (automatically try to find the  
                          optimal cluster value.)  
      -start START        start on beat N. Deafult: 1  
  
**Example:**  
  
    **rensin@dev-instance:~/src/Remixatron$** python infinite_jukebox.py i_cant_go_for_that.mp3 

    [##########] ready                                                                                                
  
       filename: i_cant_go_for_that.mp3  
       duration: 224.095782 seconds  
          beats: 396  
          tempo: 109.956782 beats per minute  
       clusters: 14  
     samplerate: 44100  
     
.........................................................[07]....................................................

The dots represent the beats of the song. The number is a countdown of how many beats until the playback will attempt to jump to a random place in the song that is musically similar. The **position** of the countdown is the part of the song that is now playing. In the above example, the song is playing at (roughly) its midpoint and will attempt a musically sensible jump in 7 beats.

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
