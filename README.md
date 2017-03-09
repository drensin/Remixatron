# Remixatron

usage: infinite_jukebox.py [-h] [-clusters CLUSTERS] [-start START] filename

Creates an infinite remix of an audio file by finding musically similar beats and computing a randomized play path through them. The default choices should be suitable for a variety of musical styles. This work is inspired by the Infinite Jukebox (http://www.infinitejuke.com) project creaeted by Paul Lemere (paul@echonest.com)  
  
    positional arguments:  
      filename            the name of the audio file to play. Most common audio  
                          types should work. (mp3, wav, ogg, etc..)  
  
    optional arguments:  
      -h, --help          show this help message and exit  
      -clusters CLUSTERS  set the number of clusters into which we want to bucket  
                          the audio. Deafult: 0 (automatically try to find the  
                          optimal cluster value.)  
      -start START        start on beat N. Deafult: 1  
  
