# Remixatron Web UI
(c) 2017-2021 - Dave Rensin - drensin@gmail.com

Creates an infinite remix of an audio file by finding musically similar beats and computing a randomized play path through them. The default choices should be suitable for a variety of musical styles. This work is inspired by the Infinite Jukebox (http://www.infinitejuke.com) project creaeted by Paul Lamere (paul@spotify.com)

It groups musically similar beats of a song into clusters and then plays a random path through the song that makes musical sense, but not does not repeat. It will do this infinitely.

This is a web ui version of the CLI found at [main page](https://github.com/drensin/Remixatron). Please see the [README.md](https://github.com/drensin/Remixatron/blob/master/README.md) at that site for a deeper explanation of how it all works under the hood.

# Requirements
You must have these installed for this to work.
1) ffmpeg
2) Python 3.6+ (Python 3.7 preferred)

# Installation
I strongly reccommend that you setup and use a python virtual environment to run this server. Once you do that, you can install with just:

    pip install --upgrade pip
    pip install -r requirements.txt
    
On Windows make sure you add the following directories to your Path:

    # The directory where you installed ffmpeg.exe, eg:
    C:\ffmpeg\bin 
    # The python subdirectory where yt-dlp.exe was installed, eg:
    C:\Users\Name\AppData\Roaming\Python\Python36\Scripts 

# Running
    python3 main.py

**NOTE:**  This program uses [yt-dlp](https://github.com/yt-dlp/yt-dlp) - a more performant fork of youtube-dl, which is updated A LOT. I strongly reccommend that you periodically make sure that you have the latest and greatest yt-dlp installed before launching the server each time.

# Connecting

Just navigate to http://localhost:8000 with a browser running on the same machine as the server. If you want to connect from another machine on your LAN, then you'll need to edit the cors.cfg file to allow it. For example, suppose your computer is named *mymachine.local*, the correct CORS config to allow it and localhost will be:

    {
	"ORIGINS":["http://localhost:8000",
               "http://mymachine.local:8000"]
    }

Now, you should be able to go to any machine on your home network and navigate to *http://mymachine.local:8000/* and use the app.
