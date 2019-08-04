# Remixatron Web UI
(c) 2017-2019 - Dave Rensin - drensin@gmail.com

Creates an infinite remix of an audio file by finding musically similar beats and computing a randomized play path through them. The default choices should be suitable for a variety of musical styles. This work is inspired by the Infinite Jukebox (http://www.infinitejuke.com) project creaeted by Paul Lamere (paul@spotify.com)

It groups musically similar beats of a song into clusters and then plays a random path through the song that makes musical sense, but not does not repeat. It will do this infinitely.

This is a web ui version of the CLI found at [main page](https://github.com/drensin/Remixatron). Please see the [README.md](https://github.com/drensin/Remixatron/blob/master/README.md) at that site for a deeper explanation of how it all works under the hood.

# Requirements
You must have these installed for this wot work.
1) ffmpeg
2) Python 3.6+ (Python 3.7 preferred)

# Installation
I strongly reccommend that you setup and use a python virtual environment to run this server. Once you do that, you can install with just:

    pip install --upgrade pip
    pip install -r requirements.txt

# Running
    pip install --upgrade youtube-dl; python main.py

**NOTE:**  This program uses the [youtube-dl](https://ytdl-org.github.io/youtube-dl/index.html) python module, which is updated A LOT. I strongly reccommend that you run the above command to make sure that you have the latest and greatest youtube-dl module before launching the server each time.

# Connecting

Just navigate to http://localhost:8000 with a browser running on the same machine as the server. If you want to connect from another machine on your LAN, then you'll need to edit the cors.cfg file to allow it. For example, suppose your computer is named *mymachine.local*, the correct CORS config to allow it and localhost will be:

    {
	"ORIGINS":["http://localhost:8000",
               "http://mymachine.local:8000"]
    }

Now, you should be able to go to any machine on your home network and navigate to *http://mymachine.local:8000/* and use the app.
