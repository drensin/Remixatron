""" rebuildBeatmapCacheFromBookmarks

    This utility will read the global bookmarks file ($USER/.remixatron/remixatron.global.bookmarks)
    and create and beat cache .bz2 files for each of them. Having these cached 
    will shortcut the lengthy beat finding algorithm when a user asks to play a bookmark.

    Normally, you should not have to run this, but it's here in case the .bz2 files in your
    .remixatron dir get deleted/corrupted/etc.

    If you want to force re-generate all the beat cache files, delete
    $USER/.remixatron/*.bz2. Then run this utility.
"""

import bz2
from genericpath import exists
from importlib.resources import path
import json
import os
import subprocess
import sys
import urllib.parse
import tempfile

from pathlib import Path
from Remixatron import InfiniteJukebox

# supress warnings from any of the imported libraries. This will keep the
# console clean.

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# remixatron_dir is the location of the cached files and global bookmarks file
# make sure the .remixatron directory exists in the home dir of the running
# user. If not, create it. Then store it for later.

remixatron_dir = (Path.home() / '.remixatron')

def fetch_from_youtube(url:str) -> str:

    """ Fetches the reuquested audio from Youtube, and trims any leading or
    trailing silence from it via ffmpeg. This uses the program yt-dlp.

    Args:
        url (string): the url to fetch
    Returns:
        string: the final file name of the retrieved and trimmed audio.
    """

    # this function runs out-of-process from the main serving thread, so
    # send an update to the client.
    print( "Asking for audio..." ) 

    # download the file (audio only) at the highest quality and save it in /tmp
    try:

        tmpfile = tempfile.gettempdir() + '/audio.tmp'

        cmd = ['yt-dlp', '--write-info-json', '-x', '--audio-format', 'best', 
               '--no-playlist', '-o', tmpfile, url]

        result = [] 
        cmdOutput = ''

        for line in subprocess.check_output(cmd).splitlines():
            line = line.decode('utf-8')
            result.append(line)
            cmdOutput = cmdOutput + "\t" + line + os.linesep

        print("###### yt-dlp output ######" + os.linesep + os.linesep + cmdOutput)

    except subprocess.CalledProcessError as e:

        print( "Failed to download the audio from Youtube. Check the logs!" )
        return None

    fn = ":".join(result[-2].split(":")[1:])[1:]

    if os.path.exists(fn) == False:
        # uh oh. there was a problem. Let's skip this.
        return None

    # trim silence from the ends and save as ogg
    of = tempfile.gettempdir() + '/audio.ogg'

    # result = subprocess.run(['ffmpeg', '-y', '-i', fn, of],
    #                         stderr=subprocess.PIPE, stdout=subprocess.PIPE)

    # delete the downlaoded file because we don't need it anymore
    # os.remove(fn)

    os.rename(fn, of)
    # return the name of the trimmed file
    return of

def loadGlobalBookmarks() -> str:
    """ Return any bookmarks stored on the server. They are a simple ordered JSON
        array stored in $USER/.remixatron/remixatron.global.bookmarks

    Returns:
        JSON string of the track info
    """

    j = ''

    globalBookmarksFn = (remixatron_dir / "remixatron.global.bookmarks")

    if os.path.exists(globalBookmarksFn) == False:
        with open(globalBookmarksFn, 'w') as f:
            j = json.dumps([])
            f.write(j)
            f.flush()
            f.close()

    with open(globalBookmarksFn, 'r') as f:
        j = f.read()
        print("json len: {}".format(len(j)))

    return j

def saveGlobalBookmarks(json_string : str):
    """ Save an array bookmarks to /tmp/remixatron.global.bookmarks
    """

    globalBookmarksFn = (remixatron_dir / 'remixatron.global.bookmarks')

    with open(globalBookmarksFn, 'w') as f:
        f.write(json_string)
        f.flush()
        f.close()

def process_audio(url, clusters=0) -> InfiniteJukebox:
    """ The main processing for the audio is done here. It makes heavy use of the
    InfiniteJukebox class (https://github.com/drensin/Remixatron).

    Args:
        url (string): the URL to fetch or the file uploaded
    """

    cached_beatmap_fn = ( remixatron_dir / (urllib.parse.quote(url, safe='') + '.beatmap.bz2') )

    # if the cache is already there, return.
    if os.path.exists(cached_beatmap_fn):
        print('Beat cache for this url already present. Skipping...')
        return None

    fn = fetch_from_youtube(url)

    if fn == None:
        print("uh oh.. Downloading failed. Moving on..")
        return None

    print('fetch complete')

    # The constructor of the InfiniteJukebox class takes a callback to which to
    # post update messages. We define that callback here so that it has access
    # to the userid variable. This uses Python 'variable capture'.
    # (See https://bit.ly/2YOKm16 for more about how this works)

    def remixatron_callback(percentage, message):
        print( str(round(percentage * 100,0)) + "%: " + message )

    remixatron_callback(0.1, 'Audio downloaded')

    # all of the core analytics and processing is done in this call
    jukebox = InfiniteJukebox(fn, clusters=clusters,
                                progress_callback=remixatron_callback,
                                start_beat=0, do_async=False)

    def skip_encoder(o):
        return ''

    with bz2.open(cached_beatmap_fn, 'wb') as f:
        f.write(json.dumps(jukebox.beats, default=skip_encoder).encode('utf-8'))

    os.remove(fn)

    return jukebox

def main():

    bookmarksStr = loadGlobalBookmarks()
    bookmarksJSON = json.loads(bookmarksStr)

    print( "Found {} bookmarks to process.".format(len(bookmarksJSON)))

    newBookmarks = []

    for bookmark in bookmarksJSON:
        print( os.linesep + os.linesep + \
               '############################################################################' + os.linesep + \
               '# Building beat cache for: ' + bookmark['title'] + os.linesep + \
               '############################################################################' + \
               os.linesep + os.linesep)

        # jukebox = process_audio(bookmark['url'], bookmark['clusters'])

        # re-find the opitmal cluster
        jukebox = process_audio(bookmark['url'], 0)

        if jukebox != None:
            bookmark['clusters'] = jukebox.clusters
        
        newBookmarks.append(bookmark)

    # write the new bookmarks with the new optimal cluster values
    saveGlobalBookmarks(json.dumps(newBookmarks, indent=3, sort_keys=True))

    print('done')

if __name__ == "__main__":
    main()