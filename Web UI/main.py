"""
    Remixatron Web UI
    (c) 2019 - Dave Rensin - dave@rensin.com

    This is a web ui for the Remixatron Infinite Jukebox project. (https://github.com/drensin/Remixatron)

    The original project was a CLI that used curses, but that caused problems on Windows and other systems. So,
    I wrote thsi UI to be a bit more friendly. See the README.MD file in this project for information about
    how to install, configure, and run this.

"""
import bz2
import collections
from genericpath import exists
import glob
from importlib.resources import path
import json
from textwrap import indent
import numpy as np
import os
import requests
import secrets
import subprocess
import soundfile as sf
import sys
import urllib.parse
import tempfile

from flask import Flask, current_app, g, make_response, redirect, request, send_from_directory, session, url_for
from flask_compress import Compress
from flask_socketio import SocketIO, emit, send, join_room, leave_room

from multiprocessing import Process
from pathlib import Path
from Remixatron import InfiniteJukebox

# supress warnings from any of the imported libraries. This will keep the
# console clean.

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

app = Flask(__name__)

# make sure that any files we send expire really quickly so caching
# doesn't screw up playback

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

# turn on gzip compression for all responses
compress = Compress(app)

socketio = None

# make sure the .remixatron directory exists in the home dir of the running
# user. If not, create it. Then store it for later.

remixatron_dir = (Path.home() / '.remixatron')

if remixatron_dir.exists() == False:
    os.mkdir(remixatron_dir)

# the cors.cfg file defines which domains will be trusted for connections. The
# default entry of localhost:8000 will be fine if you are running the web
# browser on the same host as this server. Otherwise, you'll have to modify
# accordingly. See the README.MD for this project for more info.

cors_file = None

if (remixatron_dir / 'cors.cfg').exists():
    cors_file = str(remixatron_dir / 'cors.cfg')
elif os.path.isfile("cors.cfg"):
    cors_file = 'cors.cfg'

print(cors_file)

if cors_file != None:
    with open(cors_file) as cors_file:
        origins = json.load(cors_file)
        # socketio = SocketIO(app, cors_allowed_origins = origins['ORIGINS'], logger=True, engineio_logger=True)
        socketio = SocketIO(app, cors_allowed_origins = origins['ORIGINS'])
else:
    socketio = SocketIO(app)

# this will hold the 50 most recent messages for each client
messageQueues = {}

# this will hold the Process object current being used for audio processing
# for each client. No client should have more than one process working at a
# time
procMap = {}

def get_userid():
    """ Returns the device id of the connected user form their cookies.

    Returns:
        string: the device id stored in the browser cookie
    """

    return request.cookies.get('deviceid')

def redirect_https(dir, path):

    """If this server sits behind a proxy like gunicorn or nginix, then this function
    will figure that out and re-write the redirect headers accordingly.

    Args:
        dir (string): the local directory for the item to return
        path (string): the file name for the item to return

    Returns:
        flask.Response: a Flask Response object
    """

    # Grab the device id. If it hasn't already been set, then create one and set it
    deviceid = get_userid()

    if deviceid == None:
        deviceid = secrets.token_urlsafe(16)

    # if this server is behind a proxy like ngnix or gunicorn, then we should detect that and
    # re-write the redirect URL to work correctly.

    if 'X-Forwarded-Proto' in request.headers and request.headers['X-Forwarded-Proto'] == 'https':
        url = 'https://' + request.host + url_for(dir, filename=path)

        resp = redirect(url)
        resp.set_cookie('deviceid',deviceid, max_age=31536000)
        resp.headers.add('Cache-Control', 'no-store')
        return resp

    # otherwise, just set the device id and redirect as required

    resp = redirect(url_for(dir, filename=path))
    resp.set_cookie('deviceid',deviceid, max_age=31536000)
    resp.headers.add('Cache-Control', 'no-store')
    return resp


@socketio.on_error_default
def default_error_handler(e):
    """ The default error handler for the flask_socketio module. Just print the error
    to the console.

    Args:
        e (flask_socketio.Error): the error
    """

    print(e)

@socketio.on('connect')
def on_connect():

    """ This gets fired when the client connects (or re-connects) via the
    socketio library. (See https://www.socket.io for more details)

    Returns:
        flask.Response: a Flask Response object
    """

    deviceid = get_userid()

    print('******** ' + get_userid() + ' has connected. ********')

    # if there's no device id already set for the client, create one and
    # store it

    if deviceid == None:
        deviceid = secrets.token_urlsafe(16)

        resp = make_response(deviceid,200)
        resp.set_cookie('deviceid',deviceid, max_age=31536000)

        return resp

    join_room(deviceid)

    print( deviceid + ' has connected')

    
    # make sure there's an entry for this device in the messageQueues dictionary

    if deviceid not in messageQueues:
        messageQueues[deviceid] = collections.deque(maxlen=50)


def fetch_from_youtube(url, userid):

    """ Fetches the reuquested audio from Youtube, and trims any leading or
    trailing silence from it via ffmpeg. This uses the python module youtube-dl.

    Args:
        url (string): the url to fetch
        userid (string): the device asking for this
    Returns:
        string: the final file name of the retrieved and trimmed audio.
    """

    # this function runs out-of-process from the main serving thread, so
    # send an update to the client.
    post_status_message(userid, 0.05, "Asking Youtube for audio...")

    # download the file (audio only) at the highest quality and save it in /tmp
    try:

        tmpfile = tempfile.gettempdir() + '/' + userid + '.tmp'

        cmd = ['yt-dlp', '--write-info-json', '-x', '--audio-format', 'wav', 
               '-f', 'bestaudio', '--no-playlist', '-o', tmpfile, url]

        # cmd = ['yt-dlp', '--write-info-json', '-x', '--audio-format', 'mp3', 
        #        '--no-playlist', '-o', tmpfile, url]

        result = [line.decode(encoding="utf-8") for line in subprocess.check_output(cmd).splitlines()]

        print("###### yt-dlp output ######\n\n[{}]".format(result))

    except subprocess.CalledProcessError as e:

        post_status_message(userid, 1, "Failed to download the audio from Youtube. Check the logs!")

        raise IOError("Failed to download the audio from Youtube. Please check you're running the latest "
                      "version (latest available at `https://github.com/yt-dlp/yt-dlp`)")

    fn = ":".join(result[-2].split(":")[1:])[1:]

    if os.path.exists(fn) == False:
        fn = tmpfile

    # # save as ogg
    # of = tempfile.gettempdir() + '/' + userid + '.ogg'

    # os.rename( fn, of )
    # return of

    return fn

def fetch_from_local(fn, userid):
    """ Trim and prepare an audio file uploaded from the user

    Args:
        fn (string): the filename that was uploaded
        userid (string): the client asking for this

    Returns:
        string: the file name of the final prepard file
    """

    # trim silence from the ends and save as ogg
    of = tempfile.gettempdir() + '/' + userid + '.ogg'

    post_status_message(userid, 0.1, "Saving as .ogg file...")

    subprocess.run(['ffmpeg', '-y', '-i', fn, of],
                    stderr=subprocess.PIPE, stdout=subprocess.PIPE)

    # delete the uploaded file
    os.remove(fn)

    # return the name of the trimmed file
    return of

@app.route('/healthcheck')
def healthcheck():
    """ Use /healthcheck if you put this server behind a load balancer
    that does HTTP health checks

    Returns:
        srtring: returns an HTTP 200 "OK"
    """
    return "OK"

@app.route('/')
def index():
    """ Return the main index page

    Returns:
        flask.Response: a redirect to the main index page
    """

    return redirect_https('static', 'index.html')

@app.route('/favicon.ico')
def icon():
    """ Return the application icon

    Returns:
        flask.Response: a redirect to the application icon
    """

    return redirect_https('static', 'favicon.ico')

@app.route('/mobile-icon.png')
def mobile_icon():
    """ Return the application icon suitable for an iOS desktop

    Returns:
        flask.Response: a redirect to the application icon
    """

    return redirect_https('static', 'mobile-icon.png')

@app.route('/icon.png')
def png():
    """ Return the application image for the navbar UI

    Returns:
        flask.Response: a redirect to the image
    """

    return redirect_https('static', 'icon.png')

@app.route('/fetch_url')
def fetch_url():

    """ Starts the background processing of the Youtube fetch
    request.

    Returns:
        flask.Response: If this is called from the browser address bar, then it will
                        redirect back to the index page.
    """

    clusters = 0

    url = request.args['url']

    if 'clusters' in request.args:
        clusters = int(request.args['clusters'])

    useCache = True

    if 'useCache' in request.args:
        useCache = bool(int(request.args['useCache']))

    deviceid = get_userid()

    print( deviceid, 'asked for:', url, 'with', clusters, 'clusters and useCache of', useCache)

    # if there is already a proc entry for this client, then
    # kill it if it's running.

    proc = None

    if deviceid in procMap:
        proc = procMap[deviceid]

    if proc != None and proc.is_alive():
        print('!!!!!! killing', proc.pid, '!!!!!')
        proc.terminate()

    # start the main audio processing proc and save a pointer to it
    # for this client

    procMap[deviceid] = Process(target=process_audio, args=(url, deviceid, False, clusters, useCache))
    procMap[deviceid].start()

    return index()

@app.route('/cancel_fetch')
def cancel_fetch():
    """ Cancels the current audio processing

    Returns:
        flask.Response: HTTP 200 OK
    """

    deviceid = get_userid()
    print('cancelling work for', deviceid)

    # if we're not processing for this client already, just return
    if deviceid not in procMap:
        return 'OK'

    # otherwise, kill the running process if it's still running
    proc = procMap[deviceid]

    if proc != None and proc.is_alive():
        print('!!!!!! killing', proc.pid, '!!!!!')
        proc.terminate()

    # return
    return 'OK'

def post_status_message( userid, percentage, message ):
    """ The main audio processing is done outside of the main thread. From
    time to time during that processing, we will want to post an update to
    the client about what's going on. To do this, we call back into the
    /relay endpoint. That enpoint exists in the main web thread and can send
    messages via scoket.io.

    Args:
        userid (string): the client id to whom to send the message
        percentage (float): the precentage complete
        message (string): the text update
    """

    payload = json.dumps({'percentage': percentage, 'message':message})

    requests.get(
        'http://localhost:8000/relay',
        params={'namespace': userid, 'event':'status', 'message': payload}
    )

def process_audio(url, userid, isupload=False, clusters=0, useCache=True):
    """ The main processing for the audio is done here. It makes heavy use of the
    InfiniteJukebox class (https://github.com/drensin/Remixatron).

    Args:
        url (string): the URL to fetch or the file uploaded
        userid (string): the id of the requesting client
        isupload (bool, optional): Is this processing for uploaded audio (True)
                                   or Youtube audio (False). Defaults to False.
    """

    fn = ""

    if isupload == False:
        fn = fetch_from_youtube(url,userid)
    else:
        fn = fetch_from_local(url, userid)

    print('fetch complete')

    # The constructor of the InfiniteJukebox class takes a callback to which to
    # post update messages. We define that callback here so that it has access
    # to the userid variable. This uses Python 'variable capture'.
    # (See https://bit.ly/2YOKm16 for more about how this works)

    def remixatron_callback(percentage, message):
        print( str(round(percentage * 100,0)) + "%: " + message )
        post_status_message(userid, percentage, message)

    remixatron_callback(0.1, 'Audio downloaded')

    cached_beatmap_fn = ( remixatron_dir / (urllib.parse.quote(url, safe='') + '.beatmap.bz2') )

    beats = None
    play_vector = None

    has_cached_beatmap = os.path.isfile(cached_beatmap_fn)

    if ( has_cached_beatmap == False) or (useCache == False):

        # all of the core analytics and processing is done in this call
        jukebox = InfiniteJukebox(fn, clusters=clusters,
                                  progress_callback=remixatron_callback,
                                  start_beat=0, do_async=False)


        with open(tempfile.gettempdir() + '/' + userid + '.clusterscores', 'w') as f:
            f.write(json.dumps(jukebox.cluster_ratio_log))

        beats = jukebox.beats
        play_vector = jukebox.play_vector

        def skip_encoder(o):
            return ''

        with bz2.open(cached_beatmap_fn, 'wb') as f:
            f.write(json.dumps(jukebox.beats, default=skip_encoder).encode('utf-8'))

    else:

        print("Reading beatmap from disk.")

        # if there's a saved beats cache on disk, load it
        if has_cached_beatmap == True:          
            with bz2.open(cached_beatmap_fn, 'rb') as f:
                beats = json.load(f)

        # pass the beat cache into the constructor. The code will use the existing
        # beat cache instead of running a high precision beat finding process. This
        # will save ~60s of processing.

        jukebox = InfiniteJukebox(fn, clusters=clusters,
                                    progress_callback=remixatron_callback,
                                    start_beat=0, do_async=False, starting_beat_cache=beats)
        if clusters == 0:
            with open(tempfile.gettempdir() + '/' + userid + '.clusterscores', 'w') as f:
                f.write(json.dumps(jukebox.cluster_ratio_log))

        beats = jukebox.beats
        play_vector = jukebox.play_vector

    # save off a dictionary of all the beats of the song. We care about the id, when the
    # beat starts, how long it lasts, to which segment and cluster it belongs, and which
    # other beats (jump candidates) it might make sense to jump.
    #
    # See https://github.com/drensin/Remixatron for a fuller explanation of these terms.

    beatmap = []

    for beat in beats:
        b = {'id':beat['id'], 'start':beat['start'] * 1000.00, 'duration':beat['duration'] * 1000.00,
             'segment':beat['segment'], 'cluster':beat['cluster'], 'jump_candidate':beat['jump_candidates']}

        beatmap.append(b)

    with open(tempfile.gettempdir() + '/' + userid + '.beatmap', 'w') as f:
        f.write(json.dumps(beatmap))

    # save off a 1024 * 1024 vector of beats to play. This is the random(ish)ly
    # generated play path through the song.

    with open(tempfile.gettempdir() + '/' + userid + '.playvector', 'w') as f:
        f.write(json.dumps(play_vector))

    # signal the client that we're done processing

    ready_msg = {'message':'ready'}

    requests.get(
        'http://localhost:8000/relay',
        params={'namespace': userid, 'event':'ready', 'message': json.dumps(ready_msg)}
    )

@app.route('/relay')
def relay():

    """ The audio processing sub-process will need to send messages back to the client. In
    order to use socket.io, however, you have to be in the main Flask context. So, the process
    will call this endpoint and this function will then use socket.io to pass the message along
    to the client.

    Returns:
        flask.Response: HTTP 200 OK
    """

    namespace = request.args['namespace']
    message = request.args['message']
    event_name = request.args['event']

    # get the message queue for this client
    q = messageQueues[namespace]

    # compute the next message id.
    id = 0

    if len(q) > 0:
        id = max([i['id'] for i in q]) + 1

    # append this message to the queue
    q.append({'id': id, 'event': event_name, 'message':message})

    # finally, send the message to the client, making sure to save
    # off the new message id.

    j = {'id':id, 'message':message}

    if event_name == 'status':
        j = json.loads(message)
        j['id'] = id

    socketio.emit(event_name, json.dumps(j), to=namespace)
    return 'OK', [('Content-Type', 'text/plain')]

@app.route('/getQueue')
def getQueue():
    """ The client will sometimes get disconnected and need to reconnect. When
    that happens it will ask for the current queue of messages it has received.
    The client keeps the id of the last message it processed locally, so we just
    return the entire queue (which is only 50 deep -- or 400 bytes after compression)

    Returns:
        string: json object represting the message queue
    """

    q = messageQueues[get_userid()]

    items = [i for i in q]

    return json.dumps(items), [('Content-Type', 'application/json'),
                               ('Cache-Control', 'no-store')]

@app.route('/beatmap')
def get_beatmap():

    """ Return the beatmap for this audio. See the process_audio() function
    for more info about this.

    Returns:
        string: JSON of the beatmap
    """

    json = ""

    with open(tempfile.gettempdir() + '/' + get_userid() + '.beatmap', 'r') as f:
        json = f.readlines()

    return json[0], [('Content-Type', 'application/json'),
                     ('Cache-Control', 'no-store')]

@app.route('/playvector')
def get_playvector():

    """ Return the play vector for this audio. See the process_audio() function
    for more info about this.

    Returns:
        string: JSON of the play vector
    """

    json = ""

    with open(tempfile.gettempdir() + '/' + get_userid() + '.playvector', 'r') as f:
        json = f.readlines()

    return json[0], [('Content-Type', 'application/json'),
                     ('Cache-Control', 'no-store')]

@app.route('/getaudio')
def get_audio():
    """ Sends to the client the audio to play for this song

    Returns:
        flask.Response: the audio file to play
    """

    input_file = tempfile.gettempdir() + '/' + get_userid() + '.wav'
    output_file = tempfile.gettempdir() + '/' + get_userid() + '.mp3'

    cmd = ['ffmpeg', '-i', input_file, '-b:a', '192K', output_file]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return send_from_directory(tempfile.gettempdir() + '/', get_userid() + '.mp3', cache_timeout=0)

@app.route('/trackinfo')
def get_trackinfo():

    """ Return the tack information about this audio. The key items are the
    URL (or file name) of the audio, a url to a tumbnail image to display, and
    the title of the track.

    Returns:
        flask.Response: JSON of the track info
    """

    jsonStr = ""

    with open(tempfile.gettempdir() + '/' + get_userid() + '.tmp.info.json', 'r') as f:
        jsonStr = f.readlines()

    json_data = json.loads(jsonStr[0])

    title = json_data['title']

    if title[0].islower():
        title = title.title()
        json_data['title'] = title

    return json.dumps(json_data), [('Content-Type', 'application/json'),
                     ('Cache-Control', 'no-store')]

@app.route('/lastclusterscores')
def get_lastClusterScores():

    """ Return the scores from the last optimal cluster finding exercise.

    Returns:
        flask.Response: Formatted string
    """

    jsonStr = ""

    with open(tempfile.gettempdir() + '/' + get_userid() + '.clusterscores', 'r') as f:
        jsonStr = f.readlines()

    json_data = json.loads(jsonStr[0])

    cluster_ratio_map = json_data['cluster_scores']

    output_string = ""

    for cr in cluster_ratio_map:
        c, sa, msl, r, cs = cr
        msg = "{:<10} {:<10} {:<15} {:<8} {:<15}".format(c,sa,msl,r,cs)
        output_string += msg + os.linesep

    output_string += os.linesep
    output_string += "Chosen Cluster Size: {}".format(json_data['best_cluster_size'])

    return output_string, [('Content-Type', 'application/json'),
                     ('Cache-Control', 'no-store')]

def loadGlobalBookmarks() -> str:
    """ Return any bookmarks stored on the server. They are a simple ordered JSON
        array stored in /tmp/remixatron.global.bookmarks

    Returns:
        JSON string of the track info
    """

    j = ""

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

@app.route('/globalBookmarks')
def get_globalBookmarks():

    """ Return any bookmarks stored on the server. They are a simple ordered JSON
        array stored in /tmp/remixatron.global.bookmarks

    Returns:
        flask.Response: JSON of global bookmarks
    """

    return loadGlobalBookmarks(), [('Content-Type', 'application/json'),
                                   ('Cache-Control', 'no-store')]

@app.route('/addGlobalBookmark', methods=['POST'])
def addGlobalBookmark():
    """ Adds an item to the Global Bookmarks. Sorts the bookmarks before writing
    """

    json_item = request.get_json()

    print(json.dumps(json_item))

    bookmarks = json.loads(loadGlobalBookmarks())
    bookmarks.append(json_item)
    bookmarks.sort(key=lambda x: x["title"])

    saveGlobalBookmarks(json.dumps(bookmarks, indent=3, sort_keys=True))

    return json.dumps(bookmarks), [('Content-Type', 'application/json'), 
                                   ('Cache-Control', 'no-store')]

@app.route('/deleteGlobalBookmark')
def deleteGlobalBookmark():

    title = request.args['title']

    print( "Deleting: " + title)

    bookmarks = json.loads(loadGlobalBookmarks())
    filtered_bookmarks = [i for i in bookmarks if not (i['title'] == title)] 
    saveGlobalBookmarks(json.dumps(filtered_bookmarks, indent=3, sort_keys=True))

    return "OK"

@app.route('/whoami')
def whoami():

    """ Called first by the client, this sets up the device id and
    the message and process queues.

    Returns:
        flask.Response: the device id for the client
    """

    # if this client doesn't already have a device id configured and
    # saved in a cookie, then set one up.

    deviceid = get_userid()

    if deviceid == None:
        deviceid = secrets.token_urlsafe(16)

        resp = make_response(deviceid,200)
        resp.set_cookie('deviceid',deviceid,max_age=31536000)

        return resp

    print( deviceid + ' has connected')

    # make sure the message queues and process queue is setup
    # for this client.

    if deviceid not in messageQueues:
        messageQueues[deviceid] = collections.deque(maxlen=50)

    if deviceid not in procMap:
        procMap[deviceid] = None

    
    return deviceid

@app.route('/cleanup')
def cleanup():

    """ When the client has retrieved everything it needs, then it will
    call /cleanup to get rid of any of the intermediate files.

    Returns:
        flask.Response: HTTP 200 OK
    """

    fileList = glob.glob(tempfile.gettempdir() + '/' + get_userid() + '*')

    for file in fileList:

        if str(file).endswith('.clusterscores'):
            continue

        os.remove(file)

    messageQueues[get_userid()].clear()

    return "OK"

@app.route('/upload_audio', methods=['POST'])
def upload_audio():

    """ The client uploads audio to process to this endpoint

    Returns:
        flask.Response: a redirect to the index.html page
    """

    file = request.files['file']
    deviceid = get_userid()

    # save the uploaded file

    of = tempfile.gettempdir() + '/' +deviceid + '.tmp'
    file.save(of)

    print( deviceid + ' uploaded: ' + file.filename )

    # save off the track info

    with open(tempfile.gettempdir() + '/' + deviceid + '.tmp.info.json', 'w') as f:
        j = {'title': file.filename, 'thumbnail':'/static/favicon.ico'}
        f.write(json.dumps(j))

    # if there's already an audio processing subproc running for this client,
    # then kill it.

    proc = procMap[deviceid]

    if proc != None and proc.is_alive():
        print('!!!!!! killing', proc.pid, '!!!!!')
        proc.terminate()

    # kick of a proc to process the audio.

    procMap[deviceid] = Process(target=process_audio, args=(of, deviceid, True))
    procMap[deviceid].start()

    return index()

if __name__ == '__main__':

    # The main thread. Listens on any IP address and port 8000

    compress.init_app(app)
    socketio.run(app, host="0.0.0.0", port=8000)
