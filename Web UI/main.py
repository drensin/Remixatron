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
import glob
import json
import numpy as np
import os
import requests
import secrets
import subprocess
import soundfile as sf
import sys
import urllib.parse

from flask import Flask, current_app, g, make_response, redirect, request, send_from_directory, session, url_for
from flask_compress import Compress
from flask_socketio import SocketIO, emit, send

from multiprocessing import Process

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

# the cors.cfg file defines which domains will be trusted for connections. The
# default entry of localhost:8000 will be fine if you are running the web
# browser on the same host as this server. Otherwise, you'll have to modify
# accordingly. See the README.MD for this project for more info.

if os.path.isfile('cors.cfg'):
    with open('cors.cfg') as cors_file:
        origins = json.load(cors_file)
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

    result = subprocess.run(['youtube-dl', '--write-info-json', '-x', '--audio-format', 'best', '--audio-quality', '0',
                                           '-o','/tmp/' + userid + '.tmp', url], stdout=subprocess.PIPE)
    print(result.stdout.decode('utf-8'))

    fn = result.stdout.decode('utf-8').split(':')[-1].split('\n')[0][1:]

    # trim silence from the ends and save as ogg
    of = '/tmp/' + userid + '.ogg'

    post_status_message(userid, 0.1, "Trimming silence from the ends...")

    filter = "silenceremove=start_periods=1:start_duration=1:start_threshold=-60dB:detection=peak,aformat=dblp,areverse,silenceremove=start_periods=1:start_duration=1:start_threshold=-60dB:detection=peak,aformat=dblp,areverse"
    result = subprocess.run(['ffmpeg', '-y', '-i', fn, '-af', filter, of],
                            stderr=subprocess.PIPE, stdout=subprocess.PIPE)

    # delete the downlaoded file because we don't need it anymore
    os.remove(fn)

    # return the name of the trimmed file
    return of

def fetch_from_local(fn, userid):
    """ Trim and prepare an audio file uploaded from the user

    Args:
        fn (string): the filename that was uploaded
        userid (string): the client asking for this

    Returns:
        string: the file name of the final prepard file
    """

    # trim silence from the ends and save as ogg
    of = '/tmp/' + userid + '.ogg'

    post_status_message(userid, 0.1, "Trimming silence from the ends...")

    filter = "silenceremove=start_periods=1:start_duration=1:start_threshold=-60dB:detection=peak,aformat=dblp,areverse,silenceremove=start_periods=1:start_duration=1:start_threshold=-60dB:detection=peak,aformat=dblp,areverse"
    result = subprocess.run(['ffmpeg', '-y', '-i', fn, '-af', filter, of],
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
        params={'namespace': '/' + userid, 'event':'status', 'message': payload}
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
        print( str(percentage * 100) + "%: " + message )
        post_status_message(userid, percentage, message)

    remixatron_callback(0.1, 'Audio downloaded')

    cached_beatmap_fn = "/tmp/" + urllib.parse.quote(url, safe='') + ".beatmap.bz2"

    beats = None
    play_vector = None

    if (os.path.isfile(cached_beatmap_fn) == False) or (useCache == False):

        # all of the core analytics and processing is done in this call

        jukebox = InfiniteJukebox(fn, clusters=clusters,
                                  progress_callback=remixatron_callback,
                                  start_beat=0, do_async=False)

        beats = jukebox.beats
        play_vector = jukebox.play_vector

        def skip_encoder(o):
            return ''

        with bz2.open(cached_beatmap_fn, 'wb') as f:
            f.write(json.dumps(jukebox.beats, default=skip_encoder).encode('utf-8'))

    else:

        print("Reading beatmap from disk.")

        with bz2.open(cached_beatmap_fn, 'rb') as f:
            beats = json.load(f)

        play_vector = InfiniteJukebox.CreatePlayVectorFromBeats(beats, start_beat=0)

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

    with open('/tmp/' + userid + '.beatmap', 'w') as f:
        f.write(json.dumps(beatmap))

    # save off a 1024 * 1024 vector of beats to play. This is the random(ish)ly
    # generated play path through the song.

    with open('/tmp/' + userid + '.playvector', 'w') as f:
        f.write(json.dumps(play_vector))

    # signal the client that we're done processing

    ready_msg = {'message':'ready'}

    requests.get(
        'http://localhost:8000/relay',
        params={'namespace': '/' + userid, 'event':'ready', 'message': json.dumps(ready_msg)}
    )

@app.route('/relay')
def relay():

    """ The audio processing sub-process will need to send messages back to the client. In
    order to use socket.io, however, you have to be in the main Flaks context. So, the process
    will call this endpoint and this function will then use socket.io to pass the message along
    to the client.

    Returns:
        flask.Response: HTTP 200 OK
    """

    namespace = request.args['namespace']
    message = request.args['message']
    event_name = request.args['event']

    # get the message queue for this client
    q = messageQueues[namespace[1:]]

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

    socketio.emit(event_name, json.dumps(j), namespace=namespace)
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

    items = []

    for i in q:
        items.append(i)

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

    with open('/tmp/' + get_userid() + '.beatmap', 'r') as f:
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

    with open('/tmp/' + get_userid() + '.playvector', 'r') as f:
        json = f.readlines()

    return json[0], [('Content-Type', 'application/json'),
                     ('Cache-Control', 'no-store')]

@app.route('/getaudio')
def get_audio():
    """ Sends to the client the audio to play for this song

    Returns:
        flask.Response: the audio file to play
    """

    return send_from_directory('/tmp', get_userid() + '.ogg', cache_timeout=0)

@app.route('/trackinfo')
def get_trackinfo():

    """ Return the tack information about this audio. The key items are the
    URL (or file name) of the audio, a url to a tumbnail image to display, and
    the title of the track.

    Returns:
        flask.Response: JSON of the track info
    """

    json = ""

    with open('/tmp/' + get_userid() + '.tmp.info.json', 'r') as f:
        json = f.readlines()

    return json[0], [('Content-Type', 'application/json'),
                     ('Cache-Control', 'no-store')]

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

    fileList = glob.glob('/tmp/' + get_userid() + '*')

    for file in fileList:
        os.remove(file)

    messageQueues[get_userid()].clear()

    return "OK"

@app.route('/upload_audio', methods=['POST'])
def upload_audio():

    """ The client uploads audio to process to this endpoint

    Returns:
        falsk.Response: a redirect to the index.html page
    """

    file = request.files['file']
    deviceid = get_userid()

    # save the uploaded file

    of = '/tmp/' +deviceid + '.tmp'
    file.save(of)

    print( deviceid + ' uploaded: ' + file.filename )

    # save off the track info

    with open('/tmp/' + deviceid + '.tmp.info.json', 'w') as f:
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
