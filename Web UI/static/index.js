/**
 * Remixatron Web UI js - (c) 2019 - Dave Rensin - dave@rensin.com
 *
 * Client code for web ui.
 */

 // global vars
var mychannel = null;

var userID = null;

var beatmap = null;
var playvector = null;
var trackinfo = null;

var sound = null;
var sndIndex = 0;
var timerPlaybackID = null;

var clusters = 0;
var segments = 0;

var colorMap = [];
var lastMessageID = -1;

// catch and handle window resize events

$(window).resize( ()=> {
    set_dropdown_width();
});

// set the dropdown width
set_dropdown_width();

// start the session by finding out our device id and setting up our
// socket.io channels.

$.get('/whoami', (data) => {

    console.log('I am:' + data);
    userID = data;

    mychannel = io('/' + data);
    mychannel.on('status', on_status_message);
    mychannel.on('ready', on_ready_message);
    mychannel.on('reconnect', on_reconnect)

});

// load the 'Stars' dropdown with our saved favorites
//load_history_dropdown();

$.get('/globalBookmarks',on_globalBookmarks)

/**
 * Sets the width of the favorites dropdown. Is called when the
 * page loads or resizes.
 */
function set_dropdown_width() {

    if ( window.screen.availWidth > 500 ) {
        $('#ddhistory').css('width', 'auto');
    }else{
        $('#ddhistory').css('width', window.screen.availWidth - 20);
    }

}

/**
 * Shows the progress bar and displays the appropriate percentage and
 * progress.
 *
 * @param {int} percentage
 * @param {string} message
 */
function set_progress_bar(percentage, message) {
    $('#progress-modal').modal('show');
    $('#progress-message').text(message);
    $('#progress').text(percentage.toFixed() + '%');
    $('#progress').css('width', String(percentage) + '%');
}

/**
 * Copies to the clipboard a ready-made URL you can bookmark to
 * quickly relaunch the client and re-process the current song.
 *
 * @param {string} url
 */
async function copyUrlToClipboard(url) {
    try {
      await navigator.clipboard.writeText(url);
    } catch (err) {
    }
}

/**
 * Called when the user manually sets the clustering options
 */
function advancedFetch() {

    var clusters = parseInt( $('#clusterSize').val() );

    $('.navbar-collapse').collapse('hide');

    fetchURL(clusters=clusters, useCache=1);
}

/**
 * This is called when the user tries to get audio from Youtube.
 */
function fetchURL(clusters = 0, useCache=1) {

    // hide the navbar
    $('.navbar-collapse').collapse('hide');

    // show the prgress modal
    $('#progress-modal').modal('show');

    // copy a fully qualified URL for this song
    // to the clipboard. Suitable for bookmarking in a
    // browser.

    var loc = $('#ytURL').val();

    var relUrl = '/fetch_url?url=' + loc + "&clusters=" + clusters + "&useCache=" + useCache;
    var newUrl = new URL(relUrl, window.location.href);

    copyUrlToClipboard(newUrl.href);

    // kick off the audio processing on the server and return

    
    $.get('/fetch_url',"url=" + encodeURIComponent(loc) + "&clusters=" + clusters + "&useCache=" + useCache);

    set_progress_bar(0, "Sending request to server...");

    return false;
}

/**
 * Called when the user wants to cancel the current audio processing
 */
function cancel_fetch() {

    // call the cancel endpoint
    $.get('/cancel_fetch');

    //re-open the navbar and hide the progress bar
    $('#toggler').click();

    $('#progress-modal').modal('hide');
}

/**
 * Called when the user selects a local file to upload for
 * processing.
 */
function upload_audio() {

    // hide the navbar and show the progress bar
    $('#toggler').click();

    $('#progress-modal').modal('show');
    set_progress_bar(0, "Uploading file...");

    // Get form
    var form = $('#upload_form')[0];

    // Create an FormData object
    var data = new FormData(form);

    // post the auido file to the server
    $.ajax({
        type: "POST",
        enctype: 'multipart/form-data',
        url: "/upload_audio",
        data: data,
        processData: false,
        contentType: false,
        cache: false,
        timeout: 600000
    });

    return false;
}

/**
 * This is called when the client loses its socket.io connection to the server
 * and successfully reconnects.
 *
 * @param {int} attempt
 */
function on_reconnect(attempt) {
    console.log("Reconnected with attempt: " + attempt);

    // get the last 50 messages posted to this client. Repost the ones
    // we haven't already handled.

    $.get('/getQueue', (q) => {

        q.forEach( (item) => {

            if (item.id > lastMessageID) {

                console.log('resending message: ' + item.id);

                var m = JSON.parse(item.message);
                m.id = item.id;

                m = JSON.stringify(m);

                if (item.event == 'status') {
                    on_status_message(m);
                    return;
                }

                if (item.event == 'ready') {
                    on_ready_message(m);
                    return;
                }
            }
        });
    });
}

/**
 * The server will send progress messages via the socket.io library as it
 * processes your audio. This function recieves those messages and displays them.
 *
 * @param {json} data
 */
function on_status_message(data) {
    data = JSON.parse(data);

    lastMessageID = data.id;
    set_progress_bar(data.percentage * 100, data.message);
}

/**
 * When the server is done processing, it will send a READY message
 * to signal that it's time to start downloading the finished files.
 *
 * @param {string} data
 */
function on_ready_message(data) {
    data = JSON.parse(data);
    lastMessageID = data.id;

    // download the beat map
    set_progress_bar(100, 'Requesting beatmap...');
    $.get('/beatmap?'+Math.random(), on_get_beatmap);
}

/**
 * Process the downloaded beatmap
 *
 * @param {json object} d
 */
function on_get_beatmap(d) {
    beatmap = d;

    // how many clusters are there?
    clusters = beatmap.reduce( (a,c) => {
        return a>c.cluster ? a : c.cluster;
    }) + 1;

    // how many segments?
    segments = beatmap.reduce( (a,c) => {
        return a>c.segment ? a : c.segment;
    }) + 1;

    // save a color map with a unique color per cluster
    for ( var i = 0; i < clusters; i++ ) {
        colorMap = colorMap.concat( rainbowStop(i/(clusters+1)) );
    }

    set_progress_bar(100, 'Requesting play vector...');

    // grab the play vector and track info information from the server.
    // There's a bug I haven't tracked down yet that can cause caching of
    // these files. To work around it, I append a random number. This is a
    // hack. When I track down the problem, I'll fix this up.

    $.get('/playvector?' + Math.random(), on_get_playvector);
    $.get('/trackinfo?' + Math.random(), on_get_trackinfo);
}

/**
 * Called after the play vector is received and we're ready to start
 * playing the audio
 *
 * @param {json object} d
 */
function on_get_playvector(d) {
    playvector = d;
    set_progress_bar(100, 'Starting playback...')

    // 100ms, start playing the audio
    window.setTimeout(playback, 100);
}

/**
 * Called after we receive the track info. We just save it off.
 *
 * @param {json object} d
 */
function on_get_trackinfo(d) {
    trackinfo = d;
}

/**
 * Checks to see if we've already saved this audio in our star/favorites/history. Returns a boolean.
 *
 * @param {string} title
 */
function is_in_history(title) {

    // if the local storage hasn't been setup yet..
    if (localStorage.ytHistory == undefined){
        localStorage.ytHistory ="[]";
    }

    // de-serialize the list from local storage
    var history = JSON.parse(localStorage.ytHistory);

    // get the item if it's already there
    var item = history.find( (e) => {
        return e.title == title;
    });

    // if we found the item, return true. Otherwise, false
    if (item != undefined){
        return true;
    }else{
        return false;
    }

}

/**
 * Add the currently playing item to the star/favorite/history. Right now,
 * I don't support adding uploaded files. I might fix that later.
 */
function add_to_history() {

    // if this is a local file, error out.
    if (trackinfo.thumbnail[0] == '/') {
        alert("Sorry. You can't add uploaded files to favorites.")
        return;
    }

    // if the local storage isn't set up, fix that.
    if (localStorage.ytHistory == undefined){
        localStorage.ytHistory = "[]";
    }

    // de-serialize the list
    var history = JSON.parse(localStorage.ytHistory);

    // if the item is already in the list, then make sure the Now Playing star
    // is filled in and return

    if (is_in_history(trackinfo.title)){
        $('#starIcon').text('star');
        console.log('already in history');
        return;
    }

    // otherwise.. Construct the object to save
    item = {"title": trackinfo.title, "thumbnail": trackinfo.thumbnail,
            "url": $('#ytURL').val(), "clusters": clusters};

    $.ajax({
        url:'/addGlobalBookmark',
        type:"POST",
        data:JSON.stringify(item),
        contentType:"application/json; charset=utf-8",
        dataType:"json",
    }).done(function(data){
        on_globalBookmarks(data)
    })            

    $('#starIcon').text('star');

}

/**
 * Delete the currently playing item from the save list
 */
function remove_from_history() {

    $.get('/deleteGlobalBookmark', { title: trackinfo.title }, function(data) {
        $.get('/globalBookmarks',on_globalBookmarks)
    });

    $('#starIcon').text('star_border');

}

/**
 * Gets called when you tap the Now Playing star. Toggles the
 * currently playing item in/out of the save list.
 */
function toggle_to_history() {

    if (is_in_history(trackinfo.title)){
        remove_from_history();
    }else{
        add_to_history();
    }
}

/**
 * Called when /globalBookmarks returns
 */
function on_globalBookmarks(d){

    // sort the list alphabetically

    d = d.sort( (x,y) => {return x.title.localeCompare(y.title);} );

    // save it to local storage
    localStorage.ytHistory = JSON.stringify(d);

    load_history_dropdown();
}

/**
 * Loads the save list dropdown from local storage
 */
function load_history_dropdown() {

    // if localstorage isn't setup yet, return
    if (localStorage.ytHistory == undefined) {
        return;
    }

    // clear the current dropdown
    $('#ddhistory').empty();

    // grab the list from local storage
    var history = JSON.parse(localStorage.ytHistory);

    // add each item to the drop down, making sure to attach each
    // item to the on_history_select() callback.

    history.forEach( (item) => {
        var idx = history.indexOf(item);
        $('#ddhistory').append('<a class="dropdown-item" href="#" onclick="on_history_select(' + idx + ');">' + item.title + '</a>');
    });
}

/**
 * Called when an item is selected from the save list
 *
 * @param {int} idx
 */
function on_history_select(idx) {

    // load the save list from storage
    var history = JSON.parse(localStorage.ytHistory);

    // grab the item
    var item = history[idx];

    // populate the UI with the correct URL
    $('#ytURL').val(item.url);

    console.log(history[idx]);

    var c = 0;

    if (item.clusters != undefined){
        c = item.clusters;
    }

    // start the audio processing on the server for the URL
    fetchURL(clusters = c, useCache = 1);
}

/**
 * Called when a local file is selected for upload
 */
function on_file_selected() {

    // if the user canceled, just return.
    if ( $('#fcu')[0].files.length == 0 ){
        return;
    }

    // update the display label with the file name
    $('#fcuLabel').text($('#fcu')[0].files[0].name);

    // upload the file and start the audio processing
    upload_audio();
}

/**
 * Show the Now Playing panel
 */
function showtoast() {

    // set the Now Playing image to the thumbnail
    $('#npimg').attr('src', trackinfo.thumbnail);

    // set the display text to the title
    $('#nptext').text(trackinfo.title);

    // set the clusters counter
    $('#clusterSize').val(clusters);

    // display the panel
    $('.toast').toast('show');

    // if the item is in our save list, then color in the
    // Now Playing star. Otherwise, leave it outlined.

    if (is_in_history(trackinfo.title) == true ){
        $('#starIcon').text('star');
    }else{
        $('#starIcon').text('star_border');
    }
}

/**
 * Toggles playback of the current audio OR starts playback of new
 * audio
 *
 * @param {bool} createNew
 */
function playback(createNew = true){

    // if we already have created a Howler.js soud object...
    if ( sound != null && createNew == false) {

        // and if it's already playing, pause it and stop
        // any more beats from playing..
        if ( sound.playing() == true ) {
            sound.pause();
            clearTimeout(timerPlaybackID);
            $('#playIcon').text('play_circle_outline');
            return;
        }else{
            // if it wasn't already playing, then un-pause it and start the playback events
            sound.play(String(sndIndex));
            timerPlaybackID = setTimeout(onSoundEnd, beatmap[playvector[sndIndex].beat].duration);
            $('#playIcon').text('pause_circle_outline');
            return;
        }
    }

    // if we're creating a new playback then clear out the event timer that plays
    // each beat.

    clearTimeout(timerPlaybackID);

    // change the playback icon to reflect that we're not playing anymore
    $('#playIcon').text('play_circle_outline');

    // if we have a sound object already, then stop it.
    if ( sound != null ) {
        sound.stop();
    }

    // In this section, we create a new Howler.js sound object. Each beat will be loaded as
    // an audio sprite so that we can quickly jump between them. When Howler finishes loading
    // all the audio, it will call onSoundLoad().

    sound = null;
    var spritedef = {};
    sndIndex = 0;

    beatmap.forEach(beat => {

        var start = beat.start;
        var duration = beat.duration;

        spritedef[beat.id + 1] = [start, duration];
    });

    sound = new Howl({
        src: ['/getaudio?' + Math.random()],
        format: ['ogg'],
        sprite: spritedef,
        onload: onSoundLoad,
        onplay: onSoundPlay
    });
}

/**
 * One Howler has finished loading the audio, it's time to start playback...
 */
function onSoundLoad() {

    // play the first sprite (the first beat of the song) and set
    // a timer to play the next beat in the play vector. The duration
    // of the time is the duration (in milliseconds) of the beat we just
    // started playing.

    sndIndex = 0;
    sound.play('1');
    timerPlaybackID = setTimeout(onSoundEnd, beatmap[0].duration);

    // hide the progress bar and show the Now Playing panel
    $('#progress-modal').modal('hide');
    showtoast();

    // tell the server it's safe to cleanup the intermediate audio
    // processing files and change the Now Playing icon to reflect that
    // the audio is playing.

    $.get('/cleanup');

    $('#playIcon').text('pause_circle_outline');
}

/**
 * This is called after we've waited for the current beat to play and we're
 * ready to play the next beat.
 */
function onSoundEnd() {

    // look in the play vector to see which beat we need to play next
    sndIndex = sndIndex + 1;

    var toplay = playvector[sndIndex].beat + 1;

    // play that beat and reset the play timer
    id = sound.play(String(toplay));

    timerPlaybackID = setTimeout(onSoundEnd, beatmap[toplay - 1].duration);

}

/**
 * This is called after each beat is played. It will draw the UI for the next
 * beat to play.
 */
function onSoundPlay() {
    drawViz();
}

/**
 * Computes h colors that are evenly spread across the rainbow from each other. This is
 * used when building the color map to draw the on-screen beatmap.
 *
 * @param {int} h
 */
function rainbowStop(h)
{
  let f= (n,k=(n+h*12)%12) => .5-.5*Math.max(Math.min(k-3,9-k,1),-1);
  let rgb2hex = (r,g,b) => "#"+[r,g,b].map(x=>Math.round(x*255).toString(16).padStart(2,0)).join('');
  return ( rgb2hex(f(0), f(8), f(4)) );
}

/**
 * Draw the UI for the current beat
 */
function drawViz() {

    // set/get the right height/width
    var width = screen.availWidth;
    var height = 400;

    // set the canvas height/width
    $('#viz').prop('width', width);
    $('#viz').prop('height', height);

    // get the draw context and figure out how many pixels wide
    // to draw each beat.
    ctx = $('#viz')[0].getContext('2d');

    var xOffset = 1;
    var beatWidth = (width - 5) / beatmap.length;

    ctx.font = '12px serif';

    // clear the canvas
    ctx.clearRect( 0,0, width, height );

    // draw the timeline

    var timelineHeight = 100;
    var timelineY = (height / 2) - (timelineHeight / 2);

    // paint the stats
    ctx.fillText( "Pos: " + String(sndIndex) +
                  "   Beats: " + String(beatmap.length) +
                  "   Clusters: " + String(clusters) +
                  "   Segments: " + String(segments) +
                  "   Ratio: " + (segments / clusters).toFixed(3),
                  0, timelineY + timelineHeight + 23);

    // get the current X and beat
    var currentX = (playvector[sndIndex].beat * beatWidth) + xOffset;
    var currentBeat = beatmap[playvector[sndIndex].beat];

    var segmentColorMap = ['#C2C2C2', '#FFFFFF'];

    // first, paint a rectanlge with a shadow
    ctx.strokeRect(0, timelineY, width - 5, timelineHeight);
    ctx.shadowColor = 'Black';
    ctx.shadowOffsetX = 3;
    ctx.shadowOffsetY = 3;
    ctx.shadowBlur = 5;

    // next, fill it in with white
    ctx.strokeStyle = 'Black';
    ctx.fillStyle = 'White';
    ctx.lineWidth = 1;
    ctx.fillRect(0, timelineY, width - 5, timelineHeight);

    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
    ctx.shadowBlur = 0;

    // paint each beat in the right spot and use its appropriate cluster color
    for (var i =0; i<beatmap.length; i++) {

        // is this beat in the same segment as the currently playing beat?
        var isSameSegment = beatmap[i].segment == currentBeat.segment;

        // is this beat in the list of beats that we could jump to right now?
        var isJumpCandidate = currentBeat.jump_candidate.includes(i);

        // if either of those things is true, then we'll draw this beat at full color.
        // Otherwise, we'll draw it very light

        if (isSameSegment || isJumpCandidate) {
            ctx.globalAlpha = 1.0;
        }else{
            ctx.globalAlpha = .2;
        }

        // draw this beat on the screen at the right location and with the correct
        // width

        ctx.fillStyle = colorMap[beatmap[i].cluster];
        ctx.strokeStyle = ctx.fillStyle;
        ctx.fillRect( (i * beatWidth) + xOffset, timelineY + 1,
                       ctx.globalAlpha < 1.0 ? beatWidth : beatWidth + 2, timelineHeight - 2);
    }

    // draw the jump candidate arcs -- this section draws light grey arcs from the currently
    // playing beat to all the places it might be allowed to jump (according to the beatmap).
    // This code alternates drawing each arc over or under the timeline.

    var over = true;

    ctx.globalAlpha = 1.0;

    currentBeat.jump_candidate.forEach( (c) => {
        ctx.beginPath();

        ctx.strokeStyle = '#CFCFCF';
        ctx.lineWidth = beatWidth < 2 ? 1 : beatWidth / 2;

        nextX = (c * beatWidth) + xOffset;

        if ( over == true ) {

            ctx.moveTo(currentX + (ctx.lineWidth / 2), timelineY - 5);

            ctx.arcTo( (nextX + currentX) / 2, 0,
                       nextX, timelineY,
                       Math.abs(nextX - currentX) * .5);

            ctx.lineTo(nextX + (beatWidth /2), timelineY - 2);

        }else{

            ctx.moveTo(currentX + (ctx.lineWidth / 2), timelineY + timelineHeight + 10);

            ctx.arcTo( (nextX + currentX) / 2, height,
                       nextX, timelineY + timelineHeight,
                       Math.abs(nextX - currentX) * .5);

            ctx.lineTo(nextX, timelineY + timelineHeight + 4);
        }


        ctx.stroke();
        over = !over;
    });

    // draw the now playing marker

    ctx.shadowColor = 'Black';
    ctx.shadowOffsetX = 3;
    ctx.shadowOffsetY = 3;
    ctx.shadowBlur = 5;

    ctx.fillStyle = 'Black';
    ctx.strokeStyle = 'Black';
    ctx.fillRect(currentX, timelineY - 5, beatWidth, timelineHeight + 15);

    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
    ctx.shadowBlur = 0;

    // compute how many beats until we want to jump (or '!!' if we're late)
    var leftInSeq =  playvector[sndIndex].seq_len - playvector[sndIndex].seq_pos;

    var str = '-' + String(leftInSeq).padStart(2, '0')

    if (leftInSeq == 0) {
        str = '!!!'
    }

    // paint that number to the left of the Now Playing marker.
    ctx.fillText( str, currentX - 20, timelineY + (timelineHeight / 2) );
}