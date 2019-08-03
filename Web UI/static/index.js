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

$.get('/whoami', (data) => {

    console.log('I am:' + data);
    userID = data;

    mychannel = io('/' + data);
    mychannel.on('status', on_status_message);
    mychannel.on('ready', on_ready_message);
    mychannel.on('reconnect', on_reconnect)

});

load_history_dropdown();

function set_progress_bar(percentage, message) {
    $('#progress-modal').modal('show');
    $('#progress-message').text(message);
    $('#progress').text(String(percentage) + '%');
    $('#progress').css('width', String(percentage) + '%');
}

async function copyUrlToClipboard(url) {
    try {
      await navigator.clipboard.writeText(url);
    } catch (err) {
    }
}

function fetchURL() {

    $('#toggler').click();

    $('#progress-modal').modal('show');

    var loc = $('#ytURL').val();

    var relUrl = '/fetch_url?url=' + loc;
    var newUrl = new URL(relUrl, window.location.href);

    copyUrlToClipboard(newUrl.href);

    $.get('/fetch_url',"url=" + loc);

    set_progress_bar(0, "Sending request to server...");

    return false;
}

function cancel_fetch() {
    $.get('/cancel_fetch');

    $('#toggler').click();

    $('#progress-modal').modal('hide');
}

function upload_audio() {

    $('#toggler').click();

    $('#progress-modal').modal('show');
    set_progress_bar(0, "Uploading file...");

    // Get form
    var form = $('#upload_form')[0];

    // Create an FormData object
    var data = new FormData(form);

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
function on_reconnect(attempt) {
    console.log("Reconnected with attempt: " + attempt);

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

function on_status_message(data) {
    data = JSON.parse(data);

    lastMessageID = data.id;
    set_progress_bar(data.percentage * 100, data.message);
}

function on_ready_message(data) {
    data = JSON.parse(data);
    lastMessageID = data.id;

    set_progress_bar(100, 'Requesting beatmap...');
    $.get('/beatmap?'+Math.random(), on_get_beatmap);
}

function on_get_beatmap(d) {
    beatmap = d;

    // fix durations

    beatmap.forEach( (beat) => {
        if (beat.id < (beatmap.length -1 )) {
            beat.duration = beatmap[beat.id + 1].start - beat.start + 0.0;
        }
    });

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
        colorMap = colorMap.concat( rainbowStop(i/clusters) );
    }

    set_progress_bar(100, 'Requesting play vector...');

    $.get('/playvector?' + Math.random(), on_get_playvector);
    $.get('/trackinfo?' + Math.random(), on_get_trackinfo);
}

function on_get_playvector(d) {
    playvector = d;
    set_progress_bar(100, 'Starting playback...')
    window.setTimeout(playback, 100);
}

function on_get_trackinfo(d) {
    trackinfo = d;
}

function is_in_history(title) {

    if (localStorage.ytHistory == undefined){
        localStorage.ytHistory ="[]";
    }

    var history = JSON.parse(localStorage.ytHistory);

    var item = history.find( (e) => {
        return e.title == title;
    });

    if (item != undefined){
        return true;
    }else{
        return false;
    }

}

function add_to_history() {

    if (trackinfo.thumbnail[0] == '/') {
        alert("Sorry. You can't add uploaded files to favorites.")
        return;
    }

    if (localStorage.ytHistory == undefined){
        localStorage.ytHistory = "[]";
    }

    var history = JSON.parse(localStorage.ytHistory);

    if (is_in_history(trackinfo.title)){
        $('#starIcon').text('star');
        console.log('already in history');
        return;
    }

    item = {"title": trackinfo.title, "thumbnail": trackinfo.thumbnail, "url": $('#ytURL').val()};

    history.push(item);

    history = history.sort( (a,b) => {
        return a.title < b.title ? -1 : 1;
    });

    localStorage.ytHistory = JSON.stringify(history);

    load_history_dropdown();
    $('#starIcon').text('star');

}

function remove_from_history() {

    var history = JSON.parse(localStorage.ytHistory);

    var idx = history.findIndex( (e) => {
        return e.title == trackinfo.title;
    });

    history.splice(idx,1)

    localStorage.ytHistory = JSON.stringify(history);

    load_history_dropdown();
    $('#starIcon').text('star_border');

}

function toggle_to_history() {

    if (is_in_history(trackinfo.title)){
        remove_from_history();
    }else{
        add_to_history();
    }
}

function load_history_dropdown() {

    if (localStorage.ytHistory == undefined) {
        return;
    }

    $('#ddhistory').empty();

    var history = JSON.parse(localStorage.ytHistory);

    history.forEach( (item) => {
        var idx = history.indexOf(item);
        $('#ddhistory').append('<a class="dropdown-item" href="#" onclick="on_history_select(' + idx + ');">' + item.title + '</a>');
    });
}

function on_history_select(idx) {
    var history = JSON.parse(localStorage.ytHistory);

    var item = history[idx];

    $('#ytURL').val(item.url);

    console.log(history[idx]);

    fetchURL();
}

function on_file_selected() {

    if ( $('#fcu')[0].files.length == 0 ){
        return;
    }

    $('#fcuLabel').text($('#fcu')[0].files[0].name);

    upload_audio();
}

function showtoast() {
    $('#npimg').attr('src', trackinfo.thumbnail);
    $('#nptext').text(trackinfo.title);
    $('.toast').toast('show');

    if (is_in_history(trackinfo.title) == true ){
        $('#starIcon').text('star');
    }else{
        $('#starIcon').text('star_border');
    }
}

function playback(createNew = true){

    if ( sound != null && createNew == false) {
        if ( sound.playing() == true ) {
            sound.pause();
            clearTimeout(timerPlaybackID);
            $('#playIcon').text('play_circle_outline');
            return;
        }else{
            sound.play(String(sndIndex));
            timerPlaybackID = setTimeout(onSoundEnd, beatmap[playvector[sndIndex].beat].duration);
            $('#playIcon').text('pause_circle_outline');
            return;
        }
    }

    clearTimeout(timerPlaybackID);

    $('#playIcon').text('play_circle_outline');

    if ( sound != null ) {
        sound.stop();
    }

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
        // onend: onSoundEnd
    });
}

function onSoundLoad() {
    sndIndex = 0;
    sound.play('1');
    timerPlaybackID = setTimeout(onSoundEnd, beatmap[0].duration);

    $('#progress-modal').modal('hide');
    showtoast();

    $.get('/cleanup');

    $('#playIcon').text('pause_circle_outline');
}

function onSoundEnd() {
    sndIndex = sndIndex + 1;

    var toplay = playvector[sndIndex].beat + 1;

    id = sound.play(String(toplay));

    timerPlaybackID = setTimeout(onSoundEnd, beatmap[toplay - 1].duration);

}

function onSoundPlay() {

    // setTimeout(drawViz,0);
    drawViz();
}

function rainbowStop(h)
{
  let f= (n,k=(n+h*12)%12) => .5-.5*Math.max(Math.min(k-3,9-k,1),-1);
  let rgb2hex = (r,g,b) => "#"+[r,g,b].map(x=>Math.round(x*255).toString(16).padStart(2,0)).join('');
  return ( rgb2hex(f(0), f(8), f(4)) );
}

function drawViz() {

    var width = screen.availWidth;
    var height = 400;

    $('#viz').prop('width', width);
    $('#viz').prop('height', height);

    ctx = $('#viz')[0].getContext('2d');

    var xOffset = 1;
    var beatWidth = (width - 5) / beatmap.length;

    ctx.font = '12px serif';

    ctx.clearRect( 0,0, width, height );

    // draw the timeline

    var timelineHeight = 100;
    var timelineY = (height / 2) - (timelineHeight / 2);

    ctx.fillText( "Pos: " + String(sndIndex) +
                  "   Beats: " + String(beatmap.length) +
                  "   Clusters: " + String(clusters) +
                  "   Segments: " + String(segments) +
                  "   Ratio: " + (segments / clusters).toFixed(3),
                  0, timelineY + timelineHeight + 23);

    var currentX = (playvector[sndIndex].beat * beatWidth) + xOffset;
    var currentBeat = beatmap[playvector[sndIndex].beat];

    var segmentColorMap = ['#C2C2C2', '#FFFFFF'];

    ctx.strokeRect(0, timelineY, width - 5, timelineHeight);
    ctx.shadowColor = 'Black';
    ctx.shadowOffsetX = 3;
    ctx.shadowOffsetY = 3;
    ctx.shadowBlur = 5;

    ctx.strokeStyle = 'Black';
    ctx.fillStyle = 'White';
    ctx.lineWidth = 1;
    ctx.fillRect(0, timelineY, width - 5, timelineHeight);

    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
    ctx.shadowBlur = 0;

    for (var i =0; i<beatmap.length; i++) {

        var isSameSegment = beatmap[i].segment == currentBeat.segment;
        var isJumpCandidate = currentBeat.jump_candidate.includes(i);

        if (isSameSegment || isJumpCandidate) {
            ctx.globalAlpha = 1.0;
        }else{
            ctx.globalAlpha = .1;
        }

        ctx.fillStyle = colorMap[beatmap[i].cluster];
        ctx.strokeStyle = ctx.fillStyle;
        ctx.fillRect( (i * beatWidth) + xOffset, timelineY + 1,
                       ctx.globalAlpha < 1.0 ? beatWidth : beatWidth + 2, timelineHeight - 2);
    }

    // draw the jump candidate arcs

    var over = true;

    ctx.globalAlpha = 1.0;

    currentBeat.jump_candidate.forEach( (c) => {
        ctx.beginPath();

        // ctx.strokeStyle = 'Black';
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

    // draw the marker

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

    var leftInSeq =  playvector[sndIndex].seq_len - playvector[sndIndex].seq_pos;

    var str = '-' + String(leftInSeq).padStart(2, '0')

    if (leftInSeq == 0) {
        str = '!!!'
    }

    // ctx.fillText( str, currentX - 5, timelineY + timelineHeight + 25 );
    ctx.fillText( str, currentX - 20, timelineY + (timelineHeight / 2) );
}