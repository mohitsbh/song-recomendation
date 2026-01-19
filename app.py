from flask import Flask, request, jsonify, render_template_string
import os
from recommend import load_model_info, preprocess_image, predict_emotion, get_recommendations, get_local_songs

# Use absolute path based on this file's location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model_v6_23.hdf5')

app = Flask(__name__, static_folder='static')

HOME_PAGE = '''
<!DOCTYPE html>
<html>
<head>
    <title>EmotiTunes - AI Music Recommender</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Poppins', sans-serif; 
            min-height: 100vh;
            background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
            color: #fff;
            padding: 20px;
        }
        .container { max-width: 800px; margin: 0 auto; }
        
        header { text-align: center; padding: 40px 0; }
        header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf, #e040fb);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }
        header p { color: #a0a0a0; font-size: 1.1rem; font-weight: 300; }
        
        .main-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            border-radius: 24px;
            padding: 40px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
        }
        
        .tabs {
            display: flex; gap: 10px; margin-bottom: 30px;
            background: rgba(0, 0, 0, 0.2); padding: 8px; border-radius: 16px;
        }
        .tab {
            flex: 1; padding: 14px 20px; cursor: pointer;
            background: transparent; border-radius: 12px; text-align: center;
            font-weight: 500; transition: all 0.3s ease; color: #888;
        }
        .tab:hover { color: #fff; }
        .tab.active {
            background: linear-gradient(135deg, #7b2cbf, #e040fb);
            color: #fff; box-shadow: 0 4px 15px rgba(123, 44, 191, 0.4);
        }
        .tab i { margin-right: 8px; }
        
        .tab-content { display: none; }
        .tab-content.active { display: block; animation: fadeIn 0.4s ease; }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .camera-section { text-align: center; }
        #video {
            width: 100%; max-width: 480px; height: auto; aspect-ratio: 4/3;
            border-radius: 16px; background: linear-gradient(135deg, #1a1a2e, #16213e);
            display: none; margin: 0 auto 20px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        }
        #canvas { display: none; }
        
        .camera-placeholder {
            width: 100%; max-width: 480px; aspect-ratio: 4/3;
            border-radius: 16px; background: linear-gradient(135deg, #1a1a2e, #16213e);
            display: flex; align-items: center; justify-content: center; flex-direction: column;
            margin: 0 auto 20px; border: 2px dashed rgba(255, 255, 255, 0.1);
        }
        .camera-placeholder i { font-size: 4rem; color: #444; margin-bottom: 15px; }
        .camera-placeholder p { color: #666; font-size: 0.95rem; }
        
        .btn-group { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; }
        
        button {
            padding: 14px 28px; border: none; border-radius: 12px;
            font-family: 'Poppins', sans-serif; font-size: 0.95rem; font-weight: 500;
            cursor: pointer; transition: all 0.3s ease;
            display: inline-flex; align-items: center; gap: 8px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #00d4ff, #0099cc);
            color: #000; box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
        }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4); }
        
        .btn-secondary {
            background: rgba(255, 255, 255, 0.1); color: #fff;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .btn-secondary:hover { background: rgba(255, 255, 255, 0.15); transform: translateY(-2px); }
        
        .btn-danger {
            background: rgba(255, 71, 87, 0.2); color: #ff4757;
            border: 1px solid rgba(255, 71, 87, 0.3);
        }
        .btn-danger:hover { background: rgba(255, 71, 87, 0.3); }
        
        .upload-area {
            border: 2px dashed rgba(255, 255, 255, 0.2); border-radius: 16px;
            padding: 40px; text-align: center; transition: all 0.3s ease;
            cursor: pointer; margin-bottom: 20px;
        }
        .upload-area:hover { border-color: #7b2cbf; background: rgba(123, 44, 191, 0.1); }
        .upload-area i { font-size: 3rem; color: #7b2cbf; margin-bottom: 15px; display: block; }
        .upload-area p { color: #888; margin-bottom: 5px; }
        .upload-area span { color: #00d4ff; font-weight: 500; }
        
        #imageInput { display: none; }
        
        #preview {
            max-width: 300px; max-height: 300px; border-radius: 16px;
            margin: 20px auto; display: none; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        }
        
        #result {
            margin-top: 30px; padding: 30px; background: rgba(0, 0, 0, 0.2);
            border-radius: 20px; display: none; animation: fadeIn 0.5s ease;
        }
        
        .result-header {
            display: flex; align-items: center; gap: 20px; margin-bottom: 25px;
            padding-bottom: 20px; border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .emotion-icon {
            width: 70px; height: 70px; border-radius: 50%;
            background: linear-gradient(135deg, #7b2cbf, #e040fb);
            display: flex; align-items: center; justify-content: center; font-size: 2rem;
        }
        
        .emotion-text h2 { font-size: 1.8rem; font-weight: 600; margin-bottom: 5px; }
        .emotion-text p { color: #888; font-size: 0.9rem; }
        
        .playlists-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 15px; color: #00d4ff; }
        .playlists-title i { margin-right: 8px; }
        
        .section-divider { margin: 25px 0; border-top: 1px solid rgba(255, 255, 255, 0.1); }
        
        .playlist-item {
            display: flex; align-items: center; gap: 15px; padding: 15px;
            background: rgba(255, 255, 255, 0.05); border-radius: 12px;
            margin-bottom: 10px; transition: all 0.3s ease;
            text-decoration: none; color: #fff; cursor: pointer;
        }
        .playlist-item:hover { background: rgba(255, 255, 255, 0.1); transform: translateX(5px); }
        .playlist-item .play-icon {
            width: 45px; height: 45px; border-radius: 50%;
            background: linear-gradient(135deg, #1db954, #1ed760);
            display: flex; align-items: center; justify-content: center;
            color: #fff; font-size: 1rem; flex-shrink: 0;
        }
        .playlist-item .play-icon.local {
            background: linear-gradient(135deg, #e040fb, #7b2cbf);
        }
        .playlist-item .play-icon.playing {
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.1); }
        }
        .playlist-item .playlist-info { flex: 1; min-width: 0; }
        .playlist-item .playlist-info h4 { font-weight: 500; margin-bottom: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .playlist-item .playlist-info p { font-size: 0.8rem; color: #888; }
        .playlist-item .arrow { margin-left: auto; color: #666; flex-shrink: 0; }
        
        .audio-player {
            margin-top: 25px; padding: 20px;
            background: linear-gradient(135deg, rgba(123, 44, 191, 0.2), rgba(224, 64, 251, 0.1));
            border-radius: 16px; border: 1px solid rgba(123, 44, 191, 0.3);
        }
        .audio-player-header {
            display: flex; align-items: center; gap: 15px; margin-bottom: 15px;
        }
        .audio-player-header .now-playing-icon {
            width: 50px; height: 50px; border-radius: 12px;
            background: linear-gradient(135deg, #e040fb, #7b2cbf);
            display: flex; align-items: center; justify-content: center;
            font-size: 1.2rem;
        }
        .audio-player-header .track-info h4 {
            font-weight: 600; margin-bottom: 3px;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            max-width: 300px;
        }
        .audio-player-header .track-info p { font-size: 0.85rem; color: #a0a0a0; }
        
        #audioElement {
            width: 100%; height: 45px; border-radius: 8px; outline: none;
        }
        
        .no-local-songs {
            text-align: center; padding: 20px; color: #666;
            background: rgba(255, 255, 255, 0.02); border-radius: 12px;
            border: 1px dashed rgba(255, 255, 255, 0.1);
        }
        .no-local-songs i { font-size: 2rem; margin-bottom: 10px; display: block; }
        .no-local-songs p { font-size: 0.9rem; }
        .no-local-songs code { 
            background: rgba(0, 212, 255, 0.1); padding: 2px 8px; 
            border-radius: 4px; font-size: 0.85rem; color: #00d4ff;
        }
        
        .loading { text-align: center; padding: 40px; }
        .spinner {
            width: 50px; height: 50px;
            border: 3px solid rgba(255, 255, 255, 0.1); border-top-color: #00d4ff;
            border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 15px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        
        footer { text-align: center; padding: 30px; color: #555; font-size: 0.85rem; }
        footer a { color: #7b2cbf; text-decoration: none; }
        
        @media (max-width: 600px) {
            header h1 { font-size: 1.8rem; }
            .main-card { padding: 25px; }
            .btn-group { flex-direction: column; }
            button { width: 100%; justify-content: center; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1><i class="fas fa-music"></i> Song Recomendation using Facial Expression</h1>
            <p>AI-powered music recommendations based on your emotions</p>
        </header>
        
        <div class="main-card">
            <div class="tabs">
                <div class="tab active" onclick="switchTab('camera')">
                    <i class="fas fa-camera"></i> Live Camera
                </div>
                <div class="tab" onclick="switchTab('upload')">
                    <i class="fas fa-cloud-upload-alt"></i> Upload Photo
                </div>
            </div>
            
            <div id="camera-tab" class="tab-content active">
                <div class="camera-section">
                    <div class="camera-placeholder" id="cameraPlaceholder">
                        <i class="fas fa-video"></i>
                        <p>Click "Start Camera" to begin</p>
                    </div>
                    <video id="video" autoplay playsinline></video>
                    <canvas id="canvas"></canvas>
                    <div class="btn-group">
                        <button class="btn-secondary" onclick="startCamera()">
                            <i class="fas fa-video"></i> Start Camera
                        </button>
                        <button class="btn-primary" onclick="captureAndPredict()">
                            <i class="fas fa-camera"></i> Capture & Analyze
                        </button>
                        <button class="btn-danger" onclick="stopCamera()">
                            <i class="fas fa-stop"></i> Stop
                        </button>
                    </div>
                </div>
            </div>
            
            <div id="upload-tab" class="tab-content">
                <div class="upload-area" onclick="document.getElementById('imageInput').click()">
                    <i class="fas fa-cloud-upload-alt"></i>
                    <p>Drag & drop your photo here or</p>
                    <span>Browse Files</span>
                </div>
                <input type="file" id="imageInput" accept="image/*">
                <img id="preview">
                <div class="btn-group">
                    <button class="btn-primary" onclick="predictFromFile()">
                        <i class="fas fa-magic"></i> Analyze Emotion
                    </button>
                </div>
            </div>
            
            <div id="result"></div>
        </div>
        
        <footer>
            Powered by AI • Built with <a href="#">TensorFlow</a> & <a href="#">Flask</a>
        </footer>
    </div>

    <script>
        let stream = null;
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const placeholder = document.getElementById('cameraPlaceholder');
        
        const emotionEmojis = {
            'Happy': '😊', 'Sad': '😢', 'Angry': '😠', 'Fear': '😨',
            'Surprise': '😲', 'Disgust': '🤢', 'Neutral': '😐'
        };
        
        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            if (tab === 'camera') {
                document.querySelector('.tab:nth-child(1)').classList.add('active');
                document.getElementById('camera-tab').classList.add('active');
            } else {
                document.querySelector('.tab:nth-child(2)').classList.add('active');
                document.getElementById('upload-tab').classList.add('active');
            }
        }
        
        async function startCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { facingMode: 'user', width: 640, height: 480 }, audio: false 
                });
                video.srcObject = stream;
                video.style.display = 'block';
                placeholder.style.display = 'none';
            } catch (err) {
                alert('Could not access camera: ' + err.message);
            }
        }
        
        function stopCamera() {
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
                video.style.display = 'none';
                placeholder.style.display = 'flex';
            }
        }
        
        async function captureAndPredict() {
            if (!stream) { alert('Please start the camera first'); return; }
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            
            canvas.toBlob(async (blob) => {
                const formData = new FormData();
                formData.append('image', blob, 'capture.jpg');
                await sendPrediction(formData);
            }, 'image/jpeg', 0.9);
        }

        document.getElementById('imageInput').onchange = function(e) {
            const preview = document.getElementById('preview');
            if (e.target.files.length) {
                preview.src = URL.createObjectURL(e.target.files[0]);
                preview.style.display = 'block';
            }
        };

        async function predictFromFile() {
            const input = document.getElementById('imageInput');
            if (!input.files.length) { alert('Please select an image first'); return; }
            const formData = new FormData();
            formData.append('image', input.files[0]);
            await sendPrediction(formData);
        }
        
        let currentAudio = null;
        let currentPlayingItem = null;
        
        function playLocalSong(url, title, element) {
            // Stop current audio if playing
            if (currentAudio) {
                currentAudio.pause();
                if (currentPlayingItem) {
                    currentPlayingItem.querySelector('.play-icon').classList.remove('playing');
                    currentPlayingItem.querySelector('.play-icon i').className = 'fas fa-play';
                }
            }
            
            // Update audio player
            const audioElement = document.getElementById('audioElement');
            const trackTitle = document.getElementById('trackTitle');
            const audioPlayer = document.getElementById('audioPlayer');
            
            audioElement.src = url;
            trackTitle.textContent = title;
            audioPlayer.style.display = 'block';
            audioElement.play();
            
            currentAudio = audioElement;
            currentPlayingItem = element;
            
            // Update play icon
            element.querySelector('.play-icon').classList.add('playing');
            element.querySelector('.play-icon i').className = 'fas fa-pause';
            
            // Handle audio end
            audioElement.onended = function() {
                element.querySelector('.play-icon').classList.remove('playing');
                element.querySelector('.play-icon i').className = 'fas fa-play';
                currentPlayingItem = null;
            };
            
            audioElement.onpause = function() {
                if (currentPlayingItem) {
                    currentPlayingItem.querySelector('.play-icon').classList.remove('playing');
                    currentPlayingItem.querySelector('.play-icon i').className = 'fas fa-play';
                }
            };
            
            audioElement.onplay = function() {
                if (currentPlayingItem) {
                    currentPlayingItem.querySelector('.play-icon').classList.add('playing');
                    currentPlayingItem.querySelector('.play-icon i').className = 'fas fa-pause';
                }
            };
        }
        
        async function sendPrediction(formData) {
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = '<div class="loading"><div class="spinner"></div><p>Analyzing your emotion...</p></div>';
            resultDiv.style.display = 'block';

            try {
                const res = await fetch('/predict', { method: 'POST', body: formData });
                const data = await res.json();
                if (data.error) {
                    resultDiv.innerHTML = '<div class="loading"><i class="fas fa-exclamation-circle" style="font-size:2rem;color:#ff4757;"></i><p style="color:#ff4757;margin-top:10px;">' + data.error + '</p></div>';
                    return;
                }
                
                const emoji = emotionEmojis[data.emotion] || '🎵';
                let html = `
                    <div class="result-header">
                        <div class="emotion-icon">${emoji}</div>
                        <div class="emotion-text">
                            <h2>${data.emotion}</h2>
                            <p>Detected emotion from your photo</p>
                        </div>
                    </div>
                `;
                
                // Local Songs Section
                html += `<p class="playlists-title"><i class="fas fa-folder-open"></i> Your Local Songs</p>`;
                
                if (data.local_songs && data.local_songs.length > 0) {
                    data.local_songs.forEach((song, idx) => {
                        html += `
                            <div class="playlist-item" onclick="playLocalSong('${song.url}', '${song.title.replace(/'/g, "\\'")}', this)" id="local-song-${idx}">
                                <div class="play-icon local"><i class="fas fa-play"></i></div>
                                <div class="playlist-info">
                                    <h4>${song.title}</h4>
                                    <p>Local Music</p>
                                </div>
                                <i class="fas fa-music arrow"></i>
                            </div>
                        `;
                    });
                    
                    // Audio Player
                    html += `
                        <div class="audio-player" id="audioPlayer" style="display:none;">
                            <div class="audio-player-header">
                                <div class="now-playing-icon"><i class="fas fa-music"></i></div>
                                <div class="track-info">
                                    <h4 id="trackTitle">Select a song</h4>
                                    <p>Now Playing</p>
                                </div>
                            </div>
                            <audio id="audioElement" controls></audio>
                        </div>
                    `;
                } else {
                    html += `
                        <div class="no-local-songs">
                            <i class="fas fa-folder-plus"></i>
                            <p>No local songs found. Add MP3 files to:</p>
                            <p><code>static/songs/</code></p>
                            <p style="margin-top:8px;font-size:0.8rem;">Tip: Create subfolders like <code>happy/</code>, <code>sad/</code> for emotion-specific songs</p>
                        </div>
                    `;
                }
                
                // Spotify Section
                html += `<div class="section-divider"></div>`;
                html += `<p class="playlists-title"><i class="fas fa-headphones"></i> Spotify Playlists</p>`;
                
                data.recommendations.forEach(r => {
                    html += `
                        <a href="${r.url}" target="_blank" class="playlist-item">
                            <div class="play-icon"><i class="fas fa-play"></i></div>
                            <div class="playlist-info">
                                <h4>${r.title}</h4>
                                <p>Spotify Playlist</p>
                            </div>
                            <i class="fas fa-external-link-alt arrow"></i>
                        </a>
                    `;
                });
                
                resultDiv.innerHTML = html;
            } catch (err) {
                resultDiv.innerHTML = '<div class="loading"><i class="fas fa-exclamation-circle" style="font-size:2rem;color:#ff4757;"></i><p style="color:#ff4757;margin-top:10px;">' + err.message + '</p></div>';
            }
        }
    </script>
</body>
</html>
'''

try:
    model, input_shape, channels = load_model_info(MODEL_PATH)
except Exception as e:
    model = None
    input_shape = None
    channels = None
    print('Warning: could not load model:', e)


@app.route('/')
def home():
    return render_template_string(HOME_PAGE)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on server'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided (field name: image)'}), 400

    file = request.files['image']
    try:
        img_arr = preprocess_image(file, input_shape, channels)
    except Exception as e:
        return jsonify({'error': 'Failed to preprocess image', 'detail': str(e)}), 400

    try:
        emotion, probs = predict_emotion(model, img_arr)
    except Exception as e:
        return jsonify({'error': 'Model prediction failed', 'detail': str(e)}), 500

    recs = get_recommendations(emotion)
    local_songs = get_local_songs(emotion)

    return jsonify({
        'emotion': emotion, 
        'probabilities': probs, 
        'recommendations': recs,
        'local_songs': local_songs
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
