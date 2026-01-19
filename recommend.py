import numpy as np
from PIL import Image
import io
import os
import random
from tensorflow.keras.models import load_model

# Default label order (common FER2013 ordering). Adjust if your model uses a different ordering.
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Local songs folder path
LOCAL_SONGS_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'songs')

# Mapping of emotions to song tags/folders
# You can organize songs in subfolders by emotion, or tag them in filenames
EMOTION_SONG_TAGS = {
    'Happy': ['happy', 'upbeat', 'joy', 'cheerful', 'party'],
    'Sad': ['sad', 'melancholy', 'slow', 'emotional', 'ballad'],
    'Angry': ['angry', 'rock', 'metal', 'intense', 'aggressive'],
    'Fear': ['dark', 'ambient', 'suspense', 'eerie'],
    'Surprise': ['surprise', 'energetic', 'dynamic', 'exciting'],
    'Disgust': ['experimental', 'alternative', 'grunge'],
    'Neutral': ['chill', 'relaxing', 'calm', 'acoustic', 'lofi']
}


def load_model_info(model_path):
    model = load_model(model_path)
    # model.input_shape often is (None, H, W, C) or (None, H, W)
    shape = model.input_shape
    if len(shape) == 4:
        _, h, w, c = shape
        input_shape = (h, w)
        channels = c
    elif len(shape) == 3:
        _, h, w = shape
        input_shape = (h, w)
        channels = 1
    else:
        raise ValueError('Unexpected model input shape: {}'.format(shape))
    return model, input_shape, channels


def preprocess_image(file_obj, input_shape, channels):
    # file_obj: FileStorage or file-like
    data = file_obj.read()
    img = Image.open(io.BytesIO(data))
    if channels == 1:
        img = img.convert('L')
    else:
        img = img.convert('RGB')

    img = img.resize((input_shape[1], input_shape[0]))
    arr = np.array(img).astype('float32') / 255.0
    if channels == 1:
        if arr.ndim == 2:
            arr = np.expand_dims(arr, -1)
    return arr


def predict_emotion(model, img_arr):
    x = np.expand_dims(img_arr, 0)
    preds = model.predict(x)
    if preds.ndim == 2:
        probs = preds[0].tolist()
        idx = int(np.argmax(preds[0]))
    else:
        probs = preds.flatten().tolist()
        idx = int(np.argmax(preds))
    label = EMOTION_LABELS[idx] if idx < len(EMOTION_LABELS) else str(idx)
    # return label and a mapping of label->prob
    prob_map = {EMOTION_LABELS[i] if i < len(EMOTION_LABELS) else str(i): float(probs[i]) for i in range(len(probs))}
    return label, prob_map


def get_recommendations(emotion):
    # Simple static mapping. Replace links with your own playlists or integrate with Spotify/Youtube APIs.
    mapping = {
        'Happy': [
            {'title': 'Feel Good Pop', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC', 'type': 'spotify'},
            {'title': 'Upbeat Indie', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DWY4xHQp97fN6', 'type': 'spotify'}
        ],
        'Sad': [
            {'title': 'Sad Songs', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX7qK8ma5wgG1', 'type': 'spotify'},
            {'title': 'Melancholy Piano', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX4sWSpwq3LiO', 'type': 'spotify'}
        ],
        'Angry': [
            {'title': 'Hard Rock', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DWXRqgorJj26U', 'type': 'spotify'},
        ],
        'Fear': [
            {'title': 'Dark Ambient', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DWTvNyxOwkztu', 'type': 'spotify'}
        ],
        'Surprise': [
            {'title': 'Surprise/Upbeat Mix', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DWSf2RDTDayIx', 'type': 'spotify'}
        ],
        'Disgust': [
            {'title': 'Experimental', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX4fpCWaHOned', 'type': 'spotify'}
        ],
        'Neutral': [
            {'title': 'Chill Vibes', 'url': 'https://open.spotify.com/playlist/37i9dQZF1DX4WYpdgoIcn6', 'type': 'spotify'}
        ]
    }
    return mapping.get(emotion, mapping['Neutral'])


def get_local_songs(emotion):
    """Get local songs that match the emotion."""
    songs = []
    audio_extensions = {'.mp3', '.wav', '.ogg', '.m4a', '.flac', '.aac'}
    
    if not os.path.exists(LOCAL_SONGS_FOLDER):
        return songs
    
    tags = EMOTION_SONG_TAGS.get(emotion, EMOTION_SONG_TAGS['Neutral'])
    
    # Check for emotion-specific subfolder
    emotion_folder = os.path.join(LOCAL_SONGS_FOLDER, emotion.lower())
    
    # Collect songs from emotion subfolder if exists
    if os.path.exists(emotion_folder):
        for file in os.listdir(emotion_folder):
            ext = os.path.splitext(file)[1].lower()
            if ext in audio_extensions:
                songs.append({
                    'title': os.path.splitext(file)[0],
                    'url': f'/static/songs/{emotion.lower()}/{file}',
                    'type': 'local'
                })
    
    # Also check main songs folder for tagged files
    for file in os.listdir(LOCAL_SONGS_FOLDER):
        filepath = os.path.join(LOCAL_SONGS_FOLDER, file)
        if os.path.isfile(filepath):
            ext = os.path.splitext(file)[1].lower()
            if ext in audio_extensions:
                filename_lower = file.lower()
                # Check if filename contains any emotion tag
                if any(tag in filename_lower for tag in tags):
                    songs.append({
                        'title': os.path.splitext(file)[0],
                        'url': f'/static/songs/{file}',
                        'type': 'local'
                    })
    
    # If no tagged songs found, return random songs from the folder
    if not songs:
        for file in os.listdir(LOCAL_SONGS_FOLDER):
            filepath = os.path.join(LOCAL_SONGS_FOLDER, file)
            if os.path.isfile(filepath):
                ext = os.path.splitext(file)[1].lower()
                if ext in audio_extensions:
                    songs.append({
                        'title': os.path.splitext(file)[0],
                        'url': f'/static/songs/{file}',
                        'type': 'local'
                    })
    
    # Shuffle and limit
    random.shuffle(songs)
    return songs[:5]
