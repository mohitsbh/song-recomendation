# 🎵 EmotiTunes - Emotion-based Music Recommender

AI-powered music recommendation system that detects emotions from facial expressions and suggests matching music.

## Features

- 📷 **Live Camera** - Capture your face in real-time
- 📁 **Upload Photo** - Upload an image for emotion detection
- 🎭 **7 Emotions** - Detects Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
- 🎧 **Spotify Playlists** - Curated playlists for each emotion
- 🎵 **Local Music Support** - Play your own songs organized by emotion

## Tech Stack

- **Backend**: Flask, TensorFlow/Keras
- **Frontend**: HTML5, CSS3, JavaScript
- **ML Model**: CNN for facial emotion recognition

## Setup

1. Clone the repository:
```bash
git clone https://github.com/mohitsbh/song-recomendation.git
cd song-recomendation
```

2. Create virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
python app.py
```

5. Open http://localhost:5000 in your browser

## Local Songs

Add your MP3 files to `static/songs/` folder:
- Create subfolders by emotion: `happy/`, `sad/`, `angry/`, etc.
- Or use tags in filenames: `my_happy_song.mp3`, `chill_vibes.mp3`

## Deployment

This app is configured for deployment on Render.

## License

MIT License
