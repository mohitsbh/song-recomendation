# Deployment Guide

## GitHub Upload (Manual Method)

Since Git CLI is not available, use GitHub Desktop or web interface:

### Option 1: GitHub Desktop
1. Download GitHub Desktop from https://desktop.github.com/
2. Install and sign in with your GitHub account
3. Click "Add Local Repository" and select `C:\Users\msbho\Documents\emotion`
4. Commit all files with message "Initial commit"
5. Publish repository to GitHub as `song-recomendation`

### Option 2: GitHub Web Interface
1. Go to https://github.com/mohitsbh/song-recomendation
2. Click "uploading an existing file"
3. Drag and drop all files from `C:\Users\msbho\Documents\emotion`
4. Commit changes

### Option 3: Install Git
1. Download Git from https://git-scm.com/download/win
2. Install with default settings
3. Restart VS Code/Terminal
4. Run these commands:

```bash
cd "C:\Users\msbho\Documents\emotion"
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/mohitsbh/song-recomendation.git
git push -u origin main
```

## Render Deployment

### Prerequisites
- GitHub repository must be public or connected to Render
- Model file (model_v6_23.hdf5) must be under 100MB for free tier

### Steps

1. **Go to Render Dashboard**
   - Visit https://render.com/
   - Sign in with GitHub

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository: `mohitsbh/song-recomendation`
   - Click "Connect"

3. **Configure Service**
   ```
   Name: emotitunes
   Environment: Python 3
   Region: Choose closest to you
   Branch: main
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn app:app
   ```

4. **Advanced Settings**
   - Instance Type: Free
   - Add Environment Variable (optional):
     - Key: `PYTHON_VERSION`
     - Value: `3.10.0`

5. **Deploy**
   - Click "Create Web Service"
   - Wait 5-10 minutes for deployment
   - Your app will be at: `https://emotitunes.onrender.com`

### Important Notes

⚠️ **Model File Size**: If your model is >100MB, consider:
- Using Git LFS (Large File Storage)
- Hosting model on external storage (S3, Google Drive)
- Optimizing/compressing the model

⚠️ **Free Tier Limitations**:
- App sleeps after 15 min of inactivity
- First request after sleep takes 30-60 seconds
- 750 hours/month free

### Troubleshooting

**Build fails?**
- Check logs in Render dashboard
- Ensure requirements.txt has all dependencies
- TensorFlow can be slow to install (5-10 min)

**App crashes?**
- Check if model file is present in repo
- Verify model path in code: `model/model_v6_23.hdf5`
- Check Render logs for errors

**Static files not loading?**
- Ensure `static/` folder is in repository
- Flask is configured with: `app = Flask(__name__, static_folder='static')`

### Testing Locally Before Deploy

```bash
# Install gunicorn locally
pip install gunicorn

# Test the same command Render will use
gunicorn app:app

# Visit http://localhost:8000
```

### After Deployment

1. Test the live URL
2. Try camera and upload features
3. Check if local songs work (may need to re-upload to static folder)
4. Monitor performance in Render dashboard
