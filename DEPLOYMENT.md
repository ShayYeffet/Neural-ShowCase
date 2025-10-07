# 🚀 Deployment Guide

This guide shows you how to deploy your Neural Showcase to various hosting platforms.

## Option 1: Vercel (Recommended for Frontend + Backend)

### Deploy Both Frontend and Backend to Vercel

1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Deploy Backend:**
   ```bash
   # In project root
   vercel --prod
   ```

3. **Deploy Frontend:**
   ```bash
   cd web/frontend
   vercel --prod
   ```

4. **Update API URL:**
   - Update `web/frontend/src/services/api.ts`
   - Change `http://localhost:8001` to your backend Vercel URL

## Option 2: Railway (Great for Python Backend)

### Deploy Backend to Railway

1. **Create `railway.json`:**
   ```json
   {
     "build": {
       "builder": "NIXPACKS"
     },
     "deploy": {
       "startCommand": "python backend.py",
       "healthcheckPath": "/health"
     }
   }
   ```

2. **Deploy:**
   - Go to [railway.app](https://railway.app)
   - Connect your GitHub repo
   - Deploy automatically

3. **Deploy Frontend to Vercel:**
   - Connect `web/frontend` folder to Vercel
   - Update API URL to Railway backend URL

## Option 3: Heroku

### Deploy Backend to Heroku

1. **Create `Procfile`:**
   ```
   web: python backend.py
   ```

2. **Create `runtime.txt`:**
   ```
   python-3.11.0
   ```

3. **Deploy:**
   ```bash
   heroku create your-neural-showcase-api
   git push heroku main
   ```

### Deploy Frontend to Netlify

1. **Build frontend:**
   ```bash
   cd web/frontend
   npm run build
   ```

2. **Deploy to Netlify:**
   - Drag `build` folder to [netlify.com](https://netlify.com)
   - Or connect GitHub repo

## Option 4: GitHub Pages (Frontend Only)

### For Frontend-Only Demo

1. **Create GitHub Action** (`.github/workflows/deploy.yml`):
   ```yaml
   name: Deploy to GitHub Pages
   on:
     push:
       branches: [ main ]
   jobs:
     deploy:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v2
         - name: Setup Node.js
           uses: actions/setup-node@v2
           with:
             node-version: '18'
         - name: Install and Build
           run: |
             cd web/frontend
             npm install
             npm run build
         - name: Deploy
           uses: peaceiris/actions-gh-pages@v3
           with:
             github_token: ${{ secrets.GITHUB_TOKEN }}
             publish_dir: ./web/frontend/build
   ```

2. **Enable GitHub Pages:**
   - Go to repo Settings → Pages
   - Select "Deploy from a branch" → gh-pages

## Recommended Setup

**For Portfolio/Demo:**
- **Frontend**: Vercel or Netlify (free tier)
- **Backend**: Railway or Render (free tier)

**For Production:**
- **Frontend**: Vercel Pro
- **Backend**: Railway Pro or AWS

## Environment Variables

When deploying, make sure to set:

**Backend:**
- `PORT` (usually set automatically)
- `PYTHONPATH` (if needed)

**Frontend:**
- `REACT_APP_API_URL` (your backend URL)

## Quick Deploy Commands

```bash
# 1. Deploy backend to Railway
railway login
railway link
railway up

# 2. Deploy frontend to Vercel
cd web/frontend
vercel --prod

# 3. Update API URL in frontend
# Edit web/frontend/src/services/api.ts
# Replace localhost with your backend URL
```

## Testing Deployment

After deployment:
1. Test image classification
2. Test sentiment analysis with all 3 models
3. Test time series prediction
4. Check API documentation at `/docs`

Your Neural Showcase will be live and accessible worldwide! 🌍