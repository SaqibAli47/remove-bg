# Background Removal API

A FastAPI-based service that removes backgrounds from images using the rembg library.

## Features
- Remove background from images
- Base64 image input/output
- Automatic deployment to Vercel
- CI/CD with GitHub Actions
- Model caching to improve performance

## API Endpoints

### Remove Background
- **URL**: `/api/remove-bg`
- **Method**: POST
- **Body**: JSON with base64 encoded image
- **Returns**: JSON with base64 encoded image with background removed

### Health Check
- **URL**: `/api/healthcheck`
- **Method**: GET
- **Returns**: API status information

## Setup and Deployment

1. Clone the repository
```bash
git clone <your-repo-url>
cd <your-repo-name>
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up Vercel:
- Install Vercel CLI: `npm install -g vercel`
- Login to Vercel: `vercel login`
- Link project: `vercel link`

4. Set up GitHub Secrets:
Add the following secrets to your GitHub repository:
- VERCEL_TOKEN
- VERCEL_ORG_ID
- VERCEL_PROJECT_ID

5. Push to GitHub:
```bash
git add .
git commit -m "Initial commit"
git push
```

## Local Development
```bash
pip install -r requirements.txt
uvicorn api.index:app --reload
```

## Deployment
This project is configured for deployment on Render.com.

## Testing
```bash
pytest
```

## Notes
- The U2Net model will be downloaded automatically on first use
- The model is cached to improve subsequent performance
