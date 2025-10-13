# Neural Showcase

A modern web application demonstrating different neural network architectures with real AI models. Built for educational purposes and portfolio demonstration.

## Features

- ** Image Classification**: Upload images and get real-time predictions using ResNet-18
- ** Sentiment Analysis**: Choose from 3 different AI models (DistilBERT, RoBERTa, VADER)
- ** Time Series Prediction**: Forecast future values from historical data
- ** Modern UI**: Beautiful React interface with Material-UI components
- ** Real AI Models**: Uses pre-trained PyTorch models

## Live Demo

*Demo deployment coming soon - follow the setup instructions below to run locally*

## Tech Stack

**Frontend:**
- React 18 with TypeScript
- Material-UI (MUI) for components
- Redux Toolkit for state management
- Recharts for data visualization

**Backend:**
- FastAPI (Python)
- PyTorch for CNN models
- Advanced sentiment analysis algorithms
- Real pre-trained models (no training required)

## Quick Start

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/neural-showcase.git
cd neural-showcase
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install frontend dependencies**
```bash
cd web/frontend
npm install
```

### Running the Application

1. **Start the backend** (in project root):
```bash
python backend.py
```
The API will be available at `http://localhost:8001`

2. **Start the frontend** (in another terminal):
```bash
cd web/frontend
npm start
```
The web app will open at `http://localhost:3000`

## 📖 How to Use

### Image Classification
1. Select the ResNet-18 model
2. Upload any image (JPG, PNG, etc.)
3. Get instant predictions with confidence scores
4. See top 5 most likely classes from ImageNet

### Sentiment Analysis  
1. Choose from 3 AI models:
   - **DistilBERT**: Fast and accurate for general text
   - **RoBERTa**: Most sophisticated, best for complex text
   - **VADER**: Optimized for social media text
2. Type or paste any text
3. Get sentiment classification (positive/negative/neutral)
4. View confidence scores and probabilities

### Time Series Prediction
1. Enter comma-separated numerical data
2. Choose number of future steps to predict
3. Get forecasted values with uncertainty estimates

## Educational Value

This project demonstrates:
- **CNN Architecture**: How convolutional networks process images
- **Multiple AI Models**: Compare different sentiment analysis approaches
- **Time Series Analysis**: Forecasting with sequential data
- **Full-Stack AI**: Complete pipeline from model to web interface
- **Real-World AI**: Using production-ready pre-trained models

## Project Structure

```
neural-showcase/
├── backend.py              # FastAPI backend with AI models
├── web/frontend/           # React frontend application
│   ├── src/
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/         # Main application pages
│   │   ├── services/      # API communication
│   │   └── store/         # Redux state management
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## API Endpoints

- `GET /models` - List available AI models
- `POST /predict/image` - Image classification with ResNet-18
- `POST /predict/sentiment` - Sentiment analysis with multiple models
- `POST /predict/timeseries` - Time series forecasting
- `GET /docs` - Interactive API documentation

## Models Used

**Image Classification:**
- **ResNet-18**: Pre-trained on ImageNet (1000 classes)

**Sentiment Analysis:**
- **DistilBERT**: Fast transformer model for general sentiment
- **RoBERTa**: Advanced transformer with context understanding
- **VADER**: Social media optimized sentiment analyzer

**Time Series:**
- **Linear Regression**: Simple but effective forecasting

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Acknowledgments

- PyTorch team for the pre-trained models
- Material-UI for the beautiful components
- FastAPI for the excellent web framework

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions to Vercel, Railway, Heroku, and other platforms.

## 📞 Contact

Project Link: [https://github.com/yourusername/neural-showcase](https://github.com/yourusername/neural-showcase)

---

⭐ If you found this project helpful, please give it a star!
