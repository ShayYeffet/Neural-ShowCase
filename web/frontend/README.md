# Neural Showcase Frontend

React-based frontend for the Neural Showcase deep learning demonstration platform.

## Features

- **Responsive Design**: Built with Material-UI for consistent, mobile-friendly interface
- **Model Interactions**: Dedicated interfaces for CNN, Transformer, and LSTM models
- **Real-time Updates**: WebSocket integration for training progress monitoring
- **State Management**: Redux Toolkit for predictable state management
- **TypeScript**: Full type safety throughout the application

## Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── Common/         # Generic components (FileUpload, ModelSelector, etc.)
│   └── Layout/         # Layout components (Navigation, Sidebar)
├── pages/              # Page components for each route
├── services/           # API and WebSocket services
├── store/              # Redux store and slices
└── App.tsx            # Main application component
```

## Available Scripts

- `npm start` - Start development server
- `npm build` - Build for production
- `npm test` - Run tests
- `npm run eject` - Eject from Create React App

## Environment Variables

Create a `.env` file in the frontend directory:

```
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=http://localhost:8000
```

## Development Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm start
   ```

3. Open [http://localhost:3000](http://localhost:3000) to view in browser

## Pages

- **Home**: Overview and navigation to different model demos
- **Image Classification**: CNN model interface with Grad-CAM visualization
- **Sentiment Analysis**: Transformer model interface with attention visualization
- **Time Series**: LSTM model interface with forecasting
- **Training Dashboard**: Real-time training monitoring and experiment management

## State Management

The application uses Redux Toolkit with the following slices:

- `modelsSlice`: Available models, selection, and loading states
- `trainingSlice`: Training experiments and progress tracking
- `uiSlice`: UI state (sidebar, theme, notifications)

## API Integration

Services are organized by functionality:

- `modelService`: Model loading, prediction, and visualization
- `trainingService`: Training management and experiment tracking
- `websocketService`: Real-time updates and notifications

## Component Architecture

- **Layout**: Responsive navigation with collapsible sidebar
- **Common Components**: Reusable UI elements for consistent interface
- **Page Components**: Feature-specific interfaces with integrated functionality
- **Service Layer**: Abstracted API communication with error handling