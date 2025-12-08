# NutriScan AI - Food Calorie Estimator

A web application that uses deep learning to identify food dishes and estimate their nutritional content.

## Features
- 🥗 Identifies 50 different food dishes
- 🔥 Estimates calories per serving
- 📊 Shows macronutrient breakdown (protein, carbs, fat)
- 📱 Responsive design for mobile and desktop
- ⚡ Runs entirely in the browser using TensorFlow.js

## Tech Stack
- **Frontend**: HTML, CSS, JavaScript
- **ML Framework**: TensorFlow.js
- **Model**: MobileNetV2 (transfer learning on Food101 dataset)
- **Dataset**: Food101 (50 selected classes)

## Deployment
Deployed on Netlify: [Your URL here]

## Model Details
- **Architecture**: MobileNetV2 + custom classification head
- **Input**: 224x224 RGB images
- **Output**: 50 food classes
- **Accuracy**: ~70% on test set
- **Size**: ~12MB (optimized for web)

## Local Development

1. Clone the repository
2. Navigate to `food-calorie-app` folder
3. Start a local server:
   ```bash
   python3 -m http.server 8000
   ```
4. Open `http://localhost:8000/index.html`

## Project Structure
```
food-calorie-app/
├── index.html          # Main app page
├── style.css           # Styling
├── app.js              # App logic
├── netlify.toml        # Netlify configuration
└── model/              # TensorFlow.js model files
    ├── model.json
    ├── group1-shard1of3.bin
    ├── group1-shard2of3.bin
    ├── group1-shard3of3.bin
    └── classes.json
```

## Credits
- Course: AAI 521 Computer Vision
- Team: Team 4
- Model trained on Food101 dataset
