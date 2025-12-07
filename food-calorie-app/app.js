// App State
let model = null;
let classes = [];
let caloriesMap = {};
let nutritionMap = {}; // New state for nutrition
let isCustomModel = false;

// DOM Elements
const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');
const uploadContent = document.getElementById('upload-content');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const removeBtn = document.getElementById('remove-image');
const scanOverlay = document.getElementById('scan-overlay');
const resultCard = document.getElementById('result-card');
const scanAgainBtn = document.getElementById('scan-again');

// Initialization
async function init() {
    console.log('Initializing NutriScan AI...');
    try {
        // 1. Try to load custom headers/classes first
        await loadMetadata();

        // 2. Try to load custom model
        console.log('Loading custom model...');
        model = await tf.loadLayersModel('model/model.json?v=' + new Date().getTime());
        isCustomModel = true;
        console.log('Custom Food101 model loaded successfully.');
    } catch (e) {
        console.warn('Custom model not found or failed to load. Falling back to MobileNet for demo purposes.', e);
        // Fallback to MobileNet
        model = await mobilenet.load();
        isCustomModel = false;
        console.log('MobileNet loaded as fallback.');
    }
}

async function loadMetadata() {
    try {
        const response = await fetch('model/classes.json');
        const data = await response.json();
        classes = data.classes;
        caloriesMap = data.calories;
        // Load nutrition data if available
        if (data.nutrition) {
            nutritionMap = data.nutrition;
        }
        console.log('Metadata loaded:', classes.length, 'classes');
    } catch (e) {
        console.warn('Could not load metadata, using defaults if possible.');
    }
}

// Event Listeners
window.addEventListener('DOMContentLoaded', init);

// Drag & Drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--accent-primary)';
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--glass-border)';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--glass-border)';
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleImage(file);
    }
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        handleImage(file);
    }
});

removeBtn.addEventListener('click', resetUI);
scanAgainBtn.addEventListener('click', resetUI);

// Image Handling
function handleImage(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        showPreview();
        // Start scanning after a brief delay for UI effect
        setTimeout(predict, 1000);
    };
    reader.readAsDataURL(file);
}

function showPreview() {
    uploadContent.classList.add('hidden');
    previewContainer.classList.remove('hidden');
    scanOverlay.classList.remove('hidden');
    resultCard.classList.add('hidden');
}

function resetUI() {
    fileInput.value = '';
    imagePreview.src = '';
    uploadContent.classList.remove('hidden');
    previewContainer.classList.add('hidden');
    resultCard.classList.add('hidden');
}

// Prediction
async function predict() {
    if (!model) {
        console.error('Model not loaded');
        return;
    }

    try {
        let topClass = '';
        let confidence = 0;

        if (isCustomModel) {
            // Processing for Custom Food101 Model
            // MobileNetV2 usually expects [-1, 1] normalization: (img / 127.5) - 1
            const tensor = tf.tidy(() => {
                let img = tf.browser.fromPixels(imagePreview)
                    .resizeNearestNeighbor([224, 224])
                    .toFloat();

                // NEW: Validate model expectations. Assuming MobileNetV2 style.
                // Standard formula: (x / 127.5) - 1.0
                const normalized = img.div(tf.scalar(127.5)).sub(tf.scalar(1.0));

                return normalized.expandDims();
            });

            const predictions = await model.predict(tensor).data();

            // Get top prediction
            const topK = Array.from(predictions)
                .map((p, i) => ({ probability: p, className: classes[i] || `Class ${i}` }))
                .sort((a, b) => b.probability - a.probability);

            topClass = topK[0].className;
            confidence = topK[0].probability;

            // Clean up tensor
            tf.dispose(tensor);
        } else {
            // MobileNet Fallback
            const predictions = await model.classify(imagePreview);
            if (predictions.length > 0) {
                topClass = predictions[0].className.split(',')[0]; // MobileNet often has "pizza, pizza pie"
                confidence = predictions[0].probability;
            }
        }

        displayResult(topClass, confidence);
    } catch (e) {
        console.error('Prediction failed:', e);
        alert('Something went wrong during analysis. See console.');
        scanOverlay.classList.add('hidden');
    }
}

function displayResult(className, confidence) {
    // Hide scanning animation
    scanOverlay.classList.add('hidden');

    // Show Result Card
    resultCard.classList.remove('hidden');

    // Process Name
    // Replace underscores with spaces for display
    const displayName = className.replace(/_/g, ' ');

    document.getElementById('food-name').textContent = displayName;
    document.getElementById('confidence-badge').textContent = `${Math.round(confidence * 100)}% Match`;

    // Get Nutrition Info
    let cals = caloriesMap[className];
    let protein = 0;
    let carbs = 0;
    let fat = 0;

    // Check if we have specific nutrition data
    if (nutritionMap[className]) {
        const nut = nutritionMap[className];
        cals = nut.calories;
        protein = nut.protein;
        carbs = nut.carbs;
        fat = nut.fat;
    } else if (cals) {
        // Fallback if only cals are known (shouldn't happen with our new DB)
        // Estimate based on cals
        protein = Math.round((cals * 0.15) / 4);
        carbs = Math.round((cals * 0.50) / 4);
        fat = Math.round((cals * 0.35) / 9);
    } else {
        // Unknown, fallback for demo
        cals = Math.floor(Math.random() * (600 - 200) + 200);
        protein = Math.round((cals * 0.15) / 4);
        carbs = Math.round((cals * 0.50) / 4);
        fat = Math.round((cals * 0.35) / 9);
    }

    document.getElementById('calorie-count').textContent = cals;

    document.querySelector('.protein + span + .val').textContent = `${protein}g`;
    document.querySelector('.carbs + span + .val').textContent = `${carbs}g`;
    document.querySelector('.fat + span + .val').textContent = `${fat}g`;
}
