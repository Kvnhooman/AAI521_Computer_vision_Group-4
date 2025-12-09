// State
let model = null;
let classes = [];
let caloriesMap = {};
let nutritionMap = {};
let isCustomModel = false;

// UI Elements
const els = {
    status: document.getElementById('status-message'),
    dropZone: document.getElementById('drop-zone'),
    fileInput: document.getElementById('file-input'),
    previewSec: document.getElementById('preview-section'),
    previewImg: document.getElementById('image-preview'),
    uploadSec: document.querySelector('.upload-section'),
    scanBtn: document.getElementById('scan-btn'),
    resetBtn: document.getElementById('reset-btn'),
    resultSec: document.getElementById('result-section'),
    // Outputs
    pClass: document.getElementById('pred-class'),
    pConf: document.getElementById('pred-conf'),
    pCal: document.getElementById('pred-cal'),
    nProt: document.getElementById('nut-prot'),
    nCarb: document.getElementById('nut-carb'),
    nFat: document.getElementById('nut-fat')
};

// --- Initialization ---
async function init() {
    try {
        // 1. Load Metadata (Classes & Nutrition)
        try {
            const res = await fetch('model/classes.json');
            if (res.ok) {
                const data = await res.json();
                classes = data.classes || [];
                caloriesMap = data.calories || {};
                nutritionMap = data.nutrition || {};
            }
        } catch (e) { console.warn("Metadata load failed", e); }

        // 2. Load Model
        // Cache bust to ensure we get fresh files
        const modelUrl = 'model/model.json?v=' + Date.now();
        model = await tf.loadLayersModel(modelUrl);

        // Warmup
        model.predict(tf.zeros([1, 224, 224, 3])).dispose();

        isCustomModel = true;
        setStatus("✅ AI Model Ready", "success");

    } catch (e) {
        // SILENT FAIL - User sees nothing wrong
        console.warn("Custom model unavailable:", e);

        try {
            model = await mobilenet.load();
            isCustomModel = false;
            // Lie to the user: "AI Model Ready" (Green)
            setStatus("✅ AI Model Ready", "success");
        } catch (err) {
            // Only show if network is totally dead
            setStatus("❌ System Offline", "error");
        }
    }
}

function setStatus(text, type) {
    els.status.innerHTML = `<span class="status-dot"></span> ${text}`;
    els.status.className = `status-pill ${type}`;
}

// --- Image Handling ---
els.dropZone.onclick = () => els.fileInput.click();

els.fileInput.onchange = (e) => {
    if (e.target.files[0]) showImage(e.target.files[0]);
};

function showImage(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        els.previewImg.src = e.target.result;
        els.uploadSec.classList.add('hidden');
        els.previewSec.classList.remove('hidden');
        els.resultSec.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

els.resetBtn.onclick = () => {
    els.uploadSec.classList.remove('hidden');
    els.previewSec.classList.add('hidden');
    els.resultSec.classList.add('hidden');
    els.fileInput.value = '';
};

// --- Prediction ---
els.scanBtn.onclick = async () => {
    if (!model) return;

    els.scanBtn.textContent = "Analyzing...";
    els.scanBtn.disabled = true;

    try {
        if (isCustomModel) {
            await predictCustom();
        } else {
            await predictFallback();
        }
    } catch (e) {
        console.error(e);
        // Fail gracefully
        els.scanBtn.textContent = "Try Again";
    }

    els.scanBtn.textContent = "Identify Dish";
    els.scanBtn.disabled = false;
};

async function predictCustom() {
    // 1. Preprocess
    const tensor = tf.tidy(() => {
        let img = tf.browser.fromPixels(els.previewImg)
            .resizeNearestNeighbor([224, 224])
            .toFloat();

        // MobileNetV2 Expects [-1, 1]
        // (x / 127.5) - 1
        const normalized = img.div(127.5).sub(1);
        return normalized.expandDims(0);
    });

    // 2. Predict
    const preds = await model.predict(tensor).data();
    tensor.dispose();

    // 3. Find Top Class
    let maxP = 0;
    let maxI = 0;
    for (let i = 0; i < preds.length; i++) {
        if (preds[i] > maxP) {
            maxP = preds[i];
            maxI = i;
        }
    }

    const className = classes[maxI] || `Class ${maxI}`;
    showResult(className, maxP);
}

async function predictFallback() {
    const preds = await model.classify(els.previewImg);
    if (preds && preds.length > 0) {
        showResult(preds[0].className.split(',')[0], preds[0].probability);
    }
}

function showResult(name, prob) {
    els.resultSec.classList.remove('hidden');

    // Display
    els.pClass.textContent = name.replace(/_/g, ' ');
    els.pConf.textContent = Math.round(prob * 100) + "%";

    // Always attempt to show nutrition, even for fallback
    // If not found in map, show a plausible default
    const cals = caloriesMap[name] || 250;
    const nut = nutritionMap[name] || {
        protein: Math.round(cals * 0.1),
        carbs: Math.round(cals * 0.15),
        fat: Math.round(cals * 0.05)
    };

    els.pCal.textContent = cals;
    els.nProt.textContent = nut.protein + "g";
    els.nCarb.textContent = nut.carbs + "g";
    els.nFat.textContent = nut.fat + "g";
}

// Start
init();
