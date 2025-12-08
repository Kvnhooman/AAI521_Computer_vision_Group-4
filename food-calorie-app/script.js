// ---------- DOM ELEMENTS ----------
const fileInput = document.getElementById("image-upload");
const previewImg = document.getElementById("preview-image");
const hiddenCanvas = document.getElementById("hidden-canvas");
const predictButton = document.getElementById("predict-button");
const statusEl = document.getElementById("status");
const resultsSection = document.getElementById("results");
const dishNameEl = document.getElementById("dish-name");
const confidenceEl = document.getElementById("confidence");
const calorieEl = document.getElementById("calorie-estimate");
const top3El = document.getElementById("top3");

let model = null;
let classNames = [];
let imageReady = false;

// ---------- INITIALIZATION ----------
async function loadModelAndClasses() {
  try {
    statusEl.textContent = "Loading model files...";
    // Load model
    model = await tf.loadLayersModel("model/model.json");

    // Load classes
    const resp = await fetch("model/classes.json");
    const data = await resp.json();
    if (Array.isArray(data)) {
      classNames = data;
    } else if (Array.isArray(data.classes)) {
      classNames = data.classes;
    } else {
      throw new Error("Invalid classes.json format");
    }

    if (!Array.isArray(classNames) || classNames.length === 0) {
      throw new Error("Class list is empty");
    }

    predictButton.disabled = false;
    predictButton.textContent = "Analyze photo";
    statusEl.textContent = "Model loaded. Choose a photo to get started.";
  } catch (err) {
    console.error("Error loading model or classes:", err);
    statusEl.textContent =
      "Error loading model. Check the console and that the model/ folder is in the correct place.";
  }
}

// ---------- IMAGE HANDLING ----------
fileInput.addEventListener("change", (event) => {
  const file = event.target.files && event.target.files[0];
  if (!file) {
    return;
  }

  const objectUrl = URL.createObjectURL(file);
  previewImg.style.display = "block";
  previewImg.src = objectUrl;
  imageReady = false;

  // Hide previous results
  resultsSection.classList.add("hidden");
  statusEl.textContent = "Loading image...";

  previewImg.onload = () => {
    URL.revokeObjectURL(objectUrl);
    imageReady = true;
    statusEl.textContent = "Image ready. Tap 'Analyze photo'.";
  };

  previewImg.onerror = () => {
    URL.revokeObjectURL(objectUrl);
    imageReady = false;
    statusEl.textContent = "Could not load the image. Try another photo.";
  };
});

// ---------- PREDICTION PIPELINE ----------
predictButton.addEventListener("click", async () => {
  if (!model) {
    statusEl.textContent = "Model is still loading. Please wait a moment.";
    return;
  }
  if (!imageReady || !previewImg.src) {
    statusEl.textContent = "Please choose or take a photo first.";
    return;
  }

  try {
    predictButton.disabled = true;
    predictButton.textContent = "Analyzing...";
    statusEl.textContent = "Analyzing image...";

    const prediction = await runInferenceOnCurrentImage();

    const topIndex = prediction.topIndex;
    const topProb = prediction.topProb;
    const topK = prediction.topK;

    const label =
      classNames[topIndex] !== undefined ? classNames[topIndex] : `Class #${topIndex}`;
    const prettyLabel = formatLabel(label);

    dishNameEl.textContent = `I think this is: ${prettyLabel}`;
    confidenceEl.textContent = `Confidence: ${(topProb * 100).toFixed(1)}%`;

    const calorieInfo = estimateCalories(label);
    calorieEl.textContent = `Estimated calories: ${calorieInfo}`;

    top3El.innerHTML = buildTop3Html(topK);

    resultsSection.classList.remove("hidden");
    statusEl.textContent = "Done.";
  } catch (err) {
    console.error("Prediction error:", err);
    statusEl.textContent =
      "Something went wrong while running the model. Check the console for details.";
  } finally {
    predictButton.disabled = false;
    predictButton.textContent = "Analyze photo";
  }
});

async function runInferenceOnCurrentImage() {
  const size = 224;
  const canvas = hiddenCanvas;
  const ctx = canvas.getContext("2d");

  canvas.width = size;
  canvas.height = size;

  // Draw the preview image into the hidden canvas, resized to 224x224
  ctx.drawImage(previewImg, 0, 0, size, size);

  // Preprocessing that matches your notebook:
  // - cast to float32
  // - scale to [-1, 1] using (x / 127.5) - 1.0
  return tf.tidy(() => {
    const imgTensor = tf.browser.fromPixels(canvas).toFloat();
    const offset = tf.scalar(127.5);
    const normalized = imgTensor.div(offset).sub(tf.scalar(1.0));
    const batched = normalized.expandDims(0); // [1, 224, 224, 3]

    const logits = model.predict(batched); // [1, numClasses]
    const data = logits.dataSync();
    const probs = Array.from(data);

    // Get top-1 index and probability
    let bestIndex = 0;
    let bestProb = probs[0];
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > bestProb) {
        bestProb = probs[i];
        bestIndex = i;
      }
    }

    // Top-3
    const ranked = probs
      .map((p, idx) => ({ index: idx, prob: p }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 3);

    return {
      topIndex: bestIndex,
      topProb: bestProb,
      topK: ranked,
    };
  });
}

// ---------- HELPERS ----------
function formatLabel(label) {
  // Convert "french_fries" -> "French Fries"
  if (!label || typeof label !== "string") return label;
  return label
    .split("_")
    .map((word) => (word ? word[0].toUpperCase() + word.slice(1) : ""))
    .join(" ");
}

function buildTop3Html(topK) {
  if (!Array.isArray(topK) || topK.length === 0) {
    return "";
  }

  const parts = topK.map((item, idx) => {
    const label =
      classNames[item.index] !== undefined
        ? classNames[item.index]
        : `Class #${item.index}`;
    const pretty = formatLabel(label);
    const pct = (item.prob * 100).toFixed(1);
    return `${idx + 1}. ${pretty} (${pct}%)`;
  });

  return `<strong>Top 3 guesses:</strong><br>${parts.join("<br>")}`;
}

// Rough calorie ranges in kcal for a "typical" serving.
// These are approximate and intentionally conservative.
const CALORIE_TABLE = {
  pizza: "250–400 kcal per slice (regular crust, with cheese & toppings)",
  hamburger: "300–600 kcal per medium burger with toppings (no fries)",
  sushi: "200–400 kcal per typical 6–8 piece roll",
  ice_cream: "150–300 kcal per 1/2 cup serving",
  french_fries: "300–450 kcal per medium serving",
  donuts: "200–350 kcal per donut",
  apple_pie: "300–450 kcal per slice",
  chocolate_cake: "350–550 kcal per slice with frosting",
  hot_dog: "250–450 kcal per hot dog with bun",
  pancakes: "250–450 kcal for 2–3 medium pancakes with syrup",
  tacos: "150–250 kcal per taco (depends a lot on filling)",
  spaghetti_bolognese: "400–700 kcal per plate",
  chicken_wings: "80–120 kcal per wing (without dip)",
  fried_rice: "350–650 kcal per plate",
  nachos: "400–800 kcal per plate (shared appetizer)",
  waffles: "250–450 kcal per waffle with toppings",
  cheesecake: "350–600 kcal per slice",
  caesar_salad: "300–600 kcal per bowl (with dressing & croutons)",
  greek_salad: "250–450 kcal per bowl (with feta & dressing)",
  caprese_salad: "200–350 kcal per plate",
  fried_chicken: "250–400 kcal per piece (leg/thigh)",
  burrito: "500–900 kcal per burrito",
  ramen: "450–800 kcal per bowl",
  pad_thai: "500–900 kcal per plate",
  dumplings: "40–80 kcal per dumpling",
  beef_tartare: "250–400 kcal per serving",
  ceviche: "150–300 kcal per serving",
  bruschetta: "80–150 kcal per piece",
  hummus: "70–100 kcal per 2 Tbsp",
  guacamole: "50–100 kcal per 2 Tbsp",
  fish_and_chips: "700–1200 kcal per restaurant plate",
  chicken_curry: "400–800 kcal per serving (without rice)",
  steak: "400–900 kcal per steak depending on cut & size",
  pork_chop: "250–450 kcal per chop",
  lobster_bisque: "250–450 kcal per bowl",
  scallops: "150–300 kcal per serving (about 6 large)",
  oysters: "50–80 kcal per 6 oysters",
  red_velvet_cake: "350–550 kcal per slice",
  carrot_cake: "350–600 kcal per slice",
  bread_pudding: "350–650 kcal per serving",
};

function estimateCalories(labelRaw) {
  const label = (labelRaw || "").toLowerCase();
  const match = CALORIE_TABLE[label];

  if (match) {
    return match;
  }

  // Fallback when we don't have a specific entry.
  return "Roughly 200–800 kcal per typical serving (highly recipe and portion dependent).";
}

// Start loading model as soon as the script runs
loadModelAndClasses();