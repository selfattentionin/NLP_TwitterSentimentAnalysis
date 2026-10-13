const orderedLabels = ['Positive', 'Neutral', 'Negative', 'Irrelevant'];

const labelColors = {
  Positive: '#3fe5b3',
  Neutral: '#8fd3ff',
  Negative: '#ff6a7d',
  Irrelevant: '#9ea7c3'
};

const demoTweets = [
  'I love this new feature and the product launch was amazing! Great work team.',
  'This update is confusing and the experience feels slow and frustrating.',
  'The service is okay, not great, but not terrible either.',
  'We are announcing a beta release for testing with the community.'
];

let sentimentChart = null;
let modelSession = null;
let modelMeta = null;

const textInput = document.getElementById('tweet-input');
const analyzeBtn = document.getElementById('analyze-btn');
const demoBtn = document.getElementById('demo-btn');
const statusBadge = document.getElementById('status-badge');
const sentimentLabel = document.getElementById('sentiment-label');
const confidenceValue = document.getElementById('confidence-value');
const confidenceBar = document.getElementById('confidence-bar');

function setStatus(state, text) {
  statusBadge.className = `status-badge ${state}`;
  statusBadge.textContent = text;
}

function initChart() {
  const ctx = document.getElementById('sentiment-chart');
  sentimentChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: orderedLabels,
      datasets: [{
        data: [25, 25, 25, 25],
        backgroundColor: ['#3fe5b3', '#8fd3ff', '#ff6a7d', '#9ea7c3'],
        borderWidth: 0,
        hoverOffset: 8
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '68%',
      plugins: {
        legend: {
          labels: {
            color: '#eaf2ff',
            usePointStyle: true,
            boxWidth: 10
          }
        }
      }
    }
  });
}

function normalizeText(value) {
  return value
    .toLowerCase()
    .replace(/https?\S+|www\.\S+/g, ' ')
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function computeCountVector(text, vocabularyMap) {
  const vector = new Float32Array(Object.keys(vocabularyMap).length || 1);
  const tokens = normalizeText(text).split(' ').filter(Boolean);

  for (const token of tokens) {
    const index = vocabularyMap[token];
    if (typeof index === 'number') {
      vector[index] += 1;
    }
  }

  return vector;
}

function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - max));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

async function loadModel() {
  if (modelSession) return;

  setStatus('loading', 'Loading model...');

  const response = await fetch('model/sentiment_model_meta.json');
  modelMeta = await response.json();

  modelSession = await ort.InferenceSession.create('model/sentiment_model.onnx');
  setStatus('success', 'Model ready');
  initChart();
}

async function predictSentiment() {
  const text = textInput.value.trim();
  if (!text) {
    setStatus('idle', 'Type a tweet');
    return;
  }

  await loadModel();
  setStatus('loading', 'Analyzing...');

  const vector = computeCountVector(text, modelMeta.vocabulary_map);
  const input = new ort.Tensor('float32', vector, [1, vector.length]);
  const outputs = await modelSession.run({ input });
  const labelOutput = outputs[0] ?? [];
  const probabilityOutput = outputs[1] ?? [];
  const probabilitiesFlat = Array.isArray(probabilityOutput)
    ? probabilityOutput.flat()
    : Array.from(probabilityOutput.data || probabilityOutput);

  const predictedLabel = Array.isArray(labelOutput) && labelOutput.length
    ? labelOutput[0]
    : modelMeta.labels[0];

  const labelIndex = modelMeta.labels.indexOf(predictedLabel);
  const probabilities = probabilitiesFlat.length ? probabilitiesFlat : new Array(modelMeta.labels.length).fill(1 / modelMeta.labels.length);
  const confidence = Math.max(...probabilities) * 100;

  if (labelIndex === -1) {
    const fallbackIndex = probabilities.indexOf(Math.max(...probabilities));
    const fallbackLabel = modelMeta.labels[fallbackIndex];
    sentimentLabel.textContent = fallbackLabel;
    sentimentLabel.style.color = labelColors[fallbackLabel] || '#eaf2ff';
  }

  sentimentLabel.textContent = predictedLabel;
  sentimentLabel.style.color = labelColors[predictedLabel] || '#eaf2ff';
  confidenceValue.textContent = `${confidence.toFixed(0)}%`;
  confidenceBar.style.width = `${confidence}%`;

  const percentByLabel = orderedLabels.map((label) => ({
    label,
    value: probabilities[modelMeta.labels.indexOf(label)] * 100
  }));

  sentimentChart.data.datasets[0].data = percentByLabel.map((item) => item.value);
  sentimentChart.update();

  document.getElementById('positive-value').textContent = `${(percentByLabel.find((item) => item.label === 'Positive')?.value || 0).toFixed(0)}%`;
  document.getElementById('neutral-value').textContent = `${(percentByLabel.find((item) => item.label === 'Neutral')?.value || 0).toFixed(0)}%`;
  document.getElementById('negative-value').textContent = `${(percentByLabel.find((item) => item.label === 'Negative')?.value || 0).toFixed(0)}%`;
  document.getElementById('irrelevant-value').textContent = `${(percentByLabel.find((item) => item.label === 'Irrelevant')?.value || 0).toFixed(0)}%`;

  setStatus('success', `${predictedLabel} sentiment`);
}

analyzeBtn.addEventListener('click', predictSentiment);
demoBtn.addEventListener('click', () => {
  const nextDemo = demoTweets[Math.floor(Math.random() * demoTweets.length)];
  textInput.value = nextDemo;
  predictSentiment();
});

textInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
    predictSentiment();
  }
});

window.addEventListener('load', () => {
  setStatus('idle', 'Ready');
  textInput.value = demoTweets[0];
  initChart();
});
