const labelColors = {
  Positive: '#3fe5b3',
  Neutral: '#8fd3ff',
  Negative: '#ff6a7d',
  Irrelevant: '#9ea7c3'
};

function flattenNumericArray(value) {
  if (value == null) return [];
  if (Array.isArray(value)) return value.flatMap((item) => flattenNumericArray(item));
  if (value instanceof Float32Array || value instanceof Float64Array || value instanceof Int32Array || value instanceof Uint8Array) {
    return Array.from(value);
  }
  if (value.data && typeof value.data !== 'string') {
    return Array.from(value.data).flatMap((item) => flattenNumericArray(item));
  }
  return [Number(value)];
}

const demoTweets = [
  'I love this new feature and the product launch was amazing! Great work team.',
  'This update is confusing and the experience feels slow and frustrating.',
  'The service is okay, not great, but not terrible either.',
  'We are announcing a beta release for testing with the community.'
];

let sentimentChart = null;
let modelSession = null;
let modelMeta = null;
let useFallbackModel = false;

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
      labels: ['Positive', 'Neutral', 'Negative', 'Irrelevant'],
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

function fallbackPredict(text) {
  const normalized = normalizeText(text);
  const positiveWords = ['love', 'amazing', 'great', 'excellent', 'good', 'happy', 'support', 'fast', 'nice', 'best', 'wow', 'success'];
  const negativeWords = ['bad', 'hate', 'terrible', 'slow', 'confusing', 'frustrating', 'broken', 'issue', 'error', 'worst', 'angry'];
  const neutralWords = ['okay', 'fine', 'average', 'moderate', 'maybe', 'not sure'];

  let positiveScore = 0;
  let negativeScore = 0;
  let neutralScore = 0;

  for (const word of positiveWords) {
    if (normalized.includes(word)) positiveScore += 1;
  }

  for (const word of negativeWords) {
    if (normalized.includes(word)) negativeScore += 1;
  }

  for (const word of neutralWords) {
    if (normalized.includes(word)) neutralScore += 1;
  }

  const scoreMap = {
    Positive: positiveScore,
    Neutral: neutralScore,
    Negative: negativeScore,
    Irrelevant: 0
  };

  const highestLabel = Object.keys(scoreMap).reduce((best, label) =>
    scoreMap[label] > scoreMap[best] ? label : best, 'Neutral');

  const total = Math.max(Object.values(scoreMap).reduce((sum, value) => sum + value, 0), 1);
  const byLabel = Object.fromEntries(
    Object.keys(scoreMap).map((label) => [label, (scoreMap[label] / total) * 100])
  );

  return {
    label: highestLabel,
    values: byLabel,
    confidence: byLabel[highestLabel] || 0
  };
}

async function loadModel() {
  if (modelSession) return;

  setStatus('loading', 'Loading model...');

  const baseUrl = window.location.href.includes('localhost') || window.location.protocol === 'file:'
    ? window.location.href.replace(/index\.html$/, '').replace(/\/$/, '')
    : window.location.href.replace(/index\.html$/, '');

  try {
    const response = await fetch(new URL('model/sentiment_model_meta.json', baseUrl).href);
    modelMeta = await response.json();
    modelSession = await ort.InferenceSession.create(new URL('model/sentiment_model.onnx', baseUrl).href);
    useFallbackModel = false;
    setStatus('success', 'Model ready');
    initChart();
  } catch (error) {
    console.warn('ONNX model unavailable in browser, switching to fallback prediction engine.', error);
    modelMeta = {
      labels: ['Positive', 'Neutral', 'Negative', 'Irrelevant']
    };
    useFallbackModel = true;
    setStatus('idle', 'Fallback mode');
    if (!sentimentChart) initChart();
  }
}

async function predictSentiment() {
  const text = textInput.value.trim();
  if (!text) {
    setStatus('idle', 'Type a tweet');
    return;
  }

  try {
    await loadModel();
    setStatus('loading', 'Analyzing...');

    if (useFallbackModel) {
      const fallback = fallbackPredict(text);
      const label = fallback.label;
      const values = fallback.values;

      sentimentLabel.textContent = label;
      sentimentLabel.style.color = labelColors[label] || '#eaf2ff';
      confidenceValue.textContent = `${Math.round(fallback.confidence)}%`;
      confidenceBar.style.width = `${fallback.confidence}%`;

      const percentByLabel = ['Positive', 'Neutral', 'Negative', 'Irrelevant'].map((item) => ({
        label: item,
        value: values[item] || 0
      }));

      sentimentChart.data.datasets[0].data = percentByLabel.map((item) => item.value);
      sentimentChart.update();

      document.getElementById('positive-value').textContent = `${Math.round(values.Positive || 0)}%`;
      document.getElementById('neutral-value').textContent = `${Math.round(values.Neutral || 0)}%`;
      document.getElementById('negative-value').textContent = `${Math.round(values.Negative || 0)}%`;
      document.getElementById('irrelevant-value').textContent = `${Math.round(values.Irrelevant || 0)}%`;

      setStatus('success', `${label} sentiment`);
      return;
    }

    const vector = computeCountVector(text, modelMeta.vocabulary_map);
    const input = new ort.Tensor('float32', vector, [1, vector.length]);
    const outputs = await modelSession.run({ input });

    const rawLabel = outputs[0]?.data ? outputs[0].data : outputs[0] || [];
    const rawProbabilities = outputs[1]?.data ? outputs[1].data : outputs[1] || [];

    const labelValue = Array.isArray(rawLabel) ? rawLabel[0] : String(rawLabel || modelMeta.labels[0]);
    const probabilities = flattenNumericArray(rawProbabilities);
    const safeProbabilities = probabilities
      .slice(0, modelMeta.labels.length)
      .map((value) => Number.isFinite(Number(value)) ? Number(value) : 0);

    const predictedIndex = safeProbabilities.indexOf(Math.max(...safeProbabilities, 0));
    const predictedLabel = modelMeta.labels[predictedIndex] || String(labelValue || 'Neutral');
    const confidence = Math.max(...safeProbabilities, 0) * 100;

    sentimentLabel.textContent = predictedLabel;
    sentimentLabel.style.color = labelColors[predictedLabel] || '#eaf2ff';
    confidenceValue.textContent = `${confidence.toFixed(0)}%`;
    confidenceBar.style.width = `${confidence}%`;

    const percentByLabel = modelMeta.labels.map((label, index) => ({
      label,
      value: (safeProbabilities[index] ?? 0) * 100
    }));

    sentimentChart.data.datasets[0].data = percentByLabel.map((item) => item.value);
    sentimentChart.update();

    document.getElementById('positive-value').textContent = `${(percentByLabel.find((item) => item.label === 'Positive')?.value || 0).toFixed(0)}%`;
    document.getElementById('neutral-value').textContent = `${(percentByLabel.find((item) => item.label === 'Neutral')?.value || 0).toFixed(0)}%`;
    document.getElementById('negative-value').textContent = `${(percentByLabel.find((item) => item.label === 'Negative')?.value || 0).toFixed(0)}%`;
    document.getElementById('irrelevant-value').textContent = `${(percentByLabel.find((item) => item.label === 'Irrelevant')?.value || 0).toFixed(0)}%`;

    setStatus('success', `${predictedLabel} sentiment`);
  } catch (error) {
    console.error('ONNX inference failed:', error?.message || error);
    setStatus('idle', 'Model error');
    sentimentLabel.textContent = 'Error';
    sentimentLabel.style.color = '#ff6a7d';
    confidenceValue.textContent = '0%';
    confidenceBar.style.width = '0%';
  }
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
