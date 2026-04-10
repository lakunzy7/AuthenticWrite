// AuthentiWrite Popup Script

document.addEventListener('DOMContentLoaded', () => {
  const analyzeBtn = document.getElementById('analyzeBtn');
  const resultDiv = document.getElementById('result');
  const loader = document.getElementById('loader');

  analyzeBtn.addEventListener('click', async () => {
    // Show loader
    loader.style.display = 'block';
    resultDiv.style.display = 'none';
    analyzeBtn.disabled = true;

    try {
      // Get selected text from active tab
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      
      const response = await chrome.tabs.sendMessage(tab.id, { action: 'getSelection' });
      
      if (!response || !response.text || response.text.length < 50) {
        showError('Please select more text (minimum 50 characters)');
        return;
      }

      // Send to backend for analysis
      const analysisResponse = await fetch('http://localhost:5000/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: response.text })
      });

      const data = await analysisResponse.json();
      
      if (analysisResponse.ok) {
        showResult(data);
      } else {
        showError(data.error || 'Analysis failed');
      }
    } catch (error) {
      showError('Could not connect to AuthentiWrite API. Make sure the app is running.');
    }
  });

  function showResult(data) {
    loader.style.display = 'none';
    resultDiv.style.display = 'block';
    
    const isAI = data.ai_probability >= 50;
    const prob = data.ai_probability;
    
    resultDiv.className = `result ${isAI ? 'ai' : 'human'}`;
    resultDiv.innerHTML = `
      <div class="result-score" style="color: ${isAI ? '#dc2626' : '#16a34a'}">${prob}%</div>
      <div class="result-label">${data.classification}</div>
      <p style="font-size: 12px; color: #6b7280; margin-top: 8px;">
        ${isAI ? 'This text appears to be AI-generated' : 'This text appears to be human-written'}
      </p>
    `;
    
    analyzeBtn.disabled = false;
  }

  function showError(message) {
    loader.style.display = 'none';
    resultDiv.style.display = 'block';
    resultDiv.className = 'result ai';
    resultDiv.innerHTML = `
      <p style="color: #dc2626; font-size: 14px;">${message}</p>
    `;
    analyzeBtn.disabled = false;
  }
});
