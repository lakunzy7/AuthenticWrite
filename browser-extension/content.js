// AuthentiWrite Content Script
// Runs on web pages to capture selected text

// Listen for messages from popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'getSelection') {
    // Get selected text from the page
    const selection = window.getSelection();
    const selectedText = selection.toString().trim();
    
    sendResponse({ text: selectedText });
  }
  
  if (message.action === 'highlightAI') {
    // Optional: Highlight potential AI-generated sections
    highlightSuspiciousSections();
  }
});

function highlightSuspiciousSections() {
  const paragraphs = document.querySelectorAll('p');
  paragraphs.forEach(p => {
    const text = p.textContent || '';
    if (text.length < 200) return;

    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    if (sentences.length < 3) return;

    const lengths = sentences.map(s => s.trim().split(/\s+/).length);
    const mean = lengths.reduce((a, b) => a + b, 0) / lengths.length;
    const std = Math.sqrt(lengths.reduce((a, b) => a + (b - mean) ** 2, 0) / lengths.length);
    const cv = mean > 0 ? std / mean : 0;

    const transitions = ['furthermore', 'moreover', 'additionally', 'consequently', 'therefore'];
    const transCount = transitions.filter(t => text.toLowerCase().includes(t)).length;

    if (cv < 0.25 && transCount >= 2) {
      p.classList.add('authenticwrite-highlight');
    }
  });
}
