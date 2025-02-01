import React, { useState } from 'react';

const SentimentAnalyzer = () => {
  const [userInput, setUserInput] = useState('');
  const [selectedModel, setSelectedModel] = useState("custom");
  const [result, setResult] = useState(null);

  const handleAnalyze = async () => {
    try {
      const response = await fetch('http://localhost:5000/analyze/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: userInput, model: selectedModel }),
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className="max-w-md mx-auto p-4 space-y-4">
      <div className="flex items-center">
        <select
          className="border rounded-lg px-4 py-2 mr-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
        >
          <option value="custom">Custom Model</option>
          <option value="llama">Llama 3</option>
        </select>

        <input
          type="text"
          className="w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          placeholder="Enter your text here"
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
        />
      </div>

      <button
        className="w-full px-4 py-2 text-white bg-blue-500 rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
        onClick={handleAnalyze}
      >
        Analyze Sentiment
      </button>

      {result && (
        <div className="border rounded-lg p-4 bg-white shadow">
          <p className="mb-2">
            <strong>Sentiment:</strong> {result.sentiment}
          </p>
          {result.confidence && (
            <p>
              <strong>Confidence Score:</strong> {result.confidence}%
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default SentimentAnalyzer;