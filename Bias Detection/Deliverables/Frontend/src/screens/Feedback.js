import React, { useState } from 'react';
import Dashboard from './Dashboard';
import axios from 'axios';

{/*const API_BASE_URL = "https://public-feedback-system.onrender.com";
*/}


const API_BASE_URL = 'http://localhost:8000'; 
const Feedback = () => {
  const [url, setUrl] = useState('');
  const [url2, setUrl2] = useState('')
  const [feedback, setFeedback1] = useState('');
  const [feedback2, setFeedback2] = useState('')
  const [loading, setLoading] = useState(false)
  const [prediction, setPrediction] = useState("");
  const [score, setScore] = useState(0.0)
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [entities1, setEntities1] = useState([]);
  const [entities2, setEntities2] = useState([]);
  const [dashboardKey, setDashboardKey] = useState(0);

const extractArticleGeneric = async (inputUrl, setFeedbackFn) => {
  setError('');
  setLoading(true);

  try { 
    const response = await axios.get(`${API_BASE_URL}/extract`, {
      params: { url: inputUrl }
    });

    const data = response.data;

    if (!data.article || data.article.trim() === '') {
      setError('No article content extracted. The URL might be unsupported or empty.');
      return;
    }

    setFeedbackFn(data.article);
  } catch (err) {
    console.error(err);
    setError('An error occurred while extracting the article.');
  } finally {
    setLoading(false);
  }
};


  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setResult(null);
    setScore(0.0)

    if (!feedback.trim() || !feedback2.trim()) {
      setError('No article content to analyze. Make sure the URL is valid and article was extracted.');
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/predict-omission/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          article_text_1: feedback,
          article_text_2: feedback2 }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status} ${response.statusText}`);
      }

      let data;
      try {
            data = await response.json();
         } 
      catch (parseErr) {
      throw new Error('Invalid JSON returned from server.');
           }

     setResult(data.bias_result);
     setPrediction(data.prediction_agreement);
     setScore(data.cosine_similarity);

     setEntities1(data.entities_article_1 || []);
     setEntities2(data.entities_article_2 || []);

      if (data.alertSent) {
        window.alert("Alert notification sent to company admin due to high negative feedback.");
      }

      setDashboardKey(prevKey => prevKey + 1);
    } catch (err) {
      setError(err.message);
    }
  };

  const handleResetDashboard = async () => {
    try {
      const normalizedurl = url.toLowerCase().trim();
      const response = await fetch(`${API_BASE_URL}/reset/?url=${encodeURIComponent(normalizedurl)}`, {
        method: 'POST',
      });
      if (!response.ok) {
        throw new Error(`Error resetting dashboard: ${response.status} ${response.statusText}`);
      }
      setDashboardKey(prevKey => prevKey + 1);
    } catch (err) {
      setError(err.message);
    }
  };



  return (
    <div style={styles.container}>
      <h1>Analyze Article Feedback</h1>
      <form onSubmit={handleSubmit} style={styles.form}>
        <label htmlFor="url" style={styles.label}>Enter Article URL:</label>
        <input
          id="url"
          type="url"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          style={styles.input}
          placeholder="https://example.com/article"
          required
        />

        <button type="button" style={{ ...styles.button, marginBottom: '1rem' }} onClick={()=>{extractArticleGeneric(url, setFeedback1)}}>
          Extract Article
        </button>

        <label htmlFor="feedback" style={styles.label}>Article Preview:</label>
        <textarea
          id="feedback"
          value={feedback}
          readOnly
          rows="8"
          style={{ ...styles.textarea, backgroundColor: '#f9f9f9' }}
        />

        <label htmlFor='url2' style={styles.label}>Enter Reference Article URL:</label>
        <input
        id="url2"
        type="url"
        value={url2}
        onChange={(e) => setUrl2(e.target.value)}
        style={styles.input}
        placeholder="https://example.com/second-article"
        required      
        />

        <button 
          type="button" 
          style={{ ...styles.button, marginBottom: '1rem' }} 
          onClick={() => extractArticleGeneric(url2, setFeedback2)}
        >
          Extract Reference Article
        </button>

      <label htmlFor="feedback2" style={styles.label}>Reference Article Preview:</label>
      <textarea
        id="feedback2"
        value={feedback2}
        readOnly
        rows="8"
        style={{ ...styles.textarea, backgroundColor: '#f9f9f9' }}
      />      

      <button type="submit" style={styles.button}>Analyze</button>
        
    </form>

      {error && <p style={styles.error}>Error: {error}</p>}
      {result !== null && (
        <div style={styles.result}>
          <h2>Analysis Result</h2>
          <p><strong>Tag:</strong> {result}</p>
          <p><strong>Score:</strong> {score}</p>

        </div>
      )}

      {url.trim() !== "" && (
        <div style={styles.dashboardContainer}>
          <h2>Analytics Dashboard</h2>
          <p><strong>Prediction:</strong> {prediction}</p>
  
          <Dashboard 
  key={dashboardKey} 
  entities1={entities1}
  entities2={entities2}
/>
          <button onClick={handleResetDashboard} style={{ ...styles.button, marginTop: '1rem', backgroundColor: '#dc3545' }}>
            Reset Dashboard
          </button>
        </div>
      )}
    </div>
    
  );
};

const styles = {
  container: { maxWidth: '600px', margin: '2rem auto', padding: '1rem', fontFamily: 'Arial, sans-serif' },
  form: { marginTop: '1rem' },
  label: { fontSize: '1.1rem', display: 'block', marginBottom: '0.5rem' },
  input: { width: '100%', padding: '0.75rem', fontSize: '1rem', borderRadius: '5px', border: '1px solid #ccc', marginBottom: '1rem' },
  textarea: { width: '100%', padding: '0.75rem', fontSize: '1rem', borderRadius: '5px', border: '1px solid #ccc', marginBottom: '1rem' },
  button: { padding: '0.75rem 1.5rem', fontSize: '1rem', backgroundColor: '#007BFF', color: '#fff', border: 'none', borderRadius: '5px', cursor: 'pointer' },
  error: { color: 'red', marginTop: '1rem' },
  result: { marginTop: '2rem', padding: '1rem', border: '1px solid #ccc', borderRadius: '5px' },
  dashboardContainer: { marginTop: '2rem', padding: '1rem', border: '1px solid #ccc', borderRadius: '5px' },
};

export default Feedback;
