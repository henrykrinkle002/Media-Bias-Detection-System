// // src/screens/Dashboard.js
// import React, { useState, useEffect } from 'react';
// import { Bar } from 'react-chartjs-2';
// import {
//   Chart as ChartJS,
//   BarElement,
//   CategoryScale,
//   LinearScale,
//   Title,
//   Tooltip,
//   Legend,
// } from 'chart.js';

// ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

// const API_BASE_URL = "https://public-feedback-system.onrender.com";

// const Dashboard = ({ url }) => {
//   const [analyticsData, setAnalyticsData] = useState({ No_omission_bias: 0, Omission_Bias: 0});
//   const [loading, setLoading] = useState(true);



//   const fetchAnalytics = async (urlName) => {
//     try {
//       const normalizedurlName = urlName.toLowerCase().trim();
//       const url = normalizedurlName
//         ? `${API_BASE_URL}/analytics/?url=${encodeURIComponent(normalizedurlName)}`
//         : `${API_BASE_URL}/analytics/`;
//       const res = await fetch(url);
//       const data = await res.json();
//       setAnalyticsData(data);
//       setLoading(false);
//     } catch (err) {
//       console.error('Error fetching analytics:', err);
//       setLoading(false);
//     }
//   };

//   useEffect(() => {
//     if (url && url.trim() !== "") {
//       fetchAnalytics(url);
//     } else {
//       setAnalyticsData({ No_omission_bias: 0, Omission_Bias: 0});
//     }
//   }, [url]);

//   const chartData = {
//     labels: Object.keys(entityCounts),
//     datasets: [{
//       label: 'Feedback Count',
//       data: [analyticsData.No_omission_bias, analyticsData.Omission_Bias],
//       backgroundColor: [
//         'rgba(75, 192, 192, 0.6)',
//         'rgba(255, 99, 132, 0.6)',
//         'rgba(255, 206, 86, 0.6)'
//       ],
//       borderColor: [
//         'rgba(75, 192, 192, 1)',
//         'rgba(255, 99, 132, 1)',
//         'rgba(255, 206, 86, 1)'
//       ],
//       borderWidth: 1,
//     }]
//   };

//   const options = {
//     responsive: true,
//     plugins: {
//       title: { display: true, text: url ? `Analytics for ${url}` : 'Global Analytics' },
//       legend: { display: false },
//     },
//     scales: { y: { beginAtZero: true, ticks: { precision: 0 } } },
//   };

//   return (
//     <div style={styles.container}>
//       {loading ? <p>Loading analytics...</p> : <Bar data={chartData} options={options} />}
//     </div>
//   );
// };

// const styles = {
//   container: { maxWidth: '800px', margin: '1rem auto', textAlign: 'center', fontFamily: 'Arial, sans-serif' },
// };

// export default Dashboard;


// src/screens/Dashboard.js
// src/screens/Dashboard.js
// src/screens/Dashboard.js
// src/screens/Dashboard.js
import React, { useEffect, useState } from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  BarElement,
  CategoryScale,
  LinearScale,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

const Dashboard = ({ entities1 = [], entities2 = [] }) => {
  const [chartData, setChartData] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const labels = ['Article 1', 'Article 2'];
    const dataValues = [entities1.length, entities2.length]; // total entities per article

    setChartData({
      labels,
      datasets: [
        {
          label: 'Number of Entities',
          data: dataValues,
          backgroundColor: ['rgba(75, 192, 192, 0.6)', 'rgba(255, 99, 132, 0.6)'],
          borderColor: ['rgba(75, 192, 192, 1)', 'rgba(255, 99, 132, 1)'],
          borderWidth: 1,
        },
      ],
    });

    setLoading(false);
  }, [entities1, entities2]);

  const options = {
    responsive: true,
    plugins: {
      title: { display: true, text: 'Entity Count per Article' },
      legend: { display: false },
      tooltip: { mode: 'index', intersect: false },
    },
    scales: {
      y: { beginAtZero: true, ticks: { precision: 0 } },
    },
  };

  return (
    <div style={styles.container}>
      {loading ? <p>Loading chart...</p> : <Bar data={chartData} options={options} />}
    </div>
  );
};

const styles = {
  container: { maxWidth: '600px', margin: '1rem auto', textAlign: 'center', fontFamily: 'Arial, sans-serif' },
};

export default Dashboard;
