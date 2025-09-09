import './App.css';
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from '/Users/amalkurian/Desktop/Dissertation/Bias Detection/Deliverables/Frontend/src/screens/Home.js';
import Feedback from '/Users/amalkurian/Desktop/Dissertation/Bias Detection/Deliverables/Frontend/src/screens/Feedback.js';
import Dashboard from './screens/Dashboard';

const App = () => {
return (
  <>
      <Router>
        <Routes>
          // <Route path="/" element={<Home/>} />
          <Route path="/Feedback" element={<Feedback/>}/>
          <Route path="/Dashboard" element={<Dashboard/>}/>
        </Routes>
      </Router>
  </>
);
}

export default App;
