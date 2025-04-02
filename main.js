import React from "react";

function App() {
  return (
    <div className="min-h-screen bg-gray-100">
      {/* Navigation Bar */}
      <nav className="bg-blue-600 p-4 text-white shadow-lg sticky top-0 z-10">
        <div className="container mx-auto flex justify-between items-center">
          <h1 className="text-2xl font-bold">EEG Signal Analysis</h1>
          <ul className="flex space-x-6">
            <li><a href="#home" className="hover:text-gray-300">Home</a></li>
            <li><a href="#about" className="hover:text-gray-300">About</a></li>
            <li><a href="#contact" className="hover:text-gray-300">Contact</a></li>
          </ul>
        </div>
      </nav>

      {/* Page Content */}
      <div className="container mx-auto p-6 space-y-16">
        {/* Home Section */}
        <section id="home" className="bg-white p-8 rounded-lg shadow">
          <h2 className="text-2xl font-bold mb-6">EEG Analysis Tool</h2>
          <div className="mb-8">
            <h3 className="text-xl font-semibold mb-4">Interactive EEG Analysis</h3>
            <p className="mb-4">Click the button below to access our advanced EEG analysis tool powered by Streamlit:</p>
            <a 
              href="https://your-streamlit-app-url.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="inline-block bg-green-500 text-white px-6 py-3 rounded-lg hover:bg-green-600 transition-colors"
            >
              Launch EEG Analyzer
            </a>
          </div>
          
          <div>
            <h3 className="text-xl font-semibold mb-4">Features</h3>
            <ul className="list-disc pl-5 space-y-2">
              <li>Real-time EEG signal visualization</li>
              <li>Frequency band analysis</li>
              <li>Anomaly detection</li>
              <li>Interactive plots and metrics</li>
            </ul>
          </div>
        </section>

        {/* About Section */}
        <section id="about" className="bg-white p-8 rounded-lg shadow">
          <h2 className="text-2xl font-bold mb-6">About</h2>
          <div className="space-y-4">
            <p>
              This platform provides advanced EEG signal analysis tools for researchers and clinicians.
              Our Streamlit-based analyzer offers interactive visualization and processing capabilities.
            </p>
            <p>
              The tool is designed to help identify patterns, detect anomalies, and analyze frequency components
              in electroencephalography (EEG) data.
            </p>
          </div>
        </section>

        {/* Contact Section */}
        <section id="contact" className="bg-white p-8 rounded-lg shadow">
          <h2 className="text-2xl font-bold mb-6">Contact Us</h2>
          <div className="space-y-4">
            <p>
              For questions or support, please reach out to our team:
            </p>
            <ul className="space-y-2">
              <li>Email: <a href="mailto:support@eeganalysis.com" className="text-blue-500">support@eeganalysis.com</a></li>
              <li>Phone: +1 (555) 123-4567</li>
              <li>Address: 123 Neuroscience Way, Research City, RC 98765</li>
            </ul>
          </div>
        </section>
      </div>

      {/* Footer */}
      <footer className="bg-blue-600 text-white p-4 mt-8">
        <div className="container mx-auto text-center">
          <p>© {new Date().getFullYear()} EEG Signal Analysis Tool. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}

export default App;