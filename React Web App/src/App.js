import React, { useState, useEffect } from 'react';
import './App.css';
import * as tf from '@tensorflow/tfjs';
import { loadLayersModel} from '@tensorflow/tfjs';


function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedPlant, setSelectedPlant] = useState('');
  const [prediction, setPrediction] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [isClearHovered, setIsClearHovered] = useState(false);

  const handleFileInputChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setUploadedImage(URL.createObjectURL(event.target.files[0]));
  };

  const handlePlantChange = (event) => {
    setSelectedPlant(event.target.value);
  };

  const loadModel = async (selectedPlant) => {
    try {
      const modelArchitecture = require(`../public/model/${selectedPlant}/model.json`);
      const modelWeights = require(`../public/model/${selectedPlant}/group1-shard1of1.bin`);
  
      const model = await loadLayersModel(
        tf.io.fromMemory(
          { modelTopology: modelArchitecture.modelTopology, weightSpecs: modelWeights }
        )
      );
      return model;
    } catch (error) {
      console.error('Error occurred while loading the model:', error);
      return null;
    }
  };
  

  const preprocessImage = async (file, batchSize) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => {
        const image = new Image();
        image.onload = () => {
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
  
          // Calculate the new dimensions while maintaining the aspect ratio
          const maxWidth = 224;
          const maxHeight = 224;
          let width = image.width;
          let height = image.height;
  
          const aspectRatio = width / height;
          if (width > maxWidth || height > maxHeight) {
            if (width / maxWidth > height / maxHeight) {
              width = maxWidth;
              height = width / aspectRatio;
            } else {
              height = maxHeight;
              width = height * aspectRatio;
            }
          }
  
          // Calculate the offset to center the image
          const offsetX = (maxWidth - width) / 2;
          const offsetY = (maxHeight - height) / 2;
  
          // Set the canvas dimensions and draw the resized image
          canvas.width = maxWidth;
          canvas.height = maxHeight;
          ctx.drawImage(image, offsetX, offsetY, width, height);
  
          // Convert the canvas image to a tensor with the specified batch size
          const tensor = tf.browser.fromPixels(canvas)
            .expandDims()
            .toFloat()
            .tile([batchSize, 1, 1, 3]); // Adjust the last dimension according to your model requirements (e.g., 1 for grayscale, 3 for RGB)
          resolve(tensor);
        };
        image.onerror = (error) => {
          reject(error);
        };
        image.src = reader.result;
      };
      reader.onerror = (error) => {
        reject(error);
      };
      reader.readAsDataURL(file);
    });
  };
  
  
  
  const predict = async (model, image) => {
    try {
      const predictions = await model.predict(image).data();
      return predictions;
    } catch (error) {
      console.error('Error occurred during prediction:', error);
      return 'Error occurred during prediction.';
    }
  };

  useEffect(() => {

    return () => {
      // Clean up the uploaded image URL when the component is unmounted
      if (uploadedImage) {
        URL.revokeObjectURL(uploadedImage);
      }
    };
  }, [uploadedImage]);

  const handlePredictClick = async () => {
    if (selectedFile && selectedPlant) {
      setIsLoading(true);

      const model = await loadModel(selectedPlant);
      if (model) {
        const image = await preprocessImage(selectedFile, 32);
        const predictions = await predict(model, image);
        setPrediction(predictions);
      } else {
        setPrediction('Error occurred during model loading.');
      }

      setIsLoading(false);
    } else {
      setPrediction('Please select an image and a plant.');
    }
  };

  const handleClearClick = () => {
    setSelectedFile(null);
    setSelectedPlant('');
    setPrediction('');
    setUploadedImage(null);
    const fileInput = document.getElementById('file-input');
    if (fileInput) {
      fileInput.value = ''; // Clear the input value
      fileInput.type = 'text'; // Change the input type temporarily to allow resetting the value
      fileInput.type = 'file'; // Reset the input type to file
    }
  };

  const handleClearHover = (isHovered) => {
    setIsClearHovered(isHovered);
  };

  return (
    <div className="container">
    <h1 className="title">Plant Disease Detection</h1>
    <div className="dropdown">
      <select value={selectedPlant} onChange={handlePlantChange}>
        <option value="">Select a plant</option>
        <option value="Potato">Potato</option>
        <option value="Cotton">Cotton</option>
        <option value="Tomato">Tomato</option>
      </select>
    </div>
    <input
      type="file"
      id="file-input" // Add the id attribute
      className="file-input"
      accept="image/*"
      onChange={handleFileInputChange}
    />
      {selectedFile && (
        <div className="image-container">
          <img src={uploadedImage} alt="Uploaded" className="uploaded-image" />
        </div>
      )}
    <div className="buttons-container">
    <button
        className="predict-btn"
        onClick={handlePredictClick}
        disabled={!selectedFile || !selectedPlant || isLoading}
      >
        {isLoading ? 'Predicting...' : 'Predict'}
      </button>
      {selectedFile && (
        <button
        className="clear-btn"
        onClick={handleClearClick}
        onMouseEnter={() => handleClearHover(true)}
        onMouseLeave={() => handleClearHover(false)}
        title="Clear"
        >
        {isClearHovered ? 'Clear' : 'x'}
      </button>
      )}
      
    </div>
      {prediction && <p className="result">Result: {prediction}</p>}
    </div>
  );
}

export default App;