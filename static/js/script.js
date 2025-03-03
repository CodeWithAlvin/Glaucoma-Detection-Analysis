document.addEventListener('DOMContentLoaded', function() {
    const imageUpload = document.getElementById('image-upload');
    const fileNameDisplay = document.getElementById('file-name');
    const imagePreview = document.getElementById('image-preview');
    const previewContainer = document.getElementById('preview-container');
    const resultContainer = document.getElementById('result-container');
    const predictionResult = document.getElementById('prediction-result');
    const predictionForm = document.getElementById('prediction-form');
    const loadingContainer = document.getElementById('loading-container');

    // Show filename when file is selected
    imageUpload.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            fileNameDisplay.textContent = this.files[0].name;
            
            // Show image preview
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                previewContainer.style.display = 'block';
            }
            reader.readAsDataURL(this.files[0]);
            
            // Hide previous results
            resultContainer.style.display = 'none';
        } else {
            fileNameDisplay.textContent = 'No file selected';
            previewContainer.style.display = 'none';
        }
    });

    // Handle form submission
    predictionForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (!imageUpload.files || !imageUpload.files[0]) {
            alert('Please select an image first');
            return;
        }
        
        // Show loading spinner
        loadingContainer.style.display = 'flex';
        resultContainer.style.display = 'none';
        
        const formData = new FormData();
        formData.append('image', imageUpload.files[0]);
        
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Hide loading spinner
            loadingContainer.style.display = 'none';
            
            // Display result
            predictionResult.textContent = data.prediction_text;
            resultContainer.style.display = 'block';
            
            // Add class for styling based on result
            if (data.result === 1) {
                predictionResult.className = 'positive';
            } else {
                predictionResult.className = 'negative';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            loadingContainer.style.display = 'none';
            alert('There was an error processing your image. Please try again.');
        });
    });
});