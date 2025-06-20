document.addEventListener('DOMContentLoaded', function() {
    const imageInput = document.getElementById('imageInput');
    const preview = document.getElementById('preview');
    const previewImage = document.getElementById('previewImage');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const predictionResult = document.getElementById('prediction-result');

    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                preview.style.display = 'block';
                results.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }
    });

    analyzeBtn.addEventListener('click', async function() {
        const file = imageInput.files[0];
        if (!file) return;

        loading.style.display = 'block';
        results.style.display = 'none';

        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                displayResults(result);
            } else {
                alert('Error: ' + result.error);
            }
        } catch (error) {
            console.error('Error:', error);
            alert('Failed to analyze image. Please try again.');
        } finally {
            loading.style.display = 'none';
        }
    });

    function displayResults(result) {
        const confidence = Math.round(result.confidence * 100);
        predictionResult.innerHTML = `
            <h4>${result.class_name}</h4>
            <p>Confidence: ${confidence}%</p>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${confidence}%"></div>
            </div>
        `;
        results.style.display = 'block';
    }
});
