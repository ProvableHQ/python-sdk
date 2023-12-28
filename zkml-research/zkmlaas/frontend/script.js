document.addEventListener("DOMContentLoaded", function() {
    loadModels();

    const uploadArea = document.getElementById('uploadArea');
    uploadArea.addEventListener('click', function() {
        document.getElementById('fileInput').click();
    });

    document.getElementById('fileInput').addEventListener('change', function(event) {
        const file = event.target.files[0];
        uploadFile(file);
    });
});

function loadModels() {
    fetch('/getModels')
        .then(response => response.json())
        .then(data => {
            const modelList = document.getElementById('modelList');
            if (data.length > 0) {
                modelList.innerHTML = data.map(model => `<li>${model}</li>`).join('');
            } else {
                modelList.innerHTML = 'No transpiled models yet.';
            }
        });
}

function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    fetch('/transpileModel', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        console.log('Success:', data);
        loadModels(); // Reload list
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}
