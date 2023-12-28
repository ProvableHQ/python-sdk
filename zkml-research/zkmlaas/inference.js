document.addEventListener("DOMContentLoaded", function() {
    // Assumption: modelIndex is passed via the URL or another method
    const modelIndex = 0; // Example value
    prepareInputFields(modelIndex);

    document.getElementById('proofButton').addEventListener('click', function() {
        sendInferenceData(modelIndex);
    });
});

function prepareInputFields(index) {
    fetch(`/prepare_model_inference?index=${index}`)
        .then(response => response.json())
        .then(numInputs => {
            const inputFields = document.getElementById('inputFields');
            inputFields.innerHTML = '';
            for (let i = 0; i < numInputs; i++) {
                const input = document.createElement('input');
                input.type = 'text';
                input.placeholder = `Value ${i}`;
                input.id = `input${i}`;
                inputFields.appendChild(input);
            }
        });
}

function sendInferenceData(index) {
    // Here the logic for collecting the input values and sending them to the server is implemented
}
