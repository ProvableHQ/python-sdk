from flask import Flask, send_from_directory, request, jsonify
from zkml import LeoTranspiler
import sklearn

app = Flask(__name__, static_folder='frontend', static_url_path='/static')
# ensure tmp folder in current directory exists
import os
tmp_path = os.path.join(os.getcwd(), "zkml-research", "zkmlaas", "tmp")
os.makedirs(tmp_path, exist_ok=True)

# remove all files in tmp folder
import shutil
for filename in os.listdir(tmp_path):
    file_path = os.path.join(os.getcwd(), "tmp", filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(f"Failed to delete {file_path}. Reason: {e}")

# Global lists for storing model information
model_names = []
model_data = []

# Dummy functions
def transpile(file_content):
    # Insert the transpilation logic here
    # load sklearn model. It is stored as clf in a pickle file
    import pickle
    model = pickle.loads(file_content)


    
    
    lt = LeoTranspiler(model=model)
    model_name = f"model_{len(model_names)}, {model.__class__.__name__}"
    leo_project_name = f"project_{len(model_names)}"
    lt.to_leo(path=tmp_path, project_name=leo_project_name)
    # Example: return "model_name", {"num_inputs": 3}
    return model_name, {"num_inputs": 3}

def proof(*args):
    # Insert the calculation logic here
    return "result"

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/transpileModel', methods=['POST'])
def transpile_model():
    if 'file' not in request.files:
        return "No file found", 400

    file = request.files['file']
    if file:
        model_name, data = transpile(file.read())
        model_names.append(model_name)
        model_data.append(data)
        print(model_names)
        return jsonify({"model_name": model_name, "model_data": data})
    else:
        print("No file found")
        return "No file found", 400

@app.route('/getModels', methods=['GET'])
def get_models():
    return jsonify(model_names)

@app.route('/proofInference', methods=['POST'])
def proof_inference():
    data = request.json
    # The following line is a placeholder, adjust it according to model_data
    result = proof(*data)
    return jsonify(result)

@app.route('/prepare_model_inference', methods=['GET'])
def prepare_model_inference():
    index = request.args.get('index', type=int)
    if index is not None and 0 <= index < len(model_data):
        num_inputs = model_data[index].get('num_inputs', 0)
        return jsonify(num_inputs)
    else:
        return "Invalid index or index out of range", 400

if __name__ == '__main__':
    app.run(debug=True)
