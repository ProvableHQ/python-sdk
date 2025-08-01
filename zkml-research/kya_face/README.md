# KYA Workshop — Local Setup Guide

## 1  Prerequisites (install once)

| Tool | Windows | macOS | Linux (Ubuntu) |
|------|---------|-------|----------------|
| **Python 3.10 or 3.11 × 64-bit** | [python.org](https://www.python.org) installer | `brew install python@3.11` | `sudo apt install python3.11 python3.11-venv` |
| **C++ build tools**<br>(needed for `dlib` → `face_recognition`) | *Visual Studio Build Tools 2022* → “Desktop C++” workload | `xcode-select --install` | `sudo apt install build-essential` |
| **CMake ≥ 3.22** | [cmake.org](https://cmake.org) installer | `brew install cmake` | `sudo apt install cmake` |
| **Leo CLI** | Follow steps at <https://github.com/ProvableHQ/leo> | Follow steps at <https://github.com/ProvableHQ/leo> | Follow steps at <https://github.com/ProvableHQ/leo> |

---

## 2  Clone the repo & create a virtual env

```bash
git clone https://github.com/ProvableHQ/python-sdk.git
cd zkml-research/kya_face

# Create & activate a venv named .venv
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

---

## 3  Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## 4  Open the Jupyter notebook

Open `kya.ipynb`, e.g., through VS Code (choosing the `venv`).