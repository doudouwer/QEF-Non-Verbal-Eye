# QEF-Non-Verbal-Eye

`QEF-Non-Verbal-Eye` is a submodule of the QEF project designed to detect human eye gaze direction. This module uses a deep learning model to estimate where the eyes are looking (e.g., up, down, left, right) and exposes the functionality via an API service.

## 📦 Install Dependencies

Make sure you have Python 3.8+ and pip installed.

Then, run the following command in the project root to install the dependencies:

```bash
pip install -e .
```

or

```bash
pip install -r install_requirements.txt
```

## 📥 Download Pre-trained Models

Download the pre-trained model weights from the following link and place them in the `models/` directory:

> **Download link**: https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd

After downloading, make sure your `models/` folder looks like this:

```
QEF-Non-Verbal-Eye/
├── models/
│   └── [your_model_files.pth or similar]
```

## 🚀 Launch FastAPI Service

Run the following command to start the FastAPI service:

```bash
python main.py
```

By default, the service will be available at `http://127.0.0.1:8000`. You can view and test the API through the Swagger UI:

```
http://127.0.0.1:8000/docs
```