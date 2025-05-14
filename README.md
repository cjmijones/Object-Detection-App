# 🤟 Real-Time ASL Letter & Hand Detection

A Streamlit web-app that

* **Section A** – uses a *stock* YOLOv8m model to draw bounding boxes around hands
* **Section B** – uses a fine-tuned YOLOv8 model **plus** a ResNet-18 classifier to
  predict the American Sign Language (ASL) letter being shown

<p align="center">
  <img src="docs/demo.gif" width="600" alt="demo">
</p>

---

## 1. Quick start (local)

```bash
git clone https://github.com/YOUR_USER/asl-detector.git
cd asl-detector

# create & activate an isolated environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# install python packages
pip install -r requirements.txt

# run the app
streamlit run streamlit_app.py
