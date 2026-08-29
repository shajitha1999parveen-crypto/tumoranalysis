# Brain Tumor MRI Classifier (CNN)

A convolutional neural network that classifies brain MRI images as
**tumor** vs **no tumor**, trained on the Kaggle
`paultimothymooney/brain-tumor-detection` dataset. Deployed with Streamlit.

**Scope note:** this is a binary image classifier only. It does not predict
survival time, recurrence after surgery, or future risk in currently-healthy
patients — those are separate problems requiring longitudinal clinical data
and survival-analysis models (e.g. Cox proportional hazards, time-to-event
modeling), not a single-image CNN.

## Project structure

```
brain-tumor-classifier/
├── app.py            # Streamlit UI
├── train_model.py    # Training script
├── requirements.txt
├── data/             # train images: data/tumor/, data/no_tumor/
└── models/           # generated: tumor_model.h5, class_names.txt
```

## 1. Get the data

```bash
pip install kaggle
kaggle datasets download -d paultimothymooney/brain-tumor-detection --unzip -p data
```

(Requires a Kaggle API token in `~/.kaggle/kaggle.json`.)

## 2. Set up locally (VS Code)

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 3. Train

```bash
python train_model.py --data_dir data --epochs 15
```

## 4. Run locally

```bash
streamlit run app.py
```

## 5. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: brain tumor MRI classifier"
git branch -M main
git remote add origin https://github.com/<your-username>/brain-tumor-classifier.git
git push -u origin main
```

> MRI datasets and `.h5` model files can be large — use `.gitignore` (included)
> and Git LFS (`git lfs track "*.h5"`) if needed, or host the trained weights
> separately and download them at app startup.

## 6. Deploy on Streamlit Community Cloud

1. https://share.streamlit.io → sign in with GitHub.
2. **New app** → select this repo → branch `main` → main file `app.py`.
3. Deploy.

## Model details

- Input: 128×128 RGB MRI images, rescaled to [0,1]
- Architecture: 3× (Conv2D + MaxPooling2D) → Flatten → Dense(128, relu) → Dropout(0.3) → Dense(1, sigmoid)
- Optimizer: Adam, Loss: Binary Crossentropy
- Regularization: Dropout + EarlyStopping
