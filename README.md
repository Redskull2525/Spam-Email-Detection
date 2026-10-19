# Spam Email Detection using Deep Learning (Bidirectional LSTM)

An end-to-end spam classifier: text cleaning, a Keras tokenizer, a Bidirectional LSTM model, and a Streamlit web app for real-time predictions.

**Live app:** https://spam-email-detection-abhi.streamlit.app/

---

## Results (actual run, held-out test set)

Numbers below come from the latest end-to-end run of `notebooks/training.ipynb` (TensorFlow-CPU 2.21, Python 3.13). Neural network training is not perfectly deterministic, so a re-run can differ slightly.

**Dataset:** `data/spam_ham_dataset.csv` contains 5,171 emails: 3,672 ham (71%) and 1,499 spam (29%). The notebook **downsamples ham to 1,499**, which gives a balanced set of 2,998 emails. That set is split 80/20 (stratified), so the test set has **600 emails: 300 spam and 300 ham**. 10% of the training split is held out for validation (early stopping and LR scheduling), so the test set is never used during training.

| Metric (test set, threshold 0.5) | Value |
|---|---|
| Accuracy | 0.9783 |
| Spam precision | 0.9614 |
| Spam recall | 0.9967 |
| Spam F1 | 0.9787 |
| Test loss | 0.0842 |

Confusion matrix (rows = true, columns = predicted):

|            | pred ham | pred spam |
|------------|---------:|----------:|
| **true ham**  | 288 | 12 |
| **true spam** | 1   | 299 |

At the app's threshold (**0.65**): accuracy 0.9767, spam precision 0.9613, spam recall 0.9933 (288/12 and 2/298).

**Extra check on realistic ham:** the 2,173 ham emails dropped by downsampling were never seen by the model. At threshold 0.65, it flags **74 of them (3.41%) as spam** (88, or 4.05%, at 0.5). So on real mail traffic, most errors would be false positives (legitimate mail marked as spam).

> An earlier version of the notebook reported 0.9833 accuracy. That run used the test set as validation data for early stopping, which makes the score slightly optimistic. The numbers above come from the corrected setup.

---

## How the model works

```
Email text
  -> cleaning: remove the word "Subject", strip punctuation, lowercase, drop English stopwords (NLTK list)
  -> Keras Tokenizer (top 10,000 words, <OOV> token for unknown words) -> list of integer ids
  -> pad_sequences to length 100 (shorter: zero-padded at the end; longer: only the last 100 tokens are kept)
  -> Embedding (64-dim word vectors, learned)
  -> Bidirectional LSTM (64 units each direction) reads the sequence both ways
  -> Dense(64, ReLU) -> Dropout(0.5) -> Dense(1, sigmoid) = P(spam)
  -> app.py: P(spam) >= 0.65 -> "Spam", otherwise "Not Spam"
```

- **Loss:** binary cross-entropy. **Optimizer:** Adam.
- **Training:** up to 15 epochs, batch size 32, EarlyStopping (patience 3, restores the best weights) and ReduceLROnPlateau.
- `app.py` applies the **same cleaning as training** (the stopword list is saved to `model/stopwords.txt`, so the app does not need to download NLTK data).

---

## Limitations

- **Old, single-domain data.** The emails are from the Enron corpus (energy-trading company, around 2000-2002). "Ham" is mostly Enron business mail, so the model learned what *Enron* mail looks like. In a quick manual check, modern legitimate messages such as a shipping notice ("Your Amazon order has shipped...", P=0.62) or a work message mentioning Jira (P=0.69) scored near or above the 0.65 threshold. Treat this as a learning project, not a production filter.
- **Class imbalance.** The real data is 71% ham. Training on a balanced subset makes the model more willing to predict spam, which shows up as the ~3-4% false-positive rate on unseen ham.
- **Small test set.** 600 emails, so each error moves accuracy by about 0.17 percentage points.
- **Truncation.** Only 100 tokens (after stopword removal) are used. With Keras's default `truncating="pre"`, a long email keeps only its last 100 tokens.
- **No adversarial robustness.** Spammers can deliberately misspell words or hide text, and this model has no defence against it.

---

## Project structure

```
Spam-Email-Detection/
├── app.py                    # Streamlit application
├── data/spam_ham_dataset.csv # Dataset (5,171 Enron emails)
├── notebooks/training.ipynb  # Training notebook (writes model/ files)
├── model/
│   ├── spam_model.keras      # Trained model (~30 MB)
│   ├── tokenizer.pkl         # Fitted Keras tokenizer
│   └── stopwords.txt         # Stopword list used for cleaning
├── docs/INTERVIEW_GUIDE.md   # How to explain this project
├── requirements.txt          # App runtime dependencies (pinned)
└── requirements-train.txt    # Extra packages for the notebook
```

---

## Run locally

Use Python 3.12 or 3.13 (tensorflow-cpu 2.21 has no wheels for 3.14).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Retrain the model

```bash
pip install -r requirements-train.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/training.ipynb   # or open it and Run All
```

Run it **from the repository root**. The notebook reads `data/spam_ham_dataset.csv` and writes everything in `model/`.

> On Windows, if `pip install` fails with a "No such file or directory ... tensorflow\include\..." error, the venv path is too long. Enable Windows long paths or create the venv in a short path such as `C:\venvs\spam`.

### Deploying on Streamlit Community Cloud

Main file: `app.py`. In **Advanced settings**, select Python **3.12 or 3.13**. Streamlit Cloud does not read `runtime.txt`, and changing the Python version requires deleting and redeploying the app.

---

## Author

**Abhishek Shelke**

- GitHub: [https://github.com/Redskull2525](https://github.com/Redskull2525)
- LinkedIn: [https://www.linkedin.com/in/abhishek-s-b98895249](https://www.linkedin.com/in/abhishek-s-b98895249)
