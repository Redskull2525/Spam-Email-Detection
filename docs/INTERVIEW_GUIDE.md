# Interview Guide: Spam Email Detection (Bidirectional LSTM)

All numbers here come from the run documented in the README (600-email balanced test set).

## 30-second pitch

"I built a spam email classifier with TensorFlow and deployed it as a Streamlit web app. It cleans the email text, turns the words into integer sequences with a tokenizer, and a Bidirectional LSTM predicts the probability that the email is spam. On a balanced held-out test set of 600 emails, it gets about 97.8% accuracy and 99.7% spam recall. On ham emails it never saw, about 3.4% get flagged as spam, so false positives are its main weakness."

## 60-second pitch

"The dataset is 5,171 Enron emails, 71% ham and 29% spam. I balanced it by downsampling ham to 1,499 emails, then split it 80/20 with stratification. Preprocessing removes punctuation and stopwords and lowercases the text. A Keras Tokenizer keeps the top 10,000 words and maps everything else to an OOV token. Each email is padded or truncated to 100 tokens.

The model is an Embedding layer, a Bidirectional LSTM with 64 units, a Dense ReLU layer with dropout, and a sigmoid output. I train with Adam and binary cross-entropy, using early stopping on a validation split taken from the training data.

Test results: accuracy 0.978, spam precision 0.961, spam recall 0.997. The errors are mostly false positives: 12 of 300 ham emails flagged as spam. I also tested the 2,173 ham emails left out by downsampling, and 3.4% of those are flagged at the app's 0.65 threshold. The Streamlit app loads the saved model and tokenizer and applies exactly the same cleaning as training. The biggest limitation is that the data is 20-year-old Enron mail, so the model doesn't generalise well to modern email."

## Pipeline, step by step

1. **Load data:** `data/spam_ham_dataset.csv` (columns `label`, `text`, `label_num`).
2. **Balance:** randomly sample 1,499 ham emails to match the 1,499 spam emails.
3. **Clean:** remove the word "Subject", strip punctuation, lowercase, and drop NLTK English stopwords.
4. **Split:** 80% train / 20% test, stratified, `random_state=42`. 10% of the training split is used for validation.
5. **Tokenize:** `Tokenizer(num_words=10000, oov_token="<OOV>")` is fitted on the training text only. Each word becomes an integer id.
6. **Pad:** `pad_sequences(maxlen=100, padding="post")` gives every email the same length.
7. **Model:** Embedding(64) -> Bidirectional(LSTM(64)) -> Dense(64, ReLU) -> Dropout(0.5) -> Dense(1, sigmoid).
8. **Train:** Adam, binary cross-entropy, up to 15 epochs, EarlyStopping(patience=3) and ReduceLROnPlateau.
9. **Evaluate:** accuracy, precision, recall, F1 and the confusion matrix on the test set, plus the false-positive rate on unused ham.
10. **Save and serve:** `model/spam_model.keras`, `model/tokenizer.pkl` and `model/stopwords.txt`. `app.py` loads them and uses threshold 0.65.

## What the metrics mean

- **Accuracy (0.978):** the fraction of all test emails classified correctly. On imbalanced data this can mislead: if 90% of mail is ham, predicting "ham" for everything already scores 90%.
- **Precision for spam (0.961):** of the emails flagged as spam, 96.1% really were spam. Low precision means legitimate mail ends up in the spam folder.
- **Recall for spam (0.997):** of the real spam emails, 99.7% were caught. Low recall means spam reaches the inbox.
- **F1 (0.979):** the harmonic mean of precision and recall. It is only high when both are high.
- **Confusion matrix:** [[288, 12], [1, 299]]. That is 12 false positives (ham flagged as spam) and 1 false negative (spam missed).
- **Threshold:** the model outputs a probability, and the app calls an email spam if P >= 0.65. Raising the threshold gives fewer false positives but more missed spam.

## 10 interview questions (with follow-ups)

1. **Why an LSTM and not a simple model like Naive Bayes or logistic regression on TF-IDF?**
   An LSTM can use word order and context. Honest answer: on this dataset, TF-IDF + logistic regression would probably score about as well, and it is much smaller and faster.
   *Follow-up: How would you check that?* Train the baseline on the same split and compare precision, recall and F1.

2. **Why Bidirectional?**
   It reads the sequence forwards and backwards, so each position sees context from both sides.
   *Follow-up: What does it cost?* About twice the LSTM parameters and compute.

3. **What does the Embedding layer do?**
   It maps each word id to a learned 64-dimensional vector, so words used in similar ways end up with similar vectors.
   *Follow-up: Could you use pretrained embeddings?* Yes, GloVe or word2vec, or a transformer such as DistilBERT.

4. **Why pad to 100 tokens?**
   Batches need equal-length inputs. 100 tokens covers most short emails.
   *Follow-up: What happens to long emails?* With the default `truncating="pre"`, only the last 100 tokens are kept, so the start of the email (often the subject) is lost.

5. **How did you handle class imbalance?**
   I downsampled ham to match spam.
   *Follow-up: What are the downsides?* It throws away 2,173 ham emails and shifts the model towards predicting spam. Alternatives are class weights, oversampling, or keeping all the data and tuning the threshold.

6. **Is 97.8% accuracy good?**
   Only in context. The test set is balanced, so accuracy is meaningful there, but precision, recall and the false-positive rate on real-distribution ham (3.4%) matter more.
   *Follow-up: Which error is worse for email?* Usually a false positive, because an important email gets lost.

7. **How do you prevent overfitting?**
   Dropout, early stopping on a validation split, and LR reduction on plateau.
   *Follow-up: Why not validate on the test set?* Because early stopping would then tune on the test data, and the test score would be optimistic. The original notebook did this, and I fixed it.

8. **What is OOV and why does it matter?**
   OOV ("out of vocabulary") is the token used for words the tokenizer never saw, or words outside the top 10,000. Without it, unknown words would silently disappear.
   *Follow-up: How do spammers exploit this?* With misspellings such as "fr33". Character-level or subword tokenization helps.

9. **How does the app make sure it preprocesses like training?**
   It loads the same fitted tokenizer and the saved stopword list, and uses the same cleaning function.
   *Follow-up: What is training/serving skew?* When the inference pipeline differs from the training pipeline, so the model sees inputs it wasn't trained on.

10. **What would you do to make this production-ready?**
    Train on recent, diverse email data, add a TF-IDF baseline, use calibrated thresholds, monitor drift, and use email metadata (sender, links, headers).
    *Follow-up: How would you retrain?* Collect user "not spam"/"spam" feedback, retrain on a schedule, and A/B test against the current model.

## 4 hands-on exercises

1. **Baseline comparison:** train TF-IDF + logistic regression on the same split and compare precision, recall and F1 with the LSTM.
2. **Threshold sweep:** plot precision and recall for thresholds from 0.1 to 0.9. Pick a threshold that keeps the unseen-ham false-positive rate under 1%, and report the recall you get at that threshold.
3. **No downsampling:** train on all 5,171 emails with `class_weight`, evaluate on a stratified test set that keeps the natural imbalance, and compare.
4. **Robustness test:** write 10 modern ham emails (shipping notices, meeting invites) and 10 modern spam emails, run them through the app, and record the failures. Explain them in terms of the Enron training data.
