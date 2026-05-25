Phishing Email Detection Model

Machine learning classifier using Scikit-learn (TF-IDF + URL/keyword features) to classify emails as Phishing or Safe.

```bash
cd 3-phishing-detection
python train_model.py          # Train model, show accuracy & confusion matrix
python predict.py --text "URGENT verify your account at http://fake-bank.com"
python predict.py --file sample_email.txt
```

**Features:**
- 500+ email training dataset included
- TF-IDF text features + URL count, keyword score, special char ratio
- Accuracy, F1 score, classification report
- Confusion matrix saved as `models/confusion_matrix.png`