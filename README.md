Spam SMS Detection Using LSTM - Project Report  (Model deployed link: https://huggingface.co/spaces/piyushmadhukar/SPAM_SMS_DETECTION)

![Uploading image.png…]()

1. IntroductioN :Spam messages are a common issue in digital communication, often causing inconvenience and security threats. This project aims to build an efficient SMS classification model that accurately distinguishes spam messages from legitimate ones using a Long Short-Term Memory (LSTM) neural network.

2. Dataset Description :The dataset used for this project consists of labeled SMS messages categorized as either "ham" (legitimate) or "spam" (unwanted). The dataset contains:

Columns: label (ham/spam) and message (text content)
Preprocessing: Text cleaning, tokenization, and sequence padding.

3. Preprocessing Steps : To prepare the data for training, the following steps were taken:

Text Cleaning: Lowercasing, removing special characters, and stopwords.
Tokenization: Converting words into numerical sequences.
Padding: Ensuring uniform input length by padding sequences.

4. Model Architecture:

The model is built using TensorFlow and consists of:
Embedding Layer: Converts words into dense vectors.
LSTM Layers: Two LSTM layers (64 and 32 units) to capture sequential dependencies.
Dropout Layer: Prevents overfitting by randomly dropping connections.
Dense Layer: Outputs a probability using the sigmoid activation function.

5. Model Training & Evaluation

Loss Function: Binary Cross-Entropy
Optimizer: Adam
Epochs: 5
Batch Size: 32

Evaluation Metrics:
Metric        Spam (1)  Ham (0)
Precision     0.96      0.99
Recall        0.93      0.99
F1-score      0.95      0.99
Accuracy      99%        -

6. Model Saving & Deployment

The trained LSTM model was saved as spam_classifier_lstm.h5.
The tokenizer was stored in tokenizer.pkl for later use in prediction.
A Streamlit interface was designed to allow users to input SMS texts and classify them as spam or ham in real-time.
