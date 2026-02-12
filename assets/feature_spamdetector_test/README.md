# 🛡️ Spam Slayer

**Core Functionality**
- Advanced ML Models: Uses multilingual E5 embeddings with FAISS similarity search
- Interactive Web Interface: Built with Streamlit for easy use
- Visual Analytics: Real-time performance charts and confidence visualizations
- Saliency Analysis: Highlights important words contributing to spam classification

**Spam Categories**
- 📢 Promotional/Advertisement: Marketing and sales messages
- ⚠️ System Alert/Phishing: Fake security alerts and phishing attempts
- 🔍 Other Spam Types: Various other spam categories
- ✅ Ham (Not Spam): Legitimate messages


## Quick start

1. Go to the project folder
```
cd path/to/public/feature_spamdetector_main
```

2. Create and activate a virtual environment
```
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies
```
pip install -r requirements.txt
```

4. Add optional hugging face token
```
echo "your_huggingface_token_here" > token.txt
```

5. Run the app
```
streamlit run app.py
```



### Model Pipeline

1. **Text Preprocessing**
   - Cleaning and normalization
   - Tokenization
   - Encoding preparation

2. **Embedding Generation**
   - Multilingual E5 model
   - 768-dimensional vectors
   - Batch processing optimization

3. **Similarity Search**
   - FAISS index construction
   - K-nearest neighbors
   - Distance-based classification

4. **Post-processing**
   - Confidence calculation
   - Category assignment
   - Result formatting


## 📝 Additional Resources

### Sample Data Format

Example training data structure:
```csv
text,label,category
"Khuyến mãi đặc biệt chỉ hôm nay!",spam,promotional
"Cuộc họp sẽ bắt đầu lúc 10h",ham,legitimate
"CẢNH BÁO: Tài khoản của bạn sẽ bị khóa",spam,phishing
```

### API Integration

For programmatic usage:
```python
from spam_model import SpamClassifier

# Initialize classifier
classifier = SpamClassifier()

# Load trained model
classifier.load_model()

# Make prediction
result = classifier.predict("Your text here")
print(f"Classification: {result['label']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Testing SPAM
#### Tiếng Việt (5 câu)
```
Tài khoản của bạn sẽ bị tạm khóa sau 24h, vui lòng đăng nhập ngay tại link này để xác minh.

Chúc mừng! Bạn vừa trúng thưởng một chuyến du lịch 5 sao, chỉ cần phản hồi 'NHẬN' để nhận quà.

Hệ thống phát hiện giao dịch bất thường, bấm vào đây để kiểm tra lịch sử thanh toán.

Giảm giá sốc 80% cho 50 khách hàng đầu tiên, đặt ngay kẻo hết.

Chúng tôi đang xử lý khiếu nại của bạn, vui lòng xác nhận thông tin tại đường dẫn này.
```

#### Tiếng Anh (5 câu)
```
Your PayPal account has been flagged. Verify now to avoid suspension.

Congratulations! You've been selected for an exclusive reward. Claim it here.

We noticed unusual login attempts on your account. Click here to secure it.

Limited-time offer: Get up to 90% off on premium items. Shop now.

Please confirm your shipping details to avoid delivery cancellation.

Hey John, btw I just found this app, u might get $500 cashback if u install
```

### Testing HAM
#### Tiếng Việt (5 câu Ham)
```
Tài khoản ngân hàng của bạn đã được cộng tiền lương tháng này, vui lòng kiểm tra.

Cửa hàng sẽ giao đơn hàng bạn đặt ngày hôm qua vào chiều nay.

Lịch họp tuần tới đã được cập nhật, mời bạn xem lại trên hệ thống.

Chúc mừng sinh nhật! Mong bạn có một ngày thật vui vẻ bên gia đình và bạn bè.

Hóa đơn tiền điện tháng này đã sẵn sàng, vui lòng thanh toán trước ngày 20.
```

#### Tiếng Anh (5 câu Ham)
```
Your package has been shipped and is expected to arrive tomorrow.

Reminder: Your dental appointment is scheduled for 3 PM on Friday.

Happy New Year! Wishing you health and happiness in the coming year.

Your subscription renewal was successful. Thank you for staying with us.

The meeting agenda has been updated. Please review before tomorrow.
```
