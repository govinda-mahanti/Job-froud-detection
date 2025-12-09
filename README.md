# Job Fraud Detection API

A machine learning-powered API for detecting potentially fraudulent job postings using natural language processing and company verification.

## 🚀 Features

- **Fraud Detection**: Analyzes job postings for potential fraud indicators
- **Company Verification**: Validates company information against known databases
- **Email Analysis**: Checks for phishing/suspicious email content
- **RESTful API**: Easy integration with web and mobile applications
- **Scalable**: Built with production deployment in mind

## 🛠 Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git
- Docker (optional, for containerized deployment)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/job-fraud-detection.git
   cd job-fraud-detection
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

1. Create a `.env` file in the root directory:
   ```env
   FLASK_APP=app.py
   FLASK_ENV=development
   SECRET_KEY=your-secret-key-here
   MODEL_PATH=models/
   HF_TOKEN=your-huggingface-token
   ```

2. Place your trained model files in the `models/` directory

## 🏃 Running the Application

### Development Mode
```bash
flask run
```

### Production Mode (Using Gunicorn)
```bash
gunicorn --bind 0.0.0.0:5000 app:app
```

### Using Docker
```bash
# Build the Docker image
docker build -t job-fraud-detection .

# Run the container
docker run -p 5000:5000 job-fraud-detection
```

## 📚 API Documentation

### Endpoints

#### `GET /`
Health check endpoint.

**Response:**
```json
{
  "status": "success",
  "message": "Job Fraud Detection API is running",
  "endpoints": {
    "GET /": "Health check (this endpoint)",
    "POST /predict": "Predict job fraud probability"
  }
}
```

#### `POST /predict`
Predict the probability of a job posting being fraudulent.

**Request Body:**
```json
{
  "job_title": "Senior Software Engineer",
  "job_description": "Job description here...",
  "company_name": "Tech Corp",
  "company_domain": "techcorp.com",
  "salary_raw": "$120,000 - $150,000",
  "location": "Remote",
  "email_subject": "Regarding your application",
  "email_body": "Email content here..."
}
```

**Response:**
```json
{
  "status": "success",
  "prediction": {
    "company_auth_score": 85.5,
    "job_fraud_probability": 0.12,
    "email_risk_score": 0.15,
    "final_verdict": "Legitimate",
    "confidence": 0.88
  }
}
```

## 🚀 Deployment

### Heroku
```bash
# Login to Heroku
heroku login

# Create a new Heroku app
heroku create your-app-name

# Deploy to Heroku
git push heroku main
```

### AWS Elastic Beanstalk
1. Install EB CLI: `pip install awsebcli`
2. Initialize EB: `eb init -p python-3.9 job-fraud-detection`
3. Create environment: `eb create job-fraud-detection-env`
4. Deploy: `eb deploy`

## 🧪 Testing

Run the test suite:
```bash
pytest
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/job-fraud-detection](https://github.com/yourusername/job-fraud-detection)

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Flask](https://flask.palletsprojects.com/)
- [Scikit-learn](https://scikit-learn.org/)

---

<div align="center">
  Made with ❤️ by Your Name
</div>
