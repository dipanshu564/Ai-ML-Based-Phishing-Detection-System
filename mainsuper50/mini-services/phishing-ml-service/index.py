#!/usr/bin/env python3
"""
Phishing Detection ML Service
Provides AI/ML-based phishing detection for URLs, emails, and content
"""

import asyncio
import json
import logging
import re
import ssl
from datetime import datetime
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import joblib
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Phishing Detection ML Service", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class URLDetectionRequest(BaseModel):
    url: str

class EmailDetectionRequest(BaseModel):
    email: str

class ContentDetectionRequest(BaseModel):
    content: str

class DetectionResponse(BaseModel):
    is_phishing: bool
    confidence: float
    analysis: str
    features: Dict[str, Any]
    timestamp: str

class PhishingMLModel:
    """ML Model for phishing detection"""
    
    def __init__(self):
        self.url_model = None
        self.email_model = None
        self.content_model = None
        self.url_vectorizer = None
        self.email_vectorizer = None
        self.content_vectorizer = None
        self.is_trained = False
        # Try to load persisted models from disk
        self.models_dir = Path(__file__).resolve().parent / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        try:
            if (self.models_dir / "url_model.joblib").exists():
                self.url_model = joblib.load(self.models_dir / "url_model.joblib")
            if (self.models_dir / "email_model.joblib").exists():
                self.email_model = joblib.load(self.models_dir / "email_model.joblib")
            if (self.models_dir / "content_model.joblib").exists():
                self.content_model = joblib.load(self.models_dir / "content_model.joblib")
            if (self.models_dir / "email_vectorizer.joblib").exists():
                self.email_vectorizer = joblib.load(self.models_dir / "email_vectorizer.joblib")
            if (self.models_dir / "content_vectorizer.joblib").exists():
                self.content_vectorizer = joblib.load(self.models_dir / "content_vectorizer.joblib")
            # If we have at least one model, mark as trained
            if self.url_model or self.email_model or self.content_model:
                self.is_trained = True
                logger.info("Loaded persisted ML models from disk")
        except Exception as e:
            logger.warning(f"Could not load persisted models: {e}")
        
    def train_url_model(self):
        """Train URL phishing detection model. Tries to load CSV from DATASET_PATH if provided."""
        dataset_path = os.getenv("DATASET_PATH")
        features = []
        labels = []

        if dataset_path and Path(dataset_path).exists():
            try:
                df = pd.read_csv(dataset_path)
                # Detect URL column candidates
                if 'url' in df.columns:
                    for _, row in df.iterrows():
                        url = str(row['url'])
                        features.append(self._extract_url_features(url))
                        lab = row.get('label') or row.get('is_phishing') or row.get('target')
                        labels.append(1 if str(lab).strip().lower() in ['1','true','phishing','yes'] else 0)
                else:
                    logger.info("No 'url' column found in dataset for URL model - falling back to synthetic data")
            except Exception as e:
                logger.warning(f"Failed to load dataset for URL model: {e}")

        # Fallback synthetic dataset if no real data
        if not features:
            urls = [
                "https://paypal.com/login", "https://google.com", "https://facebook.com",
                "http://paypal.com.security-update.com", "http://google.update-account.com",
                "https://amazon.com", "https://microsoft.com", "http://amazon.security-check.com",
                "https://github.com", "https://stackoverflow.com", "http://github.verify-account.com"
            ]
            labels = [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1]
            for url in urls:
                features.append(self._extract_url_features(url))

        X = np.array(features)
        y = np.array(labels)

        self.url_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.url_model.fit(X, y)
        # persist
        try:
            joblib.dump(self.url_model, self.models_dir / "url_model.joblib")
            logger.info("Saved url_model.joblib")
        except Exception as e:
            logger.warning(f"Failed to persist url_model: {e}")

        logger.info("URL phishing model trained successfully")
        
    def train_email_model(self):
        """Train email phishing detection model; loads DATASET_PATH if available."""
        dataset_path = os.getenv("DATASET_PATH")
        emails = []
        labels = []

        if dataset_path and Path(dataset_path).exists():
            try:
                df = pd.read_csv(dataset_path)
                if 'email' in df.columns:
                    for _, row in df.iterrows():
                        emails.append(str(row['email']))
                        lab = row.get('label') or row.get('is_phishing') or row.get('target')
                        labels.append(1 if str(lab).strip().lower() in ['1','true','phishing','yes'] else 0)
                else:
                    logger.info("No 'email' column found in dataset for Email model - falling back to synthetic data")
            except Exception as e:
                logger.warning(f"Failed to load dataset for Email model: {e}")

        if not emails:
            emails = [
                "Your account has been compromised, click here to reset",
                "Meeting scheduled for tomorrow at 2 PM",
                "Urgent: Verify your account immediately",
                "Project update and next steps",
                "Winner! You've won $1,000,000 claim now",
                "Weekly newsletter and updates"
            ]
            labels = [1, 0, 1, 0, 1, 0]

        self.email_vectorizer = TfidfVectorizer(max_features=1000)
        X = self.email_vectorizer.fit_transform(emails)
        y = np.array(labels)

        self.email_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.email_model.fit(X, y)
        try:
            joblib.dump(self.email_model, self.models_dir / "email_model.joblib")
            joblib.dump(self.email_vectorizer, self.models_dir / "email_vectorizer.joblib")
            logger.info("Saved email_model.joblib and email_vectorizer.joblib")
        except Exception as e:
            logger.warning(f"Failed to persist email model/vectorizer: {e}")

        logger.info("Email phishing model trained successfully")
        
    def train_content_model(self):
        """Train content phishing detection model; loads DATASET_PATH if available."""
        dataset_path = os.getenv("DATASET_PATH")
        contents = []
        labels = []

        if dataset_path and Path(dataset_path).exists():
            try:
                df = pd.read_csv(dataset_path)
                if 'content' in df.columns:
                    for _, row in df.iterrows():
                        contents.append(str(row['content']))
                        lab = row.get('label') or row.get('is_phishing') or row.get('target')
                        labels.append(1 if str(lab).strip().lower() in ['1','true','phishing','yes'] else 0)
                else:
                    logger.info("No 'content' column found in dataset for Content model - falling back to synthetic data")
            except Exception as e:
                logger.warning(f"Failed to load dataset for Content model: {e}")

        if not contents:
            contents = [
                "Click here to claim your prize now!",
                "Please review the attached document",
                "Urgent action required: account suspended",
                "Team meeting notes and action items",
                "Verify your identity within 24 hours",
                "Monthly performance report"
            ]
            labels = [1, 0, 1, 0, 1, 0]

        self.content_vectorizer = TfidfVectorizer(max_features=1000)
        X = self.content_vectorizer.fit_transform(contents)
        y = np.array(labels)

        self.content_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.content_model.fit(X, y)
        try:
            joblib.dump(self.content_model, self.models_dir / "content_model.joblib")
            joblib.dump(self.content_vectorizer, self.models_dir / "content_vectorizer.joblib")
            logger.info("Saved content_model.joblib and content_vectorizer.joblib")
        except Exception as e:
            logger.warning(f"Failed to persist content model/vectorizer: {e}")

        logger.info("Content phishing model trained successfully")
        
    def _extract_url_features(self, url: str) -> List[float]:
        """Extract features from URL for ML model"""
        try:
            parsed = urlparse(url)
            features = []
            
            # Length of URL
            features.append(len(url))
            
            # Length of domain
            features.append(len(parsed.netloc))
            
            # Number of dots in domain
            features.append(parsed.netloc.count('.'))
            
            # Has IP address
            features.append(1 if re.match(r'\d+\.\d+\.\d+\.\d+', parsed.netloc) else 0)
            
            # Has suspicious keywords
            suspicious_keywords = ['login', 'secure', 'account', 'update', 'verify', 'password']
            features.append(sum(1 for keyword in suspicious_keywords if keyword in url.lower()))
            
            # HTTPS
            features.append(1 if parsed.scheme == 'https' else 0)
            
            # URL length
            features.append(len(url))
            
            # Domain length
            features.append(len(parsed.netloc))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting URL features: {e}")
            return [0] * 8
            
    def predict_url(self, url: str) -> Dict[str, Any]:
        """Predict if URL is phishing"""
        if not self.url_model:
            self.train_url_model()
            
        features = self._extract_url_features(url)
        prediction = self.url_model.predict([features])[0]
        probability = self.url_model.predict_proba([features])[0]
        
        confidence = max(probability)
        is_phishing = prediction == 1
        
        analysis = self._generate_url_analysis(url, is_phishing, confidence)
        
        return {
            'is_phishing': bool(is_phishing),
            'confidence': float(confidence),
            'analysis': analysis,
            'features': {
                'url_length': len(url),
                'domain_length': len(urlparse(url).netloc),
                'has_https': urlparse(url).scheme == 'https',
                'suspicious_keywords': sum(1 for keyword in ['login', 'secure', 'account', 'update', 'verify', 'password'] if keyword in url.lower())
            }
        }
        
    def predict_email(self, email: str) -> Dict[str, Any]:
        """Predict if email is phishing"""
        if not self.email_model:
            self.train_email_model()
            
        # Vectorize email
        email_vector = self.email_vectorizer.transform([email])
        prediction = self.email_model.predict(email_vector)[0]
        probability = self.email_model.predict_proba(email_vector)[0]
        
        confidence = max(probability)
        is_phishing = prediction == 1
        
        analysis = self._generate_email_analysis(email, is_phishing, confidence)
        
        return {
            'is_phishing': bool(is_phishing),
            'confidence': float(confidence),
            'analysis': analysis,
            'features': {
                'email_length': len(email),
                'suspicious_keywords': sum(1 for keyword in ['urgent', 'verify', 'account', 'suspended', 'winner'] if keyword in email.lower()),
                'has_links': 'http' in email.lower()
            }
        }
        
    def predict_content(self, content: str) -> Dict[str, Any]:
        """Predict if content is phishing"""
        if not self.content_model:
            self.train_content_model()
            
        # Vectorize content
        content_vector = self.content_vectorizer.transform([content])
        prediction = self.content_model.predict(content_vector)[0]
        probability = self.content_model.predict_proba(content_vector)[0]
        
        confidence = max(probability)
        is_phishing = prediction == 1
        
        analysis = self._generate_content_analysis(content, is_phishing, confidence)
        
        return {
            'is_phishing': bool(is_phishing),
            'confidence': float(confidence),
            'analysis': analysis,
            'features': {
                'content_length': len(content),
                'suspicious_keywords': sum(1 for keyword in ['urgent', 'click', 'verify', 'account', 'prize'] if keyword in content.lower()),
                'urgency_indicators': sum(1 for keyword in ['urgent', 'immediately', 'now', 'asap'] if keyword in content.lower())
            }
        }
        
    def _generate_url_analysis(self, url: str, is_phishing: bool, confidence: float) -> str:
        """Generate analysis text for URL detection"""
        parsed = urlparse(url)
        analysis_parts = []
        
        if is_phishing:
            analysis_parts.append("⚠️ Potential phishing URL detected")
            if len(parsed.netloc) > 30:
                analysis_parts.append("- Unusually long domain name")
            if parsed.netloc.count('.') > 3:
                analysis_parts.append("- Excessive subdomains")
            if not parsed.scheme == 'https':
                analysis_parts.append("- Not using HTTPS")
            if any(keyword in url.lower() for keyword in ['login', 'secure', 'account']):
                analysis_parts.append("- Contains suspicious keywords")
        else:
            analysis_parts.append("✅ URL appears legitimate")
            if parsed.scheme == 'https':
                analysis_parts.append("- Uses secure HTTPS connection")
            if len(parsed.netloc) < 20:
                analysis_parts.append("- Reasonable domain length")
                
        return "\n".join(analysis_parts)
        
    def _generate_email_analysis(self, email: str, is_phishing: bool, confidence: float) -> str:
        """Generate analysis text for email detection"""
        analysis_parts = []
        
        if is_phishing:
            analysis_parts.append("⚠️ Potential phishing email detected")
            if any(keyword in email.lower() for keyword in ['urgent', 'immediately']):
                analysis_parts.append("- Creates false sense of urgency")
            if any(keyword in email.lower() for keyword in ['verify', 'account', 'suspended']):
                analysis_parts.append("- Contains account-related threats")
            if 'http' in email.lower():
                analysis_parts.append("- Contains links that may be malicious")
        else:
            analysis_parts.append("✅ Email appears legitimate")
            if len(email) < 200:
                analysis_parts.append("- Reasonable email length")
            if not any(keyword in email.lower() for keyword in ['urgent', 'verify', 'account']):
                analysis_parts.append("- No suspicious keywords detected")
                
        return "\n".join(analysis_parts)
        
    def _generate_content_analysis(self, content: str, is_phishing: bool, confidence: float) -> str:
        """Generate analysis text for content detection"""
        analysis_parts = []
        
        if is_phishing:
            analysis_parts.append("⚠️ Potential phishing content detected")
            if any(keyword in content.lower() for keyword in ['click', 'here', 'now']):
                analysis_parts.append("- Contains call-to-action phrases")
            if any(keyword in content.lower() for keyword in ['prize', 'winner', 'claim']):
                analysis_parts.append("- Contains prize/claim language")
            if any(keyword in content.lower() for keyword in ['urgent', 'immediately']):
                analysis_parts.append("- Creates false sense of urgency")
        else:
            analysis_parts.append("✅ Content appears legitimate")
            if len(content) < 500:
                analysis_parts.append("- Reasonable content length")
            if not any(keyword in content.lower() for keyword in ['click', 'prize', 'urgent']):
                analysis_parts.append("- No suspicious patterns detected")
                
        return "\n".join(analysis_parts)

# Initialize ML model
ml_model = PhishingMLModel()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Phishing Detection ML Service is running", "status": "healthy"}

@app.post("/detect/url", response_model=DetectionResponse)
async def detect_url_phishing(request: URLDetectionRequest):
    """Detect phishing in URL"""
    try:
        result = ml_model.predict_url(request.url)
        return DetectionResponse(
            is_phishing=result['is_phishing'],
            confidence=result['confidence'],
            analysis=result['analysis'],
            features=result['features'],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error in URL detection: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/detect/email", response_model=DetectionResponse)
async def detect_email_phishing(request: EmailDetectionRequest):
    """Detect phishing in email"""
    try:
        result = ml_model.predict_email(request.email)
        return DetectionResponse(
            is_phishing=result['is_phishing'],
            confidence=result['confidence'],
            analysis=result['analysis'],
            features=result['features'],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error in email detection: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/detect/content", response_model=DetectionResponse)
async def detect_content_phishing(request: ContentDetectionRequest):
    """Detect phishing in content"""
    try:
        result = ml_model.predict_content(request.content)
        return DetectionResponse(
            is_phishing=result['is_phishing'],
            confidence=result['confidence'],
            analysis=result['analysis'],
            features=result['features'],
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        logger.error(f"Error in content detection: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    return {
        "url_model_trained": ml_model.url_model is not None,
        "email_model_trained": ml_model.email_model is not None,
        "content_model_trained": ml_model.content_model is not None,
        "model_type": "RandomForestClassifier",
        "features": {
            "url": ["url_length", "domain_length", "dot_count", "has_ip", "suspicious_keywords", "https", "url_length", "domain_length"],
            "email": ["tfidf_features", "email_length", "suspicious_keywords", "has_links"],
            "content": ["tfidf_features", "content_length", "suspicious_keywords", "urgency_indicators"]
        }
    }

if __name__ == "__main__":
    # Train models on startup if not loaded from disk
    logger.info("Starting ML service...")
    dataset_path = os.getenv("DATASET_PATH")
    if ml_model.is_trained:
        logger.info("Models already loaded from disk; skipping training")
    else:
        logger.info("Training ML models...")
        # If a dataset path provided, report it
        if dataset_path:
            logger.info(f"Using dataset path: {dataset_path}")
        ml_model.train_url_model()
        ml_model.train_email_model()
        ml_model.train_content_model()
        logger.info("ML models trained successfully")

    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8001)