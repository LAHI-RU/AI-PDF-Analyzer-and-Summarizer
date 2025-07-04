# app.py - Fixed version with better error handling and fallback options
import os
import sqlite3
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pdfplumber
from openai import OpenAI
from dotenv import load_dotenv
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import traceback

# Load environment variables from .env file
load_dotenv()

# Create Flask application
app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'pdf'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize OpenAI client with API key from environment variable
openai_api_key = os.getenv('OPENAI_API_KEY')
client = None

if openai_api_key and openai_api_key != 'your-openai-api-key-here':
    try:
        client = OpenAI(api_key=openai_api_key)
        print("✅ OpenAI client initialized successfully")
    except Exception as e:
        print(f"⚠️ Warning: OpenAI client initialization failed: {e}")
        client = None
else:
    print("⚠️ Warning: No valid OpenAI API key found. AI features will use fallback methods.")

class PDFAnalyzer:
    def __init__(self):
        self.init_database()
        self.ai_available = client is not None
    
    def init_database(self):
        """Initialize SQLite database to store document information"""
        conn = sqlite3.connect('pdf_analyzer.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                content TEXT NOT NULL,
                summary TEXT,
                key_points TEXT,
                word_count INTEGER,
                page_count INTEGER
            )
        ''')
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")
    
    def allowed_file(self, filename):
        """Check if the uploaded file has an allowed extension"""
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text content from PDF file"""
        text = ""
        page_count = 0
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
        
        return text.strip(), page_count
    
    def clean_text(self, text):
        """Clean and normalize extracted text"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    
    def generate_fallback_summary(self, text, summary_type="comprehensive"):
        """Generate a basic summary without AI when OpenAI is not available"""
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if summary_type == "executive":
            # Take first few and last few sentences
            summary_sentences = sentences[:3] + sentences[-2:]
            summary = ". ".join(summary_sentences) + "."
            return f"Executive Summary (Fallback Mode):\n\n{summary}\n\nNote: This is a basic text extraction. For AI-powered analysis, please configure your OpenAI API key."
        
        elif summary_type == "bullet":
            # Create bullet points from first sentence of each paragraph
            paragraphs = text.split('\n\n')
            bullets = []
            for para in paragraphs[:8]:  # Limit to 8 points
                first_sentence = para.split('.')[0].strip()
                if len(first_sentence) > 20:
                    bullets.append(f"• {first_sentence}")
            
            return f"Key Points (Fallback Mode):\n\n" + "\n".join(bullets) + "\n\nNote: This is a basic text extraction. For AI-powered analysis, please configure your OpenAI API key."
        
        else:  # comprehensive
            # Take a portion of the text
            word_count = len(text.split())
            if word_count > 500:
                words = text.split()[:500]
                summary = " ".join(words) + "..."
            else:
                summary = text
            
            return f"Document Content (Fallback Mode):\n\n{summary}\n\nWord Count: {word_count}\n\nNote: This is a basic text extraction. For AI-powered analysis, please configure your OpenAI API key."
    
    def generate_summary(self, text, summary_type="comprehensive"):
        """Generate AI-powered summary using OpenAI API with fallback"""
        if not self.ai_available:
            return self.generate_fallback_summary(text, summary_type)
        
        try:
            # Truncate text if too long (to stay within API limits)
            max_chars = 12000  # Roughly 3000 tokens
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            
            # Create different prompts based on summary type
            if summary_type == "executive":
                prompt = f"""
                Create a concise executive summary of the following document. Focus on:
                - Main objectives and key findings
                - Critical decisions or recommendations
                - Business impact and outcomes
                - Action items if any
                
                Document:
                {text}
                
                Executive Summary:
                """
            elif summary_type == "bullet":
                prompt = f"""
                Create a bullet-point summary of the following document:
                - Extract the most important points
                - Use clear, concise bullet points
                - Organize by themes or topics
                - Limit to 10-15 key points
                
                Document:
                {text}
                
                Key Points:
                """
            else:  # comprehensive
                prompt = f"""
                Create a comprehensive summary of the following document:
                - Capture the main themes and arguments
                - Include important details and context
                - Maintain logical flow and structure
                - Aim for 2-3 paragraphs
                
                Document:
                {text}
                
                Summary:
                """
            
            # Call OpenAI API using new interface
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert document analyzer. Provide clear, accurate summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            error_msg = str(e)
            print(f"OpenAI API Error: {error_msg}")
            
            # Check for specific error types
            if "insufficient_quota" in error_msg or "429" in error_msg:
                fallback_summary = self.generate_fallback_summary(text, summary_type)
                return f"⚠️ OpenAI API Quota Exceeded\n\n{fallback_summary}\n\nTo fix this:\n1. Check your OpenAI billing at https://platform.openai.com/account/billing\n2. Add payment method if needed\n3. Verify your API key has sufficient credits"
            elif "401" in error_msg or "invalid" in error_msg.lower():
                fallback_summary = self.generate_fallback_summary(text, summary_type)
                return f"⚠️ Invalid OpenAI API Key\n\n{fallback_summary}\n\nTo fix this:\n1. Get a valid API key from https://platform.openai.com/api-keys\n2. Update your .env file with the correct key"
            else:
                fallback_summary = self.generate_fallback_summary(text, summary_type)
                return f"⚠️ AI Service Temporarily Unavailable\n\n{fallback_summary}\n\nError: {error_msg}"
    
    def extract_key_information(self, text):
        """Extract key information from document using AI or fallback method"""
        if not self.ai_available:
            return self.extract_key_info_fallback(text)
        
        try:
            prompt = f"""
            Analyze the following document and extract key information:
            - Important dates and deadlines
            - Names of people, organizations, or companies
            - Key numbers, percentages, or financial figures
            - Important terms or concepts
            - Action items or next steps
            
            Format the response as clear, organized text:
            
            Document:
            {text[:8000]}
            
            Key Information:
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Extract key information and present it clearly."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=600,
                temperature=0.2
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error extracting key information: {e}")
            return self.extract_key_info_fallback(text)
    
    def extract_key_info_fallback(self, text):
        """Fallback method to extract key information without AI"""
        info = []
        
        # Extract dates (basic pattern matching)
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
        dates = re.findall(date_pattern, text)
        if dates:
            info.append(f"Dates Found: {', '.join(set(dates[:5]))}")
        
        # Extract numbers and percentages
        number_pattern = r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*%?\b'
        numbers = re.findall(number_pattern, text)
        if numbers:
            significant_numbers = [n for n in set(numbers) if len(n) > 2][:10]
            info.append(f"Numbers/Figures: {', '.join(significant_numbers)}")
        
        # Extract capitalized words (potential names/organizations)
        caps_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        caps_words = re.findall(caps_pattern, text)
        if caps_words:
            unique_caps = list(set(caps_words))[:10]
            info.append(f"Important Terms: {', '.join(unique_caps)}")
        
        if info:
            return "Key Information (Basic Extraction):\n\n" + "\n\n".join(info) + "\n\nNote: For advanced AI analysis, please configure your OpenAI API key."
        else:
            return "No specific key information patterns detected. For advanced AI analysis, please configure your OpenAI API key."
    
    def answer_question(self, text, question):
        """Answer questions about the document using AI or fallback method"""
        if not self.ai_available:
            return self.answer_question_fallback(text, question)
        
        try:
            prompt = f"""
            Based on the following document, please answer this question accurately and concisely:
            
            Question: {question}
            
            Document:
            {text[:10000]}
            
            Answer:
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Answer questions based only on the provided document content. If the answer isn't in the document, say so clearly."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"Error answering question: {e}")
            return self.answer_question_fallback(text, question)
    
    def answer_question_fallback(self, text, question):
        """Fallback method to answer questions without AI"""
        question_lower = question.lower()
        text_lower = text.lower()
        
        # Simple keyword matching
        if any(word in text_lower for word in question_lower.split()):
            # Find sentences containing question keywords
            sentences = text.split('.')
            relevant_sentences = []
            
            for sentence in sentences:
                if any(word in sentence.lower() for word in question_lower.split() if len(word) > 3):
                    relevant_sentences.append(sentence.strip())
                    if len(relevant_sentences) >= 3:
                        break
            
            if relevant_sentences:
                return f"Relevant information found:\n\n" + "\n\n".join(relevant_sentences) + "\n\nNote: This is basic text matching. For intelligent AI-powered answers, please configure your OpenAI API key."
            else:
                return "No directly relevant information found for your question. For intelligent AI-powered answers, please configure your OpenAI API key."
        else:
            return "No matching content found. For intelligent AI-powered answers, please configure your OpenAI API key."
    
    def compare_documents(self, doc1_text, doc2_text, doc1_name, doc2_name):
        """Compare two documents for similarities and differences"""
        try:
            # Calculate similarity using TF-IDF
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform([doc1_text, doc2_text])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            if not self.ai_available:
                return {
                    "similarity_score": round(similarity * 100, 2),
                    "ai_analysis": f"Similarity Analysis:\n\nDocuments '{doc1_name}' and '{doc2_name}' have a {round(similarity * 100, 2)}% similarity score based on text analysis.\n\nFor detailed AI-powered comparison analysis, please configure your OpenAI API key.\n\nBasic comparison:\n- Document 1: {len(doc1_text.split())} words\n- Document 2: {len(doc2_text.split())} words\n- Common vocabulary overlap: {round(similarity * 100, 2)}%"
                }
            
            # Generate detailed comparison using AI
            prompt = f"""
            Compare these two documents and provide:
            1. Main similarities between the documents
            2. Key differences
            3. Unique aspects of each document
            4. Overall assessment
            
            Document 1 ({doc1_name}):
            {doc1_text[:6000]}
            
            Document 2 ({doc2_name}):
            {doc2_text[:6000]}
            
            Comparison Analysis:
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Provide detailed document comparison analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            ai_analysis = response.choices[0].message.content.strip()
            
            return {
                "similarity_score": round(similarity * 100, 2),
                "ai_analysis": ai_analysis
            }
        
        except Exception as e:
            print(f"Error comparing documents: {e}")
            return {
                "similarity_score": 0,
                "ai_analysis": f"Error comparing documents: {str(e)}\n\nFor advanced comparison analysis, please ensure your OpenAI API key is properly configured."
            }
    
    def save_document(self, filename, content, summary, key_points, page_count):
        """Save document information to database"""
        conn = sqlite3.connect('pdf_analyzer.db')
        cursor = conn.cursor()
        
        word_count = len(content.split())
        
        cursor.execute('''
            INSERT INTO documents (filename, content, summary, key_points, word_count, page_count)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (filename, content, summary, key_points, word_count, page_count))
        
        doc_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return doc_id
    
    def get_document(self, doc_id):
        """Get a specific document from database"""
        conn = sqlite3.connect('pdf_analyzer.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM documents WHERE id = ?', (doc_id,))
        doc = cursor.fetchone()
        conn.close()
        
        if doc:
            return {
                'id': doc[0],
                'filename': doc[1],
                'upload_date': doc[2],
                'content': doc[3],
                'summary': doc[4],
                'key_points': doc[5],
                'word_count': doc[6],
                'page_count': doc[7]
            }
        return None
    
    def get_all_documents(self):
        """Get list of all documents from database"""
        conn = sqlite3.connect('pdf_analyzer.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, filename, upload_date, word_count, page_count FROM documents ORDER BY upload_date DESC')
        docs = cursor.fetchall()
        conn.close()
        
        return [{'id': doc[0], 'filename': doc[1], 'upload_date': doc[2], 'word_count': doc[3], 'page_count': doc[4]} for doc in docs]

# Initialize the PDF analyzer
analyzer = PDFAnalyzer()

# API Routes
@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get application status"""
    return jsonify({
        'ai_available': analyzer.ai_available,
        'openai_configured': openai_api_key is not None and openai_api_key != 'your-openai-api-key-here',
        'message': 'AI features available' if analyzer.ai_available else 'Using fallback mode - configure OpenAI API key for full features'
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle PDF file upload and processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not analyzer.allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PDF files are allowed'}), 400
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract text from PDF
        text, page_count = analyzer.extract_text_from_pdf(filepath)
        text = analyzer.clean_text(text)
        
        if not text:
            return jsonify({'error': 'Could not extract text from PDF. The file might be empty, corrupted, or contain only images.'}), 400
        
        # Generate summary and extract key information using AI
        summary = analyzer.generate_summary(text)
        key_info = analyzer.extract_key_information(text)
        
        # Save document to database
        doc_id = analyzer.save_document(filename, text, summary, key_info, page_count)
        
        # Clean up temporary file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'document_id': doc_id,
            'filename': filename,
            'page_count': page_count,
            'word_count': len(text.split()),
            'summary': summary,
            'key_information': key_info,
            'ai_available': analyzer.ai_available
        })
    
    except Exception as e:
        # Clean up temporary file if it exists
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        
        print(f"Upload error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    """Get list of all uploaded documents"""
    try:
        documents = analyzer.get_all_documents()
        return jsonify({'documents': documents})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/document/<int:doc_id>', methods=['GET'])
def get_document(doc_id):
    """Get details of a specific document"""
    try:
        document = analyzer.get_document(doc_id)
        if not document:
            return jsonify({'error': 'Document not found'}), 404
        return jsonify(document)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/question', methods=['POST'])
def ask_question():
    """Answer questions about a document"""
    data = request.get_json()
    doc_id = data.get('document_id')
    question = data.get('question')
    
    if not doc_id or not question:
        return jsonify({'error': 'Document ID and question are required'}), 400
    
    try:
        document = analyzer.get_document(doc_id)
        if not document:
            return jsonify({'error': 'Document not found'}), 404
        
        answer = analyzer.answer_question(document['content'], question)
        return jsonify({
            'question': question,
            'answer': answer,
            'document': document['filename'],
            'ai_available': analyzer.ai_available
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare_documents():
    """Compare two documents"""
    data = request.get_json()
    doc1_id = data.get('document1_id')
    doc2_id = data.get('document2_id')
    
    if not doc1_id or not doc2_id:
        return jsonify({'error': 'Both document IDs are required'}), 400
    
    try:
        doc1 = analyzer.get_document(doc1_id)
        doc2 = analyzer.get_document(doc2_id)
        
        if not doc1 or not doc2:
            return jsonify({'error': 'One or both documents not found'}), 404
        
        comparison = analyzer.compare_documents(
            doc1['content'], doc2['content'],
            doc1['filename'], doc2['filename']
        )
        
        return jsonify({
            'document1': doc1['filename'],
            'document2': doc2['filename'],
            'comparison': comparison,
            'ai_available': analyzer.ai_available
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/summarize/<int:doc_id>', methods=['POST'])
def regenerate_summary():
    """Generate different types of summaries for a document"""
    data = request.get_json()
    summary_type = data.get('type', 'comprehensive')  # comprehensive, executive, bullet
    
    try:
        document = analyzer.get_document(doc_id)
        if not document:
            return jsonify({'error': 'Document not found'}), 404
        
        new_summary = analyzer.generate_summary(document['content'], summary_type)
        
        return jsonify({
            'document': document['filename'],
            'summary_type': summary_type,
            'summary': new_summary,
            'ai_available': analyzer.ai_available
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error. Please try again.'}), 500

# Run the application
if __name__ == '__main__':
    print("🚀 Starting AI PDF Analyzer...")
    print(f"📝 OpenAI API Status: {'✅ Available' if analyzer.ai_available else '⚠️ Fallback Mode'}")
    if not analyzer.ai_available:
        print("💡 To enable full AI features:")
        print("   1. Get API key from: https://platform.openai.com/api-keys")
        print("   2. Update your .env file")
        print("   3. Ensure billing is set up on your OpenAI account")
    print("🌐 Access the app at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)