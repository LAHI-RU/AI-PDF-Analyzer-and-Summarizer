# app.py - Fixed for OpenAI API v1.0+
import os
import sqlite3
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pdfplumber
from openai import OpenAI  # Updated import
from dotenv import load_dotenv
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class PDFAnalyzer:
    def __init__(self):
        self.init_database()
    
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
    
    def generate_summary(self, text, summary_type="comprehensive"):
        """Generate AI-powered summary using OpenAI API"""
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
            return f"Error generating summary: {str(e)}"
    
    def extract_key_information(self, text):
        """Extract key information from document using AI"""
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
            return f"Error extracting key information: {str(e)}"
    
    def answer_question(self, text, question):
        """Answer questions about the document using AI"""
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
            return f"Error answering question: {str(e)}"
    
    def compare_documents(self, doc1_text, doc2_text, doc1_name, doc2_name):
        """Compare two documents for similarities and differences"""
        try:
            # Calculate similarity using TF-IDF
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform([doc1_text, doc2_text])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
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
            return {"error": f"Error comparing documents: {str(e)}"}
    
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
            return jsonify({'error': 'Could not extract text from PDF'}), 400
        
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
            'key_information': key_info
        })
    
    except Exception as e:
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
            'document': document['filename']
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
            'comparison': comparison
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
            'summary': new_summary
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the application
if __name__ == '__main__':
    print("🚀 Starting PDF Analyzer...")
    print("📝 Make sure you have added your OpenAI API key to the .env file")
    print("🌐 Access the app at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)