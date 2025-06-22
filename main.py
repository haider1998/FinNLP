import streamlit as st
import pdfplumber
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import re
from typing import List, Dict, Tuple, Optional
import hashlib
from dataclasses import dataclass
import logging
import time
from io import BytesIO
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# Page configuration
st.set_page_config(
    page_title="🏦 Financial Reports AI Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Enhanced CSS for financial theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main {
        font-family: 'Inter', sans-serif;
    }

    .financial-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
        margin: 1rem 0;
        transition: transform 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }

    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background: #fafafa;
    }

    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 15px 15px 5px 15px;
        margin: 0.5rem 0;
        margin-left: 20%;
    }

    .assistant-message {
        background: white;
        color: #333;
        padding: 1rem;
        border-radius: 15px 15px 15px 5px;
        margin: 0.5rem 0;
        margin-right: 20%;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .financial-insight {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #ff6b6b;
    }

    .success-alert {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }

    .processing-status {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background: white;
        border-radius: 8px;
        font-weight: 500;
        padding: 0 1.5rem;
        border: 1px solid #e0e0e0;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
        color: white;
        border: none;
    }
</style>
""", unsafe_allow_html=True)


@dataclass
class FinancialMetrics:
    """Data class for financial metrics"""
    revenue: Optional[str] = None
    profit: Optional[str] = None
    assets: Optional[str] = None
    liabilities: Optional[str] = None
    cash_flow: Optional[str] = None
    debt_ratio: Optional[str] = None


class FinancialReportsAnalyzer:
    def __init__(self):
        self.initialize_session_state()
        self.setup_models()

    def initialize_session_state(self):
        """Initialize all session state variables"""
        session_vars = {
            'chat_history': [],
            'processed_documents': {},
            'current_document': None,
            'financial_metrics': {},
            'analysis_cache': {},
            'processing_status': None
        }

        for var, default in session_vars.items():
            if var not in st.session_state:
                st.session_state[var] = default

    @st.cache_resource
    def load_embedding_model(_self):
        """Load embedding model with error handling"""
        try:
            with st.spinner("🔄 Loading AI models..."):
                model = SentenceTransformer("all-MiniLM-L6-v2")  # Lighter, more stable model
                return model
        except Exception as e:
            st.error(f"❌ Failed to load embedding model: {e}")
            st.info("💡 Trying alternative model...")
            try:
                # Fallback to a different model
                model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
                return model
            except Exception as e2:
                st.error(f"❌ Failed to load fallback model: {e2}")
                return None

    def setup_models(self):
        """Setup all required models"""
        # Load embedding model
        self.embedder = self.load_embedding_model()

        # Setup Gemini
        self.setup_gemini_api()

    def setup_gemini_api(self):
        """Setup Gemini API with comprehensive error handling"""
        try:
            # Check for API key
            api_key = os.environ.get("GEMINI_API_KEY")

            if not api_key:
                st.sidebar.error("🔑 Gemini API Key Required")
                st.sidebar.info("Please set GEMINI_API_KEY environment variable or enter it below:")

                api_key_input = st.sidebar.text_input(
                    "Enter Gemini API Key:",
                    type="password",
                    help="Get your API key from https://makersuite.google.com/app/apikey"
                )

                if api_key_input:
                    os.environ["GEMINI_API_KEY"] = api_key_input
                    api_key = api_key_input
                else:
                    return False

            # Configure Gemini
            genai.configure(api_key=api_key)

            # Initialize model with safety settings
            self.model = genai.GenerativeModel(
                'gemini-2.5-pro',
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                ]
            )

            return True

        except Exception as e:
            st.error(f"❌ Failed to setup Gemini API: {e}")
            return False

    def create_document_hash(self, file_content: bytes) -> str:
        """Create unique hash for document"""
        return hashlib.sha256(file_content).hexdigest()[:16]

    def extract_financial_keywords(self, text: str) -> List[str]:
        """Extract financial keywords and terms"""
        financial_terms = [
            'revenue', 'profit', 'loss', 'assets', 'liabilities', 'cash flow',
            'EBITDA', 'operating income', 'net income', 'gross profit',
            'debt', 'equity', 'shareholders', 'dividends', 'earnings per share',
            'market cap', 'valuation', 'growth rate', 'margin', 'ROI', 'ROE'
        ]

        found_terms = []
        text_lower = text.lower()

        for term in financial_terms:
            if term.lower() in text_lower:
                found_terms.append(term)

        return found_terms

    def intelligent_chunking(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """Advanced chunking with financial context awareness"""
        # Split by financial sections if possible
        section_patterns = [
            r'(?i)(financial\s+statements?)',
            r'(?i)(income\s+statement)',
            r'(?i)(balance\s+sheet)',
            r'(?i)(cash\s+flow\s+statement)',
            r'(?i)(notes?\s+to\s+financial\s+statements?)',
            r'(?i)(management\s+discussion)',
            r'(?i)(risk\s+factors?)',
            r'(?i)(business\s+overview)'
        ]

        chunks = []
        current_pos = 0

        # Simple sentence-based chunking with overlap
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_size + sentence_size <= chunk_size:
                current_chunk += sentence + " "
                current_size += sentence_size
            else:
                if current_chunk.strip():
                    financial_terms = self.extract_financial_keywords(current_chunk)
                    chunks.append({
                        'text': current_chunk.strip(),
                        'financial_terms': financial_terms,
                        'start_pos': current_pos,
                        'size': current_size
                    })
                    current_pos += current_size - overlap

                current_chunk = sentence + " "
                current_size = sentence_size

        # Add last chunk
        if current_chunk.strip():
            financial_terms = self.extract_financial_keywords(current_chunk)
            chunks.append({
                'text': current_chunk.strip(),
                'financial_terms': financial_terms,
                'start_pos': current_pos,
                'size': current_size
            })

        return chunks

    def process_financial_document(self, uploaded_file) -> Optional[Dict]:
        """Process financial document with enhanced analysis"""
        try:
            # Read file content
            file_content = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer

            doc_hash = self.create_document_hash(file_content)

            # Check if already processed
            if doc_hash in st.session_state.processed_documents:
                st.session_state.current_document = doc_hash
                return st.session_state.processed_documents[doc_hash]

            # Create progress tracking
            progress_container = st.container()

            with progress_container:
                st.markdown("""
                <div class="processing-status">
                    <h4>🔄 Processing Financial Document</h4>
                    <p>Analyzing your financial report with AI...</p>
                </div>
                """, unsafe_allow_html=True)

                progress_bar = st.progress(0.0)
                status_text = st.empty()

                # Step 1: Extract text
                status_text.text("📖 Extracting text from document...")
                progress_bar.progress(0.1)

                pages_data = []
                total_text = ""

                with pdfplumber.open(uploaded_file) as pdf:
                    total_pages = len(pdf.pages)

                    for i, page in enumerate(pdf.pages):
                        try:
                            text = page.extract_text()
                            if text:
                                pages_data.append({
                                    'page_num': i + 1,
                                    'text': text,
                                    'char_count': len(text),
                                    'financial_terms': self.extract_financial_keywords(text)
                                })
                                total_text += text + "\n"

                            # Update progress safely
                            if total_pages > 0:
                                page_progress = 0.1 + (i + 1) / total_pages * 0.3
                                progress_bar.progress(min(page_progress, 0.4))

                        except Exception as e:
                            st.warning(f"⚠️ Could not extract text from page {i + 1}: {e}")
                            continue

                if not total_text.strip():
                    st.error("❌ No text found in the document. Please ensure it's a text-based PDF.")
                    return None

                # Step 2: Intelligent chunking
                status_text.text("✂️ Creating intelligent text segments...")
                progress_bar.progress(0.5)

                chunks_data = self.intelligent_chunking(total_text)

                # Step 3: Generate embeddings
                status_text.text("🧠 Generating semantic embeddings...")
                progress_bar.progress(0.7)

                if self.embedder is None:
                    st.error("❌ Embedding model not loaded. Please refresh the page.")
                    return None

                try:
                    chunk_texts = [chunk['text'] for chunk in chunks_data]
                    embeddings = self.embedder.encode(chunk_texts, convert_to_numpy=True)

                    # Create FAISS index
                    dimension = embeddings.shape[1]
                    index = faiss.IndexFlatL2(dimension)
                    index.add(embeddings.astype(np.float32))

                except Exception as e:
                    st.error(f"❌ Failed to create embeddings: {e}")
                    return None

                # Step 4: Financial analysis
                status_text.text("📊 Analyzing financial content...")
                progress_bar.progress(0.9)

                # Extract financial metrics
                financial_metrics = self.extract_financial_metrics(total_text)

                # Complete processing
                progress_bar.progress(1.0)
                status_text.text("✅ Processing complete!")

                # Prepare result
                result = {
                    'file_name': uploaded_file.name,
                    'file_hash': doc_hash,
                    'pages_data': pages_data,
                    'chunks_data': chunks_data,
                    'embeddings': embeddings,
                    'faiss_index': index,
                    'total_text': total_text,
                    'financial_metrics': financial_metrics,
                    'processing_time': datetime.now(),
                    'stats': {
                        'total_pages': len(pages_data),
                        'total_chunks': len(chunks_data),
                        'total_characters': len(total_text),
                        'total_words': len(total_text.split()),
                        'avg_chunk_size': np.mean([chunk['size'] for chunk in chunks_data])
                    }
                }

                # Cache result
                st.session_state.processed_documents[doc_hash] = result
                st.session_state.current_document = doc_hash

                # Clear progress indicators
                time.sleep(1)
                progress_container.empty()

                return result

        except Exception as e:
            st.error(f"❌ Error processing document: {e}")
            return None

    def extract_financial_metrics(self, text: str) -> FinancialMetrics:
        """Extract key financial metrics from text"""
        metrics = FinancialMetrics()

        # Revenue patterns
        revenue_patterns = [
            r'(?i)revenue[:\s]+\$?([\d,\.]+\s*(?:million|billion|thousand)?)',
            r'(?i)total\s+revenue[:\s]+\$?([\d,\.]+\s*(?:million|billion|thousand)?)',
            r'(?i)net\s+sales[:\s]+\$?([\d,\.]+\s*(?:million|billion|thousand)?)'
        ]

        for pattern in revenue_patterns:
            match = re.search(pattern, text)
            if match:
                metrics.revenue = match.group(1)
                break

        # Profit patterns
        profit_patterns = [
            r'(?i)net\s+income[:\s]+\$?([\d,\.]+\s*(?:million|billion|thousand)?)',
            r'(?i)profit[:\s]+\$?([\d,\.]+\s*(?:million|billion|thousand)?)',
            r'(?i)earnings[:\s]+\$?([\d,\.]+\s*(?:million|billion|thousand)?)'
        ]

        for pattern in profit_patterns:
            match = re.search(pattern, text)
            if match:
                metrics.profit = match.group(1)
                break

        return metrics

    def semantic_search(self, query: str, k: int = 5) -> List[Tuple[str, float, List[str]]]:
        """Perform semantic search with financial context"""
        if not st.session_state.current_document:
            return []

        doc_data = st.session_state.processed_documents[st.session_state.current_document]

        try:
            # Get query embedding
            query_embedding = self.embedder.encode([query])

            # Search in FAISS index
            distances, indices = doc_data['faiss_index'].search(
                query_embedding.astype(np.float32), k
            )

            # Prepare results with relevance scores and financial terms
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(doc_data['chunks_data']):
                    chunk = doc_data['chunks_data'][idx]
                    relevance_score = 1 / (1 + distance)  # Convert distance to similarity

                    results.append((
                        chunk['text'],
                        relevance_score,
                        chunk['financial_terms']
                    ))

            return results

        except Exception as e:
            st.error(f"❌ Search error: {e}")
            return []

    def generate_financial_answer(self, question: str, context_chunks: List[str]) -> str:
        """Generate comprehensive financial analysis answer"""
        context = "\n\n".join(context_chunks)

        prompt = f"""You are a professional financial analyst with expertise in corporate finance, accounting, and investment analysis. 

Analyze the following context from a corporate financial report and provide a comprehensive, accurate answer to the user's question.

FINANCIAL REPORT CONTEXT:
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Provide specific, data-driven insights based on the context
- Include relevant financial figures, ratios, and metrics when available
- Explain financial concepts clearly for both experts and non-experts
- Highlight key trends, risks, and opportunities
- If information is limited, clearly state what additional data would be helpful
- Structure your response professionally with clear sections
- Use bullet points for key findings when appropriate

FINANCIAL ANALYSIS:"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 1500,
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "top_k": 40
                }
            )
            return response.text.strip()

        except Exception as e:
            return f"❌ Error generating analysis: {e}\n\nPlease check your API connection and try again."

    def generate_executive_summary(self) -> str:
        """Generate comprehensive executive summary"""
        if not st.session_state.current_document:
            return "No document loaded."

        doc_data = st.session_state.processed_documents[st.session_state.current_document]

        # Get key sections
        sample_chunks = []
        chunks = doc_data['chunks_data']

        # Strategic sampling of chunks
        if len(chunks) > 0:
            sample_chunks.append(chunks[0]['text'])  # Beginning
        if len(chunks) > 10:
            sample_chunks.append(chunks[len(chunks) // 3]['text'])  # First third
            sample_chunks.append(chunks[2 * len(chunks) // 3]['text'])  # Second third
        if len(chunks) > 1:
            sample_chunks.append(chunks[-1]['text'])  # End

        context = "\n\n".join(sample_chunks)

        prompt = f"""As a senior financial analyst, create a comprehensive executive summary of this corporate financial report.

REPORT CONTENT SAMPLE:
{context}

Create a professional executive summary covering:

1. **COMPANY OVERVIEW & BUSINESS MODEL**
2. **FINANCIAL PERFORMANCE HIGHLIGHTS**
3. **KEY FINANCIAL METRICS & RATIOS**
4. **STRATEGIC INITIATIVES & INVESTMENTS**
5. **RISK FACTORS & CHALLENGES**
6. **FUTURE OUTLOOK & GUIDANCE**
7. **RECOMMENDATIONS FOR STAKEHOLDERS**

Focus on quantitative data, trends, and strategic implications. Make it suitable for executives, investors, and analysts."""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 2000,
                    "temperature": 0.3
                }
            )
            return response.text.strip()

        except Exception as e:
            return f"❌ Error generating summary: {e}"


def main():
    """Main application function"""

    # Header
    st.markdown("""
    <div class="financial-header">
        <h1>🏦 AI-Powered Financial Reports Analyzer</h1>
        <p>Advanced Corporate Annual & Financial Reports Analysis with Google Gemini AI</p>
        <small>Upload annual reports, 10-K filings, financial statements for instant AI analysis</small>
    </div>
    """, unsafe_allow_html=True)

    # Initialize analyzer
    analyzer = FinancialReportsAnalyzer()

    # Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ Configuration")

        # API Key status
        if os.environ.get("GEMINI_API_KEY"):
            st.success("✅ Gemini API Connected")
        else:
            st.warning("⚠️ API Key Required")

        st.markdown("### 📊 Quick Stats")
        if st.session_state.current_document:
            doc_data = st.session_state.processed_documents[st.session_state.current_document]
            stats = doc_data['stats']

            st.metric("📄 Pages", stats['total_pages'])
            st.metric("📝 Text Chunks", stats['total_chunks'])
            st.metric("🔤 Total Words", f"{stats['total_words']:,}")
            st.metric("⚡ Avg Chunk Size", f"{stats['avg_chunk_size']:.0f}")

            # Financial metrics if available
            fin_metrics = doc_data['financial_metrics']
            if fin_metrics.revenue:
                st.metric("💰 Revenue", fin_metrics.revenue)
            if fin_metrics.profit:
                st.metric("📈 Profit", fin_metrics.profit)

        st.markdown("### 💾 Export & Actions")

        # Export chat history
        if st.session_state.chat_history:
            chat_export = {
                'timestamp': datetime.now().isoformat(),
                'document': st.session_state.processed_documents.get(
                    st.session_state.current_document, {}
                ).get('file_name', 'Unknown'),
                'conversation': st.session_state.chat_history
            }

            st.download_button(
                "📥 Export Chat",
                data=json.dumps(chat_export, indent=2),
                file_name=f"financial_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        # Clear actions
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

        if st.button("🔄 Reset All"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # File upload section
    st.markdown("### 📁 Upload Financial Document")

    uploaded_file = st.file_uploader(
        "Choose a financial report (PDF)",
        type="pdf",
        help="Upload annual reports, 10-K/10-Q filings, financial statements, or any corporate financial document"
    )

    if uploaded_file:
        # Process document
        doc_data = analyzer.process_financial_document(uploaded_file)

        if doc_data:
            # Success message
            st.markdown(f"""
            <div class="success-alert">
                <h4>✅ Document Successfully Processed!</h4>
                <p><strong>{doc_data['file_name']}</strong> has been analyzed with {doc_data['stats']['total_chunks']} intelligent segments.</p>
                <p>🎯 Ready for AI-powered financial analysis and insights!</p>
            </div>
            """, unsafe_allow_html=True)

            # Create enhanced tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "💬 AI Chat",
                "📊 Executive Summary",
                "🔍 Advanced Search",
                "📈 Financial Analysis",
                "📋 Document Insights"
            ])

            with tab1:
                st.markdown("### 💬 Chat with Financial AI Assistant")

                # Display chat history
                if st.session_state.chat_history:
                    chat_container = st.container()
                    with chat_container:
                        for chat in st.session_state.chat_history:
                            if chat['type'] == 'user':
                                st.markdown(f"""
                                <div class="user-message">
                                    <strong>You:</strong> {chat['content']}
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="assistant-message">
                                    <strong>🤖 Financial AI:</strong><br>
                                    {chat['content']}
                                </div>
                                """, unsafe_allow_html=True)

                # Chat input
                question = st.text_input(
                    "Ask about the financial report:",
                    placeholder="e.g., What was the revenue growth? What are the key risks? How is the cash flow?"
                )

                col1, col2 = st.columns([4, 1])

                with col1:
                    if st.button("🚀 Ask AI", type="primary") and question:
                        # Add user message
                        st.session_state.chat_history.append({
                            'type': 'user',
                            'content': question,
                            'timestamp': datetime.now().isoformat()
                        })

                        # Get relevant context
                        with st.spinner("🔍 Searching financial data..."):
                            search_results = analyzer.semantic_search(question, k=4)
                            context_chunks = [result[0] for result in search_results]

                        # Generate answer
                        with st.spinner("🧠 Analyzing with AI..."):
                            answer = analyzer.generate_financial_answer(question, context_chunks)

                        # Add AI response
                        st.session_state.chat_history.append({
                            'type': 'assistant',
                            'content': answer,
                            'timestamp': datetime.now().isoformat(),
                            'relevance_scores': [score for _, score, _ in search_results]
                        })

                        st.rerun()

                with col2:
                    if st.button("🧹 Clear"):
                        st.session_state.chat_history = []
                        st.rerun()

                # Quick questions for financial analysis
                if not st.session_state.chat_history:
                    st.markdown("### 💡 Financial Analysis Quick Start")

                    financial_questions = [
                        "📊 What are the key financial highlights and performance metrics?",
                        "💰 How did revenue and profitability change compared to last year?",
                        "⚠️ What are the main risk factors and challenges facing the company?",
                        "🎯 What is the company's strategic outlook and future plans?",
                        "💸 How is the company's cash flow and liquidity position?",
                        "📈 What are the key growth drivers and market opportunities?"
                    ]

                    cols = st.columns(2)
                    for i, q in enumerate(financial_questions):
                        with cols[i % 2]:
                            if st.button(q, key=f"quick_q_{i}"):
                                # Process quick question
                                st.session_state.chat_history.append({
                                    'type': 'user',
                                    'content': q,
                                    'timestamp': datetime.now().isoformat()
                                })

                                search_results = analyzer.semantic_search(q, k=4)
                                context_chunks = [result[0] for result in search_results]
                                answer = analyzer.generate_financial_answer(q, context_chunks)

                                st.session_state.chat_history.append({
                                    'type': 'assistant',
                                    'content': answer,
                                    'timestamp': datetime.now().isoformat()
                                })
                                st.rerun()

            with tab2:
                st.markdown("### 📊 Executive Summary")

                if st.button("🎯 Generate Executive Summary", type="primary"):
                    with st.spinner("📝 Creating comprehensive executive summary..."):
                        summary = analyzer.generate_executive_summary()

                        st.markdown(f"""
                        <div class="financial-insight">
                            <h4>📋 Executive Summary - {doc_data['file_name']}</h4>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown(summary)

            with tab3:
                st.markdown("### 🔍 Advanced Financial Search")

                search_query = st.text_input(
                    "Search within the financial document:",
                    placeholder="e.g., debt ratio, operating expenses, market share"
                )

                if search_query:
                    search_results = analyzer.semantic_search(search_query, k=8)

                    st.markdown(f"### 📋 Found {len(search_results)} relevant sections")

                    for i, (text, score, fin_terms) in enumerate(search_results, 1):
                        with st.expander(f"📄 Result {i} - Relevance: {score:.3f}"):
                            st.write(text)

                            if fin_terms:
                                st.markdown("**🏷️ Financial Terms Found:**")
                                st.write(", ".join(fin_terms))

            with tab4:
                st.markdown("### 📈 Financial Analysis Dashboard")

                # Document statistics
                stats = doc_data['stats']

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📄 Total Pages", stats['total_pages'])
                with col2:
                    st.metric("📝 Text Segments", stats['total_chunks'])
                with col3:
                    st.metric("🔤 Word Count", f"{stats['total_words']:,}")
                with col4:
                    st.metric("📊 Avg Segment Size", f"{stats['avg_chunk_size']:.0f}")

                # Financial terms analysis
                st.markdown("### 🏷️ Financial Terms Distribution")

                # Collect all financial terms
                all_terms = []
                for chunk in doc_data['chunks_data']:
                    all_terms.extend(chunk['financial_terms'])

                if all_terms:
                    # Create frequency distribution
                    terms_df = pd.DataFrame(all_terms, columns=['term'])
                    term_counts = terms_df['term'].value_counts().head(15)

                    fig = px.bar(
                        x=term_counts.values,
                        y=term_counts.index,
                        orientation='h',
                        title="Top Financial Terms in Document",
                        labels={'x': 'Frequency', 'y': 'Financial Terms'}
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

                # Page analysis
                st.markdown("### 📊 Page-wise Content Analysis")
                pages_df = pd.DataFrame(doc_data['pages_data'])

                if not pages_df.empty:
                    fig2 = px.line(
                        pages_df,
                        x='page_num',
                        y='char_count',
                        title="Content Distribution Across Pages",
                        labels={'page_num': 'Page Number', 'char_count': 'Character Count'}
                    )
                    st.plotly_chart(fig2, use_container_width=True)

            with tab5:
                st.markdown("### 📋 Document Intelligence Insights")

                # Document metadata
                st.markdown("#### 📄 Document Information")
                info_cols = st.columns(3)

                with info_cols[0]:
                    st.info(f"**📂 File Name:** {doc_data['file_name']}")
                    st.info(f"**🕒 Processed:** {doc_data['processing_time'].strftime('%Y-%m-%d %H:%M:%S')}")

                with info_cols[1]:
                    st.info(f"**📊 Pages:** {doc_data['stats']['total_pages']}")
                    st.info(f"**✂️ Segments:** {doc_data['stats']['total_chunks']}")

                with info_cols[2]:
                    st.info(f"**📝 Words:** {doc_data['stats']['total_words']:,}")
                    st.info(f"**📏 Characters:** {doc_data['stats']['total_characters']:,}")

                # Financial metrics summary
                fin_metrics = doc_data['financial_metrics']
                if any([fin_metrics.revenue, fin_metrics.profit]):
                    st.markdown("#### 💰 Extracted Financial Metrics")
                    metrics_cols = st.columns(3)

                    with metrics_cols[0]:
                        if fin_metrics.revenue:
                            st.success(f"**💰 Revenue:** {fin_metrics.revenue}")

                    with metrics_cols[1]:
                        if fin_metrics.profit:
                            st.success(f"**📈 Profit:** {fin_metrics.profit}")

                    with metrics_cols[2]:
                        st.info("**🔄 More metrics:** Available in AI chat")

                # Sample content preview
                st.markdown("#### 👀 Document Content Preview")
                if doc_data['chunks_data']:
                    preview_text = doc_data['chunks_data'][0]['text'][:500] + "..."
                    st.text_area("First segment preview:", preview_text, height=150, disabled=True)


if __name__ == "__main__":
    main()
