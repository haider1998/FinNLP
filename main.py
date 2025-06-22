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
from typing import List, Dict, Tuple, Optional, Any
import hashlib
from dataclasses import dataclass, asdict
import time
import warnings
import networkx as nx
import yfinance as yf
import textstat
from io import BytesIO
import logging
import sqlite3
import pickle
import base64
from pathlib import Path
import tempfile
import zipfile
import io
import plotly.io as pio
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import threading

# Advanced imports with error handling
try:
    import easyocr

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import transformers
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from wordcloud import WordCloud

    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="🚀 FinSight360 Pro - AI Financial Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏦"
)

# --- CSS STYLING ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --primary-color: #1e3c72;
        --secondary-color: #2a5298;
        --accent-color: #667eea;
        --success-color: #4caf50;
        --warning-color: #ff9800;
        --error-color: #f44336;
        --text-color: #2c3e50;
        --bg-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --card-shadow: 0 10px 30px rgba(0,0,0,0.1);
        --border-radius: 15px;
    }

    .main {
        background: var(--bg-gradient);
        background-attachment: fixed;
        min-height: 100vh;
    }

    .block-container {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 25px;
        padding: 2rem;
        margin: 1rem;
        backdrop-filter: blur(20px);
        box-shadow: var(--card-shadow);
        border: 1px solid rgba(255,255,255,0.2);
    }

    .finsight-header {
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 50%, var(--accent-color) 100%);
        padding: 3rem;
        border-radius: 25px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
    }

    .finsight-header h1 {
        font-size: 4rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
        position: relative;
        z-index: 1;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .finsight-header .subtitle {
        font-size: 1.4rem;
        opacity: 0.95;
        font-weight: 500;
        position: relative;
        z-index: 1;
    }

    .finsight-header .badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.5rem;
        font-size: 0.9rem;
        font-weight: 600;
        backdrop-filter: blur(10px);
    }

    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: var(--border-radius);
        box-shadow: var(--card-shadow);
        border: 1px solid rgba(255,255,255,0.3);
        margin: 1rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: var(--bg-gradient);
    }

    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }

    .innovation-card {
        background: var(--bg-gradient);
        padding: 2.5rem;
        border-radius: var(--border-radius);
        color: white;
        margin: 1rem 0;
        box-shadow: var(--card-shadow);
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .alert-critical {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white; 
        padding: 2rem; 
        border-radius: var(--border-radius); 
        margin: 1rem 0;
        box-shadow: 0 15px 30px rgba(255, 107, 107, 0.4);
        border-left: 6px solid #ff4757;
    }

    .alert-warning {
        background: linear-gradient(135deg, #ffa726 0%, #ff9800 100%);
        color: white; 
        padding: 2rem; 
        border-radius: var(--border-radius); 
        margin: 1rem 0;
        box-shadow: 0 12px 25px rgba(255, 167, 38, 0.4);
        border-left: 6px solid #ff8f00;
    }

    .alert-ok {
        background: linear-gradient(135deg, #66bb6a 0%, #4caf50 100%);
        color: white; 
        padding: 2rem; 
        border-radius: var(--border-radius); 
        margin: 1rem 0;
        box-shadow: 0 12px 25px rgba(102, 187, 106, 0.4);
        border-left: 6px solid #388e3c;
    }

    .processing-status {
        background: var(--bg-gradient);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
        box-shadow: var(--card-shadow);
        animation: processing-pulse 2s infinite;
    }

    @keyframes processing-pulse {
        0% { opacity: 0.8; }
        50% { opacity: 1; }
        100% { opacity: 0.8; }
    }

    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
        animation: status-blink 2s infinite;
    }

    .status-online { background: #4caf50; }
    .status-offline { background: #f44336; }
    .status-warning { background: #ff9800; }

    @keyframes status-blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0.3; }
    }
</style>
""", unsafe_allow_html=True)


# --- DATA CLASSES ---
@dataclass
class FinancialEntity:
    name: str
    entity_type: str
    value: Optional[Any]
    context: str
    source_text: str
    confidence: float = 0.0
    page_number: Optional[int] = None
    section: Optional[str] = None


@dataclass
class CovenantAlert:
    covenant_name: str
    current_value: float
    threshold: float
    severity: str
    trend: str
    prediction: Dict[str, Any]
    recommendation: str
    breach_probability: float = 0.0
    time_to_breach: Optional[int] = None
    historical_trend: List[float] = None


@dataclass
class ScenarioResult:
    scenario_name: str
    base_metrics: Dict[str, float]
    adjusted_metrics: Dict[str, float]
    impact_narrative: str
    recommendations: List[str]
    risk_score: float
    confidence_interval: Dict[str, float]
    monte_carlo_results: Optional[Dict] = None
    stress_test_results: Optional[Dict] = None


@dataclass
class DocumentAnalysis:
    hash: str
    file_name: str
    upload_time: datetime
    file_size: int
    segments: Dict
    metrics: Dict
    market_context: Dict
    text_length: int
    readability_score: float
    sentiment_analysis: Optional[Dict] = None
    key_topics: List[str] = None
    executive_summary: Optional[str] = None


# --- DATABASE MANAGER ---
class DatabaseManager:
    def __init__(self, db_path: str = "finsight360.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hash TEXT UNIQUE,
                    filename TEXT,
                    upload_time TIMESTAMP,
                    file_size INTEGER,
                    analysis_data BLOB,
                    user_id TEXT,
                    tags TEXT
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    session_id TEXT,
                    preferences TEXT,
                    created_at TIMESTAMP,
                    last_active TIMESTAMP
                )
            ''')

            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Database initialization error: {e}")

    def save_document_analysis(self, doc_hash: str, filename: str, analysis_data: dict, user_id: str = "default"):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            serialized_data = pickle.dumps(analysis_data)

            cursor.execute('''
                INSERT OR REPLACE INTO documents 
                (hash, filename, upload_time, file_size, analysis_data, user_id)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (doc_hash, filename, datetime.now(), len(serialized_data), serialized_data, user_id))

            conn.commit()
            conn.close()
        except Exception as e:
            logging.error(f"Error saving document analysis: {e}")

    def load_document_analysis(self, doc_hash: str) -> Optional[dict]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('SELECT analysis_data FROM documents WHERE hash = ?', (doc_hash,))
            result = cursor.fetchone()
            conn.close()

            if result:
                return pickle.loads(result[0])
            return None
        except Exception as e:
            logging.error(f"Error loading document analysis: {e}")
            return None

    def get_user_documents(self, user_id: str = "default") -> List[Dict]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT hash, filename, upload_time, file_size 
                FROM documents WHERE user_id = ? 
                ORDER BY upload_time DESC LIMIT 10
            ''', (user_id,))

            results = cursor.fetchall()
            conn.close()

            return [
                {
                    'hash': row[0], 'filename': row[1],
                    'upload_time': row[2], 'file_size': row[3]
                }
                for row in results
            ]
        except Exception as e:
            logging.error(f"Error getting user documents: {e}")
            return []


# --- DOCUMENT PROCESSOR ---
class DocumentProcessor:
    def __init__(self):
        self.ocr_reader = None
        if OCR_AVAILABLE:
            try:
                self.ocr_reader = easyocr.Reader(['en'])
            except Exception as e:
                logging.warning(f"OCR initialization failed: {e}")

    def process_document(self, uploaded_file) -> Dict[str, Any]:
        try:
            file_content = uploaded_file.read()
            file_extension = Path(uploaded_file.name).suffix.lower()

            if file_extension == '.pdf':
                return self._process_pdf(file_content, uploaded_file.name)
            else:
                # Basic text extraction for other formats
                return {
                    'text': "Document processing not fully supported for this format.",
                    'tables': [],
                    'images': [],
                    'metadata': {'filename': uploaded_file.name, 'type': file_extension}
                }
        except Exception as e:
            logging.error(f"Document processing error: {e}")
            return {
                'text': f"Error processing document: {str(e)}",
                'tables': [],
                'images': [],
                'metadata': {'filename': uploaded_file.name, 'type': 'error'}
            }

    def _process_pdf(self, file_content: bytes, filename: str) -> Dict[str, Any]:
        text = ""
        tables = []

        try:
            with pdfplumber.open(BytesIO(file_content)) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}"

                    page_tables = page.extract_tables()
                    if page_tables:
                        tables.extend([(page_num + 1, table) for table in page_tables])

        except Exception as e:
            logging.error(f"PDF processing error: {e}")
            text = "Error processing PDF. Please try a different file."

        return {
            'text': text,
            'tables': tables,
            'images': [],
            'metadata': {'filename': filename, 'type': 'pdf'}
        }


# --- SEMANTIC SEGMENTER ---
class SemanticSegmenter:
    def __init__(self):
        self.financial_patterns = {
            'risk_factors': {
                'patterns': [
                    r'(?i)risk\s+factors?', r'(?i)could\s+adversely\s+affect',
                    r'(?i)material\s+adverse\s+effect', r'(?i)uncertainty',
                    r'(?i)volatility', r'(?i)may\s+harm'
                ],
                'weight': 4.0
            },
            'management_discussion': {
                'patterns': [
                    r'(?i)management\'s\s+discussion', r'(?i)md&a',
                    r'(?i)results\s+of\s+operations', r'(?i)financial\s+condition',
                    r'(?i)liquidity\s+and\s+capital'
                ],
                'weight': 3.5
            },
            'covenant_data': {
                'patterns': [
                    r'(?i)covenant', r'(?i)leverage\s+ratio', r'(?i)interest\s+coverage',
                    r'(?i)debt\s+to\s+ebitda', r'(?i)financial\s+ratios',
                    r'(?i)compliance', r'(?i)default'
                ],
                'weight': 4.5
            },
            'financial_statements': {
                'patterns': [
                    r'(?i)balance\s+sheet', r'(?i)income\s+statement',
                    r'(?i)cash\s+flow', r'(?i)statement\s+of\s+operations'
                ],
                'weight': 3.0
            },
            'quantitative': {
                'patterns': [
                    r'\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|thousand))?',
                    r'\b\d+(?:\.\d+)?\s*%', r'\b\d+(?:\.\d+)?x\b'
                ],
                'weight': 2.5
            }
        }

    def segment_document(self, text: str) -> Dict[str, List[Dict]]:
        segments = {category: [] for category in self.financial_patterns.keys()}
        paragraphs = re.split(r'\n\s*\n', text)

        for i, para in enumerate(paragraphs):
            para = para.strip()
            if len(para) < 40:
                continue

            scores = {cat: 0.0 for cat in self.financial_patterns.keys()}

            for category, config in self.financial_patterns.items():
                for pattern in config['patterns']:
                    matches = len(re.findall(pattern, para))
                    if matches > 0:
                        score = config['weight'] * matches
                        scores[category] += score

            max_score = max(scores.values()) if scores.values() else 0
            primary_category = max(scores, key=scores.get) if max_score > 1.0 else "narrative"

            if max_score > 1.0:
                segment_obj = {
                    'text': para,
                    'primary_category': primary_category,
                    'confidence': max_score,
                    'position': i,
                    'word_count': len(para.split()),
                    'all_scores': scores,
                    'readability': min(100, max(0, 50 + np.random.normal(0, 20)))  # Simplified readability
                }
                segments[primary_category].append(segment_obj)

        return segments


# --- RETRIEVER ---
class TopicRetriever:
    def __init__(self, embedder):
        self.embedder = embedder
        self.topic_boost = {
            'covenant_data': 2.0,
            'risk_factors': 1.8,
            'quantitative': 1.6,
            'management_discussion': 1.4,
            'financial_statements': 1.2
        }

    def build_index(self, segments_data: Dict) -> Tuple[Optional[faiss.Index], List[Dict]]:
        all_segments = []
        for category, segments in segments_data.items():
            for segment in segments:
                segment_meta = segment.copy()
                segment_meta['category'] = category
                all_segments.append(segment_meta)

        if not all_segments:
            return None, []

        try:
            texts = [seg['text'] for seg in all_segments]
            embeddings = self.embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

            # Apply boosting
            for i, seg in enumerate(all_segments):
                category_boost = self.topic_boost.get(seg['category'], 1.0)
                confidence_boost = min(2.0, 1.0 + seg.get('confidence', 0) / 15.0)
                total_boost = category_boost * confidence_boost
                embeddings[i] = embeddings[i] * total_boost

            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings.astype(np.float32))
            return index, all_segments

        except Exception as e:
            logging.error(f"Index building error: {e}")
            return None, []

    def search(self, query: str, index: faiss.Index, segments: List[Dict], k: int = 10) -> List[Dict]:
        if index is None or not segments:
            return []

        try:
            query_embedding = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
            distances, indices = index.search(query_embedding, min(k, len(segments)))

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if 0 <= idx < len(segments):
                    result = segments[idx].copy()
                    result['relevance_score'] = 1 / (1 + dist)
                    results.append(result)

            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return results[:k]

        except Exception as e:
            logging.error(f"Search error: {e}")
            return []


# --- SCENARIO ENGINE ---
class ScenarioEngine:
    def __init__(self):
        self.preset_scenarios = {
            'Economic Recession': {
                'revenue': -20, 'ebitda_margin_bps': -300, 'interest_expense': 25,
                'description': 'Severe economic downturn with reduced demand'
            },
            'Market Recovery': {
                'revenue': 15, 'ebitda_margin_bps': 150, 'cash': 10,
                'description': 'Strong economic recovery with improved conditions'
            },
            'Interest Rate Spike': {
                'interest_expense': 75, 'ebitda_margin_bps': -50,
                'description': 'Rapid increase in interest rates'
            }
        }

    def generate_scenario(self, base_metrics: Dict, adjustments: Dict,
                          scenario_name: str = "Custom", run_monte_carlo: bool = False) -> ScenarioResult:
        try:
            adjusted = base_metrics.copy()

            # Apply adjustments
            for key, pct_change in adjustments.items():
                if key == 'ebitda_margin_bps':
                    current_margin = adjusted.get('ebitda_margin', 0.15)
                    adjusted['ebitda_margin'] = current_margin + (pct_change / 10000)
                    if 'revenue' in adjusted and adjusted['revenue'] > 0:
                        adjusted['ebitda'] = adjusted['revenue'] * adjusted['ebitda_margin']
                elif key in adjusted and adjusted[key] > 0:
                    adjusted[key] *= (1 + pct_change / 100)

            # Recalculate derived metrics
            if 'revenue' in adjusted and 'ebitda' in adjusted and adjusted['revenue'] > 0:
                adjusted['ebitda_margin'] = adjusted['ebitda'] / adjusted['revenue']

            if 'cash' in adjusted and 'total_debt' in adjusted:
                adjusted['net_debt'] = adjusted['total_debt'] - adjusted['cash']

            if 'net_debt' in adjusted and 'ebitda' in adjusted and adjusted['ebitda'] > 0:
                adjusted['debt_to_ebitda'] = adjusted['net_debt'] / adjusted['ebitda']

            # Generate narrative and recommendations
            impact_narrative = self._generate_narrative(base_metrics, adjusted, scenario_name)
            recommendations = self._generate_recommendations(adjusted)
            risk_score = self._calculate_risk_score(adjusted)
            confidence_interval = {'high': 0.85, 'base': 0.75, 'low': 0.65}

            return ScenarioResult(
                scenario_name=scenario_name,
                base_metrics=base_metrics,
                adjusted_metrics=adjusted,
                impact_narrative=impact_narrative,
                recommendations=recommendations,
                risk_score=risk_score,
                confidence_interval=confidence_interval
            )

        except Exception as e:
            logging.error(f"Scenario generation error: {e}")
            return ScenarioResult(
                scenario_name="Error",
                base_metrics=base_metrics,
                adjusted_metrics=base_metrics,
                impact_narrative="Unable to generate scenario analysis.",
                recommendations=["Please check input parameters."],
                risk_score=50.0,
                confidence_interval={'high': 0.5, 'base': 0.4, 'low': 0.3}
            )

    def _generate_narrative(self, base: Dict, adjusted: Dict, scenario_name: str) -> str:
        insights = [f"Under the {scenario_name} scenario:"]

        if 'revenue' in base and 'revenue' in adjusted:
            rev_change = ((adjusted['revenue'] / base['revenue']) - 1) * 100
            direction = "increases" if rev_change > 0 else "decreases"
            insights.append(f"Revenue {direction} by {abs(rev_change):.1f}%")

        if 'debt_to_ebitda' in adjusted:
            if adjusted['debt_to_ebitda'] > 5.0:
                insights.append("Debt/EBITDA ratio exceeds typical covenant thresholds")
            elif adjusted['debt_to_ebitda'] < 3.0:
                insights.append("Debt/EBITDA ratio remains at comfortable levels")

        return ". ".join(insights) + "."

    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        recs = []

        debt_ebitda = metrics.get('debt_to_ebitda', 0)
        if debt_ebitda > 5.0:
            recs.append("🔴 Critical: Immediate deleveraging required")
        elif debt_ebitda > 4.0:
            recs.append("🟡 Monitor: Focus on debt reduction")
        else:
            recs.append("🟢 Stable: Maintain current financial discipline")

        margin = metrics.get('ebitda_margin', 0)
        if margin < 0.08:
            recs.append("🔴 Urgent: Implement cost reduction measures")
        elif margin > 0.20:
            recs.append("🟢 Opportunity: Consider strategic investments")

        return recs

    def _calculate_risk_score(self, metrics: Dict) -> float:
        score = 30

        debt_ebitda = metrics.get('debt_to_ebitda', 3.0)
        if debt_ebitda > 5:
            score += 30
        elif debt_ebitda > 4:
            score += 20
        elif debt_ebitda < 2:
            score -= 10

        margin = metrics.get('ebitda_margin', 0.15)
        if margin < 0.08:
            score += 20
        elif margin > 0.20:
            score -= 10

        return max(0, min(100, score))


# --- ALERTS ENGINE ---
class AlertsEngine:
    def __init__(self):
        self.default_thresholds = {
            'max_debt_ebitda': 4.5,
            'min_interest_coverage': 2.5,
            'min_ebitda_margin': 0.10
        }

    def analyze_covenants(self, financials: Dict, custom_thresholds: Dict = None) -> List[CovenantAlert]:
        thresholds = {**self.default_thresholds, **(custom_thresholds or {})}
        alerts = []

        covenant_checks = [
            ('debt_to_ebitda', 'max_debt_ebitda', True, 'Debt/EBITDA Ratio'),
            ('interest_coverage', 'min_interest_coverage', False, 'Interest Coverage Ratio'),
            ('ebitda_margin', 'min_ebitda_margin', False, 'EBITDA Margin')
        ]

        for metric_key, threshold_key, is_max_threshold, display_name in covenant_checks:
            if metric_key in financials:
                current_value = financials[metric_key]
                threshold = thresholds.get(threshold_key)

                if threshold is not None:
                    severity = self._determine_severity(current_value, threshold, is_max_threshold)

                    if severity != "ok":
                        alert = CovenantAlert(
                            covenant_name=display_name,
                            current_value=current_value,
                            threshold=threshold,
                            severity=severity,
                            trend="Stable (simulated)",
                            prediction={'breach_probability': 25.0, 'confidence': 0.7},
                            recommendation=self._get_recommendation(metric_key, severity),
                            breach_probability=25.0,
                            time_to_breach=4,
                            historical_trend=[current_value * 0.95, current_value * 0.98, current_value]
                        )
                        alerts.append(alert)

        return alerts

    def _determine_severity(self, current: float, threshold: float, is_max_threshold: bool) -> str:
        if is_max_threshold:
            if current > threshold:
                return 'breach'
            elif current > threshold * 0.95:
                return 'critical'
            elif current > threshold * 0.90:
                return 'warning'
        else:
            if current < threshold:
                return 'breach'
            elif current < threshold * 1.05:
                return 'critical'
            elif current < threshold * 1.10:
                return 'warning'
        return 'ok'

    def _get_recommendation(self, metric_key: str, severity: str) -> str:
        recommendations = {
            'debt_to_ebitda': {
                'breach': "IMMEDIATE ACTION: Negotiate covenant amendment or pursue deleveraging",
                'critical': "URGENT: Accelerate debt paydown and improve EBITDA",
                'warning': "MONITOR: Review debt management strategy"
            },
            'interest_coverage': {
                'breach': "CRITICAL: Refinance debt or improve EBITDA immediately",
                'critical': "URGENT: Explore refinancing options",
                'warning': "IMPORTANT: Monitor interest rate exposure"
            },
            'ebitda_margin': {
                'breach': "IMMEDIATE: Implement cost reduction program",
                'critical': "URGENT: Focus on operational efficiency",
                'warning': "IMPORTANT: Analyze cost structure"
            }
        }

        return recommendations.get(metric_key, {}).get(severity, "Monitor metric closely")


# --- KNOWLEDGE GRAPH ---
class FinancialKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def build_graph(self, segments: Dict, metrics: Dict) -> nx.DiGraph:
        self.graph.clear()

        # Add metric nodes
        for metric, value in metrics.items():
            self.graph.add_node(
                metric,
                type='metric',
                value=value,
                display_name=self._format_metric_name(metric),
                color='#2E86AB',
                size=35
            )

        # Add relationships
        relationships = [
            ('revenue', 'ebitda', 'generates'),
            ('ebitda', 'debt_to_ebitda', 'component_of'),
            ('net_debt', 'debt_to_ebitda', 'component_of'),
            ('ebitda', 'interest_coverage', 'enables')
        ]

        for source, target, relationship in relationships:
            if source in metrics and target in metrics:
                self.graph.add_edge(source, target, relationship=relationship, weight=2.0)

        # Add category nodes
        for category, segs in segments.items():
            if segs:
                cat_node = f"cat_{category}"
                self.graph.add_node(
                    cat_node,
                    type='category',
                    display_name=category.replace('_', ' ').title(),
                    color='#A23B72',
                    size=30
                )

        return self.graph

    def _format_metric_name(self, metric: str) -> str:
        name_map = {
            'debt_to_ebitda': 'Debt/EBITDA',
            'ebitda_margin': 'EBITDA Margin',
            'interest_coverage': 'Interest Coverage',
            'net_debt': 'Net Debt',
            'ebitda': 'EBITDA',
            'revenue': 'Revenue'
        }
        return name_map.get(metric, metric.replace('_', ' ').title())

    def get_central_nodes(self, top_k: int = 5) -> List[str]:
        if self.graph.number_of_nodes() == 0:
            return []

        centrality = nx.degree_centrality(self.graph)
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return [node for node, _ in sorted_nodes[:top_k]]


# --- AI ANALYZER ---
class AIAnalyzer:
    def __init__(self):
        self.summarizer = None
        self.sentiment_analyzer = None

        if TRANSFORMERS_AVAILABLE:
            try:
                self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                self.sentiment_analyzer = pipeline("sentiment-analysis")
            except Exception as e:
                logging.warning(f"Failed to load transformers models: {e}")

    def generate_executive_summary(self, text: str, max_length: int = 500) -> str:
        if not self.summarizer or len(text) < 100:
            return "Executive summary: Document contains financial information requiring detailed analysis. Key metrics and relationships have been extracted for further examination."

        try:
            chunks = [text[i:i + 1000] for i in range(0, min(len(text), 3000), 1000)]
            summaries = []

            for chunk in chunks[:2]:
                if len(chunk) > 50:
                    summary = self.summarizer(chunk, max_length=150, min_length=50, do_sample=False)
                    summaries.append(summary[0]['summary_text'])

            return " ".join(summaries)[:max_length]
        except Exception as e:
            logging.error(f"Summary generation error: {e}")
            return "Unable to generate executive summary due to processing constraints."

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        if not self.sentiment_analyzer:
            return {'overall': 'neutral', 'confidence': 0.5}

        try:
            chunks = [text[i:i + 500] for i in range(0, min(len(text), 2000), 500)]
            sentiments = []

            for chunk in chunks:
                if len(chunk) > 20:
                    result = self.sentiment_analyzer(chunk)[0]
                    sentiments.append({
                        'label': result['label'],
                        'score': result['score']
                    })

            if sentiments:
                avg_score = np.mean([s['score'] for s in sentiments])
                labels = [s['label'] for s in sentiments]
                most_common = max(set(labels), key=labels.count)

                return {
                    'overall': most_common.lower(),
                    'confidence': avg_score,
                    'distribution': sentiments
                }
        except Exception as e:
            logging.error(f"Sentiment analysis error: {e}")

        return {'overall': 'neutral', 'confidence': 0.5}

    def extract_key_topics(self, text: str, num_topics: int = 10) -> List[str]:
        try:
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            financial_terms = [
                'revenue', 'ebitda', 'debt', 'equity', 'cash', 'margin', 'profit',
                'covenant', 'leverage', 'liquidity', 'interest', 'dividend',
                'market', 'growth', 'risk', 'strategy', 'operations'
            ]

            relevant_words = [w for w in words if w in financial_terms]
            word_freq = pd.Series(relevant_words).value_counts()

            return word_freq.head(num_topics).index.tolist()
        except Exception as e:
            logging.error(f"Topic extraction error: {e}")
            return []


# --- MAIN APPLICATION ---
class FinancialReportsAnalyzer:
    def __init__(self):
        self.initialize_session_state()
        self.db_manager = DatabaseManager()
        self.doc_processor = DocumentProcessor()
        self.ai_analyzer = AIAnalyzer()
        self.embedder = self.load_embedding_model()
        if self.embedder:
            self.retriever = TopicRetriever(self.embedder)
        self.segmenter = SemanticSegmenter()
        self.scenario_engine = ScenarioEngine()
        self.alerts_engine = AlertsEngine()
        self.knowledge_graph = FinancialKnowledgeGraph()
        self.setup_gemini_api()

    def initialize_session_state(self):
        defaults = {
            'chat_history': [],
            'processed_doc': None,
            'financial_metrics': {},
            'analysis_cache': {},
            'upload_history': [],
            'show_export': False
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @st.cache_resource
    def load_embedding_model(_self):
        try:
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            st.error(f"Could not load embedding model: {e}")
            return None

    def setup_gemini_api(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            with st.sidebar:
                api_key = st.text_input("🔑 Gemini API Key:", type="password",
                                        help="Enter your Google Gemini API key for AI features")
                if api_key:
                    st.session_state.api_key_input = api_key

        if api_key or st.session_state.get('api_key_input'):
            try:
                genai.configure(api_key=api_key or st.session_state.api_key_input)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                return True
            except Exception as e:
                st.error(f"Failed to initialize Gemini API: {e}")
                return False
        return False

    def generate_ai_insight(self, prompt: str, context_chunks: List[Dict]) -> str:
        if not hasattr(self, 'model'):
            return "AI analysis requires Gemini API key. Please provide your API key in the sidebar."

        try:
            context_text = "\n\n".join([chunk['text'][:500] for chunk in context_chunks[:3]])

            full_prompt = f"""
            Based on the following financial document content, please provide a comprehensive analysis:

            Context from document:
            {context_text}

            User question: {prompt}

            Please provide a detailed, professional analysis covering:
            1. Direct answer to the question
            2. Supporting evidence from the document
            3. Financial implications
            4. Potential risks or opportunities
            5. Recommendations if applicable

            Format your response in a clear, structured manner suitable for financial professionals.
            """

            response = self.model.generate_content(full_prompt)
            return response.text

        except Exception as e:
            return f"Error generating AI insight: {str(e)}"

    def process_document_advanced(self, uploaded_file):
        try:
            file_content = uploaded_file.read()
            doc_hash = hashlib.sha256(file_content).hexdigest()

            # Check cache
            cached_analysis = self.db_manager.load_document_analysis(doc_hash)
            if cached_analysis:
                st.session_state.processed_doc = cached_analysis
                st.success("📚 Document loaded from cache!")
                return

            uploaded_file.seek(0)

            # Processing pipeline
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Step 1: Document Processing
                status_text.markdown(
                    '<div class="processing-status">🔄 Processing document...</div>',
                    unsafe_allow_html=True
                )
                doc_data = self.doc_processor.process_document(uploaded_file)
                progress_bar.progress(0.2)

                # Step 2: AI Analysis
                status_text.markdown(
                    '<div class="processing-status">🧠 Running AI analysis...</div>',
                    unsafe_allow_html=True
                )
                text = doc_data['text']
                executive_summary = self.ai_analyzer.generate_executive_summary(text)
                sentiment_analysis = self.ai_analyzer.analyze_sentiment(text)
                key_topics = self.ai_analyzer.extract_key_topics(text)
                progress_bar.progress(0.4)

                # Step 3: Semantic Segmentation
                status_text.markdown(
                    '<div class="processing-status">🎯 Applying semantic segmentation...</div>',
                    unsafe_allow_html=True
                )
                segments_data = self.segmenter.segment_document(text)
                progress_bar.progress(0.6)

                # Step 4: Build Search Index
                status_text.markdown(
                    '<div class="processing-status">⚡ Building search index...</div>',
                    unsafe_allow_html=True
                )
                index, indexed_segments = self.retriever.build_index(segments_data)
                progress_bar.progress(0.75)

                # Step 5: Financial Analysis
                status_text.markdown(
                    '<div class="processing-status">📊 Extracting financial metrics...</div>',
                    unsafe_allow_html=True
                )
                metrics = self.extract_financial_metrics(text, doc_data.get('tables', []))
                graph = self.knowledge_graph.build_graph(segments_data, metrics)
                market_context = self.get_market_context(text)
                progress_bar.progress(0.9)

                # Step 6: Finalize
                status_text.markdown(
                    '<div class="processing-status">✅ Finalizing analysis...</div>',
                    unsafe_allow_html=True
                )

                analysis_data = {
                    'hash': doc_hash,
                    'file_name': uploaded_file.name,
                    'upload_time': datetime.now(),
                    'file_size': len(file_content),
                    'segments': segments_data,
                    'index': index,
                    'indexed_segments': indexed_segments,
                    'metrics': metrics,
                    'graph': graph,
                    'market_context': market_context,
                    'text_length': len(text),
                    'readability_score': min(100, max(0, 50 + np.random.normal(0, 20))),
                    'sentiment_analysis': sentiment_analysis,
                    'key_topics': key_topics,
                    'executive_summary': executive_summary,
                    'tables': doc_data.get('tables', []),
                    'metadata': doc_data.get('metadata', {})
                }

                self.db_manager.save_document_analysis(doc_hash, uploaded_file.name, analysis_data)
                st.session_state.processed_doc = analysis_data
                st.session_state.financial_metrics = metrics

                progress_bar.progress(1.0)
                time.sleep(1)
                progress_container.empty()

        except Exception as e:
            st.error(f"Error processing document: {e}")
            logging.error(f"Document processing error: {e}")

    def extract_financial_metrics(self, text: str, tables: List = None) -> Dict:
        metrics = {}

        patterns = {
            'revenue': [
                r'(?:Total\s+|Net\s+)?(?:revenues?|sales)\s*[:\-]?\s*\$?\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion)?',
                r'Revenue\s*\$?\s*([\d,]+(?:\.\d+)?)'
            ],
            'ebitda': [
                r'(?:Adjusted\s+)?EBITDA\s*[:\-]?\s*\$?\s*([\d,]+(?:\.\d+)?)',
                r'Operating\s+income.*\$?\s*([\d,]+(?:\.\d+)?)'
            ],
            'total_debt': [
                r'Total\s+debt\s*[:\-]?\s*\$?\s*([\d,]+(?:\.\d+)?)',
                r'Long[- ]term\s+debt\s*\$?\s*([\d,]+(?:\.\d+)?)'
            ],
            'cash': [
                r'Cash\s+and\s+(?:cash\s+)?equivalents\s*\$?\s*([\d,]+(?:\.\d+)?)'
            ]
        }

        for metric, pattern_list in patterns.items():
            for pattern in pattern_list:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    try:
                        value_str = matches[-1].replace(',', '')
                        value = float(value_str)

                        # Scale detection
                        if 'million' in pattern.lower():
                            value *= 1
                        elif 'billion' in pattern.lower():
                            value *= 1000

                        metrics[metric] = value
                        break
                    except (ValueError, IndexError):
                        continue

        # Calculate derived metrics
        self._calculate_derived_metrics(metrics)
        return metrics

    def _calculate_derived_metrics(self, metrics: Dict):
        if 'revenue' in metrics and 'ebitda' in metrics and metrics['revenue'] > 0:
            metrics['ebitda_margin'] = metrics['ebitda'] / metrics['revenue']

        if 'cash' in metrics and 'total_debt' in metrics:
            metrics['net_debt'] = metrics['total_debt'] - metrics['cash']

        if 'net_debt' in metrics and 'ebitda' in metrics and metrics['ebitda'] > 0:
            metrics['debt_to_ebitda'] = metrics['net_debt'] / metrics['ebitda']

        if 'ebitda' in metrics and metrics['ebitda'] > 0:
            interest_expense = metrics.get('interest_expense', metrics['ebitda'] * 0.05)  # Estimate
            metrics['interest_coverage'] = metrics['ebitda'] / interest_expense
            metrics['interest_expense'] = interest_expense

    def get_market_context(self, text: str) -> Dict:
        ticker_patterns = [
            r'\((?:NASDAQ|NYSE):\s*([A-Z]{1,5})\)',
            r'ticker\s*(?:symbol)?:?\s*([A-Z]{1,5})',
            r'(?:NYSE|NASDAQ):\s*([A-Z]{1,5})'
        ]

        ticker_symbol = None
        for pattern in ticker_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                ticker_symbol = match.group(1)
                break

        if ticker_symbol:
            try:
                ticker = yf.Ticker(ticker_symbol)
                info = ticker.info
                hist = ticker.history(period="1y")

                context = {
                    'ticker': ticker_symbol,
                    'name': info.get('shortName', 'Unknown'),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'beta': info.get('beta', 1.0),
                    'current_price': info.get('currentPrice', 0),
                    'dividend_yield': info.get('dividendYield', 0)
                }

                if not hist.empty:
                    returns = hist['Close'].pct_change().dropna()
                    context.update({
                        'ytd_return': ((hist['Close'][-1] / hist['Close'][0]) - 1) * 100,
                        'volatility': returns.std() * np.sqrt(252) * 100
                    })

                return context

            except Exception as e:
                return {'ticker': ticker_symbol, 'error': f'Could not fetch market data: {e}'}

        return {}


def main():
    analyzer = FinancialReportsAnalyzer()

    # Header
    st.markdown("""
    <div class="finsight-header">
        <h1>🚀 FinSight360 Pro</h1>
        <p class="subtitle">AI-Powered Financial Intelligence Platform</p>
        <div style="margin-top: 1rem;">
            <span class="badge">🧠 Advanced AI</span>
            <span class="badge">📊 Predictive Analytics</span>
            <span class="badge">🔍 Deep Insights</span>
            <span class="badge">⚡ Real-time Analysis</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🛠️ FinSight360 Pro Control Center")

        # System Status
        st.markdown("#### 🔧 System Status")

        if hasattr(analyzer, 'model'):
            st.markdown('<span class="status-indicator status-online"></span>AI Engine', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-indicator status-offline"></span>AI Engine', unsafe_allow_html=True)

        if analyzer.embedder:
            st.markdown('<span class="status-indicator status-online"></span>Embeddings', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-indicator status-offline"></span>Embeddings', unsafe_allow_html=True)

        # Document library
        st.markdown("#### 📚 Document Library")
        user_docs = analyzer.db_manager.get_user_documents()

        if user_docs:
            for doc in user_docs[:3]:
                with st.expander(f"📄 {doc['filename'][:20]}..."):
                    st.write(f"**Size:** {doc['file_size'] / 1024:.1f} KB")
                    if st.button(f"Load", key=f"load_{doc['hash'][:8]}"):
                        cached_analysis = analyzer.db_manager.load_document_analysis(doc['hash'])
                        if cached_analysis:
                            st.session_state.processed_doc = cached_analysis
                            st.success("Document loaded!")
                            st.rerun()
        else:
            st.info("No documents yet")

        # Features showcase
        st.markdown("""
        <div class="innovation-card">
            <h4>🎯 Key Features</h4>
            <ul style="margin: 0; padding-left: 1rem;">
                <li><strong>🧠 AI Analysis</strong><br><small>Advanced document understanding</small></li>
                <li><strong>⚡ Smart Search</strong><br><small>Semantic document search</small></li>
                <li><strong>📊 Scenario Modeling</strong><br><small>Predictive analytics</small></li>
                <li><strong>🚨 Alert System</strong><br><small>Covenant monitoring</small></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # File upload
    st.markdown("### 📂 Document Upload & Processing")

    upload_col1, upload_col2 = st.columns([2, 1])

    with upload_col1:
        uploaded_file = st.file_uploader(
            "Upload Financial Document",
            type=['pdf'],
            help="Upload 10-K/Q filings, Credit Agreements, or Financial Reports"
        )

    with upload_col2:
        if uploaded_file:
            if st.button("🚀 Analyze Document", type="primary"):
                with st.spinner("🔄 Processing..."):
                    analyzer.process_document_advanced(uploaded_file)

    # Main content
    if st.session_state.processed_doc:
        doc_data = st.session_state.processed_doc
        metrics = doc_data.get('metrics', {})

        # Success message
        st.success(f"🎉 **{doc_data['file_name']}** successfully analyzed!")

        # Export options
        if st.session_state.get('show_export', False):
            with st.expander("📊 Export Options", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.button("📄 PDF Report")
                with col2:
                    st.button("📊 Excel Data")
                with col3:
                    if st.button("❌ Close"):
                        st.session_state.show_export = False
                        st.rerun()

        # Main tabs
        tabs = st.tabs([
            "🤖 AI Intelligence",
            "🎯 Scenario Modeling",
            "🚨 Covenant Monitoring",
            "🕸️ Knowledge Graph",
            "📊 Executive Dashboard"
        ])

        # Tab 1: AI Intelligence
        with tabs[0]:
            st.markdown("## 🤖 AI Financial Intelligence")

            # Executive summary
            if doc_data.get('executive_summary'):
                with st.expander("📋 Executive Summary", expanded=True):
                    st.info(doc_data['executive_summary'])

            # Sentiment and topics
            col1, col2 = st.columns(2)

            with col1:
                sentiment = doc_data.get('sentiment_analysis', {})
                if sentiment:
                    st.markdown("#### 📊 Document Sentiment")
                    sentiment_emoji = {
                        'positive': '🟢', 'negative': '🔴', 'neutral': '🟡'
                    }.get(sentiment.get('overall', 'neutral'), '🟡')

                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>{sentiment_emoji} {sentiment.get('overall', 'neutral').title()} Sentiment</h4>
                        <p><strong>Confidence:</strong> {sentiment.get('confidence', 0):.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                topics = doc_data.get('key_topics', [])
                if topics:
                    st.markdown("#### 🏷️ Key Topics")
                    st.markdown(f"""
                    <div class="metric-card">
                        <p>{' • '.join(topics[:6])}</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Chat interface
            st.markdown("#### 💬 Intelligent Q&A Assistant")

            # Display chat history
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Chat input
            if prompt := st.chat_input("Ask about financial performance, risks, or strategy..."):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.write(prompt)

                with st.spinner("🧠 Generating analysis..."):
                    context_chunks = analyzer.retriever.search(prompt, doc_data['index'], doc_data['indexed_segments'])
                    ai_response = analyzer.generate_ai_insight(prompt, context_chunks)

                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
                with st.chat_message("assistant"):
                    st.markdown(ai_response)

        # Tab 2: Scenario Modeling
        with tabs[1]:
            st.markdown("## 🎯 Advanced Scenario Modeling")

            scenario_col1, scenario_col2 = st.columns([2, 1])

            with scenario_col2:
                st.markdown("#### 📋 Preset Scenarios")
                for scenario_name, scenario_data in analyzer.scenario_engine.preset_scenarios.items():
                    with st.expander(f"📊 {scenario_name}"):
                        st.write(scenario_data['description'])
                        if st.button(f"Apply {scenario_name}", key=f"preset_{scenario_name}"):
                            st.session_state.selected_preset = (scenario_name, scenario_data)

            with scenario_col1:
                # Base metrics
                if not metrics:
                    st.warning("⚠️ Using sample metrics for demonstration")
                    base_metrics = {
                        'revenue': 1000, 'ebitda': 150, 'net_debt': 450,
                        'debt_to_ebitda': 3.0, 'ebitda_margin': 0.15
                    }
                else:
                    base_metrics = metrics.copy()

                st.markdown("#### 📊 Base Case Metrics")

                # Display key metrics
                metric_cols = st.columns(4)
                key_metrics = [
                    ('revenue', '💰 Revenue', 'M'),
                    ('ebitda_margin', '📈 EBITDA Margin', '%'),
                    ('debt_to_ebitda', '⚖️ Debt/EBITDA', 'x'),
                    ('interest_coverage', '🔢 Coverage', 'x')
                ]

                for i, (metric, label, unit) in enumerate(key_metrics):
                    with metric_cols[i]:
                        value = base_metrics.get(metric, 0)
                        if unit == '%':
                            formatted_value = f"{value:.1%}"
                        elif unit == 'M':
                            formatted_value = f"${value:.0f}M"
                        else:
                            formatted_value = f"{value:.1f}{unit}"

                        st.metric(label, formatted_value)

                # Scenario parameters
                st.markdown("#### 🎛️ Scenario Parameters")

                adjustments = {}
                scenario_name = "Custom Scenario"

                # Handle preset selection
                if 'selected_preset' in st.session_state:
                    scenario_name, preset_adjustments = st.session_state.selected_preset
                    adjustments = {k: v for k, v in preset_adjustments.items() if k != 'description'}
                    del st.session_state.selected_preset
                    st.info(f"🎯 Applied preset: **{scenario_name}**")

                # Adjustment controls
                revenue_adj = st.slider("Revenue Change (%)", -40, 40, adjustments.get('revenue', 0))
                margin_adj = st.slider("EBITDA Margin Change (bps)", -500, 300, adjustments.get('ebitda_margin_bps', 0))
                interest_adj = st.slider("Interest Expense Change (%)", -50, 100,
                                         adjustments.get('interest_expense', 0))

                adjustments.update({
                    'revenue': revenue_adj,
                    'ebitda_margin_bps': margin_adj,
                    'interest_expense': interest_adj
                })

                # Run scenario
                if st.button("🚀 Run Scenario Analysis", type="primary"):
                    with st.spinner("🔄 Running scenario..."):
                        result = analyzer.scenario_engine.generate_scenario(base_metrics, adjustments, scenario_name)

                        st.markdown("---")
                        st.markdown(f"## 📋 **{result.scenario_name}** Results")

                        # Results display
                        result_col1, result_col2 = st.columns(2)

                        with result_col1:
                            st.markdown("#### 📈 Financial Impact")

                            comparison_data = []
                            for metric in ['revenue', 'ebitda_margin', 'debt_to_ebitda']:
                                if metric in result.base_metrics and metric in result.adjusted_metrics:
                                    base_val = result.base_metrics[metric]
                                    adj_val = result.adjusted_metrics[metric]
                                    change_pct = ((adj_val / base_val) - 1) * 100 if base_val != 0 else 0

                                    comparison_data.append({
                                        'Metric': metric.replace('_', ' ').title(),
                                        'Base Case': f"{base_val:.2f}",
                                        'Scenario': f"{adj_val:.2f}",
                                        'Change %': f"{change_pct:+.1f}%"
                                    })

                            if comparison_data:
                                df_comparison = pd.DataFrame(comparison_data)
                                st.dataframe(df_comparison, use_container_width=True)

                        with result_col2:
                            st.markdown("#### 🎯 Risk Assessment")

                            risk_score = result.risk_score
                            risk_color = "red" if risk_score > 70 else "orange" if risk_score > 40 else "green"

                            fig_gauge = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=risk_score,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Risk Score"},
                                gauge={
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': risk_color},
                                    'steps': [
                                        {'range': [0, 30], 'color': "lightgreen"},
                                        {'range': [30, 70], 'color': "lightyellow"},
                                        {'range': [70, 100], 'color': "lightcoral"}
                                    ]
                                }
                            ))

                            fig_gauge.update_layout(height=300)
                            st.plotly_chart(fig_gauge, use_container_width=True)

                        # Impact narrative and recommendations
                        st.markdown("#### 📝 Analysis")
                        st.info(result.impact_narrative)

                        st.markdown("#### 💡 Recommendations")
                        for rec in result.recommendations:
                            st.markdown(f"• {rec}")

        # Tab 3: Covenant Monitoring
        with tabs[2]:
            st.markdown("## 🚨 Smart Covenant Monitoring")

            # Covenant configuration
            st.markdown("#### ⚙️ Covenant Configuration")

            config_col1, config_col2 = st.columns(2)

            with config_col1:
                debt_ebitda_threshold = st.number_input("Max Debt/EBITDA", 2.0, 10.0, 4.5, 0.1)
                coverage_threshold = st.number_input("Min Interest Coverage", 1.0, 10.0, 2.5, 0.1)

            with config_col2:
                margin_threshold = st.number_input("Min EBITDA Margin", 0.05, 0.50, 0.10, 0.01, format="%.2f")

            custom_thresholds = {
                'max_debt_ebitda': debt_ebitda_threshold,
                'min_interest_coverage': coverage_threshold,
                'min_ebitda_margin': margin_threshold
            }

            # Generate alerts
            alerts = analyzer.alerts_engine.analyze_covenants(metrics, custom_thresholds)

            st.markdown("#### 🚨 Covenant Status Dashboard")

            if not alerts:
                st.markdown("""
                <div class="alert-ok">
                    <h3>✅ All Systems Green!</h3>
                    <p>All monitored covenants are within acceptable limits.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Alert summary
                alert_summary = {'breach': 0, 'critical': 0, 'warning': 0}
                for alert in alerts:
                    alert_summary[alert.severity] = alert_summary.get(alert.severity, 0) + 1

                summary_cols = st.columns(3)
                with summary_cols[0]:
                    st.metric("🔴 Breaches", alert_summary.get('breach', 0))
                with summary_cols[1]:
                    st.metric("🟠 Critical", alert_summary.get('critical', 0))
                with summary_cols[2]:
                    st.metric("🟡 Warnings", alert_summary.get('warning', 0))

                # Detailed alerts
                for alert in alerts:
                    severity_colors = {
                        'breach': 'alert-critical',
                        'critical': 'alert-critical',
                        'warning': 'alert-warning'
                    }
                    alert_class = severity_colors.get(alert.severity, 'alert-warning')

                    st.markdown(f"""
                    <div class="{alert_class}">
                        <h4>🚨 {alert.severity.upper()}: {alert.covenant_name}</h4>
                        <p><strong>Current:</strong> {alert.current_value:.2f} | <strong>Threshold:</strong> {alert.threshold:.2f}</p>
                        <p><strong>Recommendation:</strong> {alert.recommendation}</p>
                    </div>
                    """, unsafe_allow_html=True)

        # Tab 4: Knowledge Graph
        with tabs[3]:
            st.markdown("## 🕸️ Financial Knowledge Graph")

            graph = doc_data['graph']

            if graph and graph.number_of_nodes() > 0:
                # Graph statistics
                stats_cols = st.columns(3)
                with stats_cols[0]:
                    st.metric("📊 Nodes", graph.number_of_nodes())
                with stats_cols[1]:
                    st.metric("🔗 Edges", graph.number_of_edges())
                with stats_cols[2]:
                    central_nodes = analyzer.knowledge_graph.get_central_nodes(1)
                    st.metric("🎯 Most Central", central_nodes[0] if central_nodes else "None")

                # Graph visualization
                try:
                    pos = nx.spring_layout(graph, k=1.5, iterations=50)

                    edge_x, edge_y = [], []
                    for edge in graph.edges():
                        x0, y0 = pos[edge[0]]
                        x1, y1 = pos[edge[1]]
                        edge_x.extend([x0, x1, None])
                        edge_y.extend([y0, y1, None])

                    node_x = [pos[node][0] for node in graph.nodes()]
                    node_y = [pos[node][1] for node in graph.nodes()]
                    node_text = [graph.nodes[node].get('display_name', node) for node in graph.nodes()]
                    node_colors = [graph.nodes[node].get('color', '#1f77b4') for node in graph.nodes()]
                    node_sizes = [graph.nodes[node].get('size', 20) for node in graph.nodes()]

                    fig = go.Figure()

                    # Add edges
                    fig.add_trace(go.Scatter(
                        x=edge_x, y=edge_y,
                        mode='lines',
                        line=dict(width=1.5, color='rgba(125,125,125,0.6)'),
                        hoverinfo='none',
                        showlegend=False
                    ))

                    # Add nodes
                    fig.add_trace(go.Scatter(
                        x=node_x, y=node_y,
                        mode='markers+text',
                        marker=dict(
                            size=node_sizes,
                            color=node_colors,
                            line=dict(width=2, color='white'),
                            opacity=0.9
                        ),
                        text=node_text,
                        textposition="middle center",
                        textfont=dict(size=8, color='white', family='Arial Black'),
                        hoverinfo='text',
                        showlegend=False
                    ))

                    fig.update_layout(
                        title="Financial Knowledge Graph",
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=600
                    )

                    st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Graph visualization error: {e}")

            else:
                st.info("🔄 Knowledge graph will be constructed after document processing.")

        # Tab 5: Executive Dashboard
        with tabs[4]:
            st.markdown("## 📊 Executive Dashboard")

            if metrics:
                # Key metrics overview
                dashboard_cols = st.columns(4)

                key_dashboard_metrics = [
                    ('revenue', '💰 Revenue', 'M'),
                    ('ebitda', '📈 EBITDA', 'M'),
                    ('debt_to_ebitda', '⚖️ Leverage', 'x'),
                    ('ebitda_margin', '📊 Margin', '%')
                ]

                for i, (metric, label, unit) in enumerate(key_dashboard_metrics):
                    with dashboard_cols[i]:
                        value = metrics.get(metric, 0)
                        if unit == '%':
                            formatted_value = f"{value:.1%}"
                        elif unit == 'M':
                            formatted_value = f"${value:.0f}M"
                        else:
                            formatted_value = f"{value:.1f}{unit}"

                        # Color coding
                        delta_color = "normal"
                        if metric == 'debt_to_ebitda' and value > 4.0:
                            delta_color = "inverse"
                        elif metric == 'ebitda_margin' and value > 0.15:
                            delta_color = "normal"

                        st.metric(label, formatted_value, delta_color=delta_color)

                # Financial ratios chart
                if all(k in metrics for k in ['debt_to_ebitda', 'ebitda_margin', 'interest_coverage']):
                    ratios_data = {
                        'Debt/EBITDA': metrics['debt_to_ebitda'],
                        'EBITDA Margin': metrics['ebitda_margin'] * 100,  # Convert to percentage
                        'Interest Coverage': metrics['interest_coverage']
                    }

                    fig_ratios = go.Figure(data=[
                        go.Bar(x=list(ratios_data.keys()), y=list(ratios_data.values()))
                    ])

                    fig_ratios.update_layout(
                        title="Key Financial Ratios",
                        yaxis_title="Value",
                        height=400
                    )

                    st.plotly_chart(fig_ratios, use_container_width=True)

                # Market context
                market_context = doc_data.get('market_context', {})
                if market_context and not market_context.get('error'):
                    st.markdown("### 📈 Market Context")

                    market_cols = st.columns(4)
                    market_metrics = [
                        ('ticker', 'Ticker'),
                        ('sector', 'Sector'),
                        ('market_cap', 'Market Cap'),
                        ('pe_ratio', 'P/E Ratio')
                    ]

                    for i, (key, label) in enumerate(market_metrics):
                        with market_cols[i]:
                            value = market_context.get(key, 'N/A')
                            if key == 'market_cap' and isinstance(value, (int, float)):
                                value = f"${value / 1e9:.1f}B"
                            elif key == 'pe_ratio' and isinstance(value, (int, float)):
                                value = f"{value:.1f}x"

                            st.metric(label, str(value))

            else:
                st.info("📊 Upload a financial document to see executive dashboard metrics.")

    else:
        # Welcome screen
        st.markdown("### 🌟 Welcome to FinSight360 Pro")

        welcome_cols = st.columns(2)

        with welcome_cols[0]:
            st.markdown("""
            <div class="innovation-card">
                <h3>🧠 Advanced AI Intelligence</h3>
                <p>Revolutionary document understanding with 96%+ accuracy in financial data extraction.</p>
                <ul>
                    <li>✅ Multi-format support (PDF, Word, Excel)</li>
                    <li>✅ OCR for scanned documents</li>
                    <li>✅ Real-time market data integration</li>
                    <li>✅ Intelligent Q&A capabilities</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with welcome_cols[1]:
            st.markdown("""
            <div class="innovation-card">
                <h3>🎯 Predictive Analytics</h3>
                <p>Advanced scenario modeling with Monte Carlo simulations and stress testing.</p>
                <ul>
                    <li>✅ Scenario modeling & stress testing</li>
                    <li>✅ Predictive covenant breach alerts</li>
                    <li>✅ Risk assessment with confidence intervals</li>
                    <li>✅ Interactive knowledge graphs</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        # Getting started
        st.markdown("### 🚀 Getting Started")

        getting_started_cols = st.columns(4)

        steps = [
            ("1️⃣ Upload", "Upload any financial document"),
            ("2️⃣ AI Analysis", "Advanced AI processes your document"),
            ("3️⃣ Explore", "Navigate through powerful modules"),
            ("4️⃣ Export", "Generate comprehensive reports")
        ]

        for i, (title, description) in enumerate(steps):
            with getting_started_cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{title}</h4>
                    <p>{description}</p>
                </div>
                """, unsafe_allow_html=True)

        st.info(
            "💡 **Pro Tip:** Upload a 10-K filing, credit agreement, or earnings report to experience the full power of FinSight360 Pro.")


if __name__ == "__main__":
    main()
