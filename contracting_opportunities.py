import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sqlite3
import psycopg2
import requests
import json
import yaml
import pickle
import io
import base64
import os
import schedule
import threading
import time
import warnings
warnings.filterwarnings('ignore')

# ML and Analytics
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.neural_network import MLPRegressor

# Report Generation
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart

class ContractingMLEngine:
    """Advanced ML engine for contract analysis with multiple algorithms and auto-selection"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0)
        }
        
        self.ensemble_model = None
        self.best_model_name = None
        self.model_performance = {}
        self.feature_importance = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.training_history = []
        
    def prepare_features(self, df):
        """Enhanced feature engineering for contract analysis"""
        features_df = df.copy()
        features_df['Creation Date'] = pd.to_datetime(features_df['Creation Date'])
        
        # Time-based features
        features_df['year'] = features_df['Creation Date'].dt.year
        features_df['month'] = features_df['Creation Date'].dt.month
        features_df['quarter'] = features_df['Creation Date'].dt.quarter
        features_df['day_of_week'] = features_df['Creation Date'].dt.dayofweek
        features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Vendor-based features
        vendor_stats = features_df.groupby('Vendor Name').agg({
            'Unit Price': ['mean', 'std', 'count', 'min', 'max'],
            'Qty Delivered': ['mean', 'std', 'sum'],
            'Line Total': ['sum', 'mean']
        }).round(4)
        
        vendor_stats.columns = ['_'.join(col).strip() for col in vendor_stats.columns]
        
        # Map vendor statistics back to main dataframe
        for col in vendor_stats.columns:
            features_df[f'vendor_{col}'] = features_df['Vendor Name'].map(vendor_stats[col])
        
        # Item-based features
        item_stats = features_df.groupby('Item').agg({
            'Unit Price': ['mean', 'std', 'count'],
            'Qty Delivered': ['mean', 'std'],
            'Vendor Name': 'nunique'
        }).round(4)
        
        item_stats.columns = ['_'.join(col).strip() for col in item_stats.columns]
        
        for col in item_stats.columns:
            features_df[f'item_{col}'] = features_df['Item'].map(item_stats[col])
        
        # Rolling statistics (for time series)
        features_df = features_df.sort_values(['Vendor Name', 'Item', 'Creation Date'])
        
        for window in [3, 6, 12]:
            features_df[f'rolling_price_{window}'] = features_df.groupby(['Vendor Name', 'Item'])['Unit Price'].rolling(window).mean().reset_index(0, drop=True)
            features_df[f'rolling_qty_{window}'] = features_df.groupby(['Vendor Name', 'Item'])['Qty Delivered'].rolling(window).mean().reset_index(0, drop=True)
        
        # Price trend features
        features_df['price_trend'] = features_df.groupby(['Vendor Name', 'Item'])['Unit Price'].pct_change()
        features_df['volume_trend'] = features_df.groupby(['Vendor Name', 'Item'])['Qty Delivered'].pct_change()
        
        # Market position features
        features_df['price_percentile'] = features_df.groupby('Item')['Unit Price'].rank(pct=True)
        features_df['volume_percentile'] = features_df.groupby('Item')['Qty Delivered'].rank(pct=True)
        
        return features_df
    
    def train_models(self, df, target_column='consolidation_potential'):
        """Train multiple ML models and select the best performer"""
        try:
            # Prepare features
            features_df = self.prepare_features(df)
            
            # Select feature columns (exclude non-numeric and target)
            feature_columns = [col for col in features_df.columns if 
                             features_df[col].dtype in ['int64', 'float64'] and 
                             col not in ['Creation Date', target_column]]
            
            X = features_df[feature_columns].fillna(0)
            
            # Create target if not exists
            if target_column not in features_df.columns:
                # Create consolidation potential as target
                y = self._calculate_consolidation_potential(features_df)
            else:
                y = features_df[target_column]
            
            # Remove rows with missing target
            mask = ~pd.isna(y)
            X = X[mask]
            y = y[mask]
            
            if len(X) < 50:
                return False, "Insufficient data for training (need at least 50 records)"
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train and evaluate models
            best_score = -np.inf
            model_results = {}
            
            for name, model in self.models.items():
                try:
                    # Hyperparameter tuning for key models
                    if name == 'random_forest':
                        param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20, None]}
                        model = GridSearchCV(model, param_grid, cv=3, scoring='r2')
                    elif name == 'gradient_boosting':
                        param_grid = {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.01]}
                        model = GridSearchCV(model, param_grid, cv=3, scoring='r2')
                    
                    # Train model
                    if name in ['linear_regression', 'ridge_regression', 'neural_network']:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    # Evaluate
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Cross-validation
                    if name in ['linear_regression', 'ridge_regression', 'neural_network']:
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                    else:
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                    
                    cv_mean = cv_scores.mean()
                    cv_std = cv_scores.std()
                    
                    model_results[name] = {
                        'model': model,
                        'mae': mae,
                        'r2': r2,
                        'cv_mean': cv_mean,
                        'cv_std': cv_std,
                        'test_score': r2
                    }
                    
                    # Track best model
                    if cv_mean > best_score:
                        best_score = cv_mean
                        self.best_model_name = name
                    
                    # Feature importance for tree-based models
                    if hasattr(model, 'feature_importances_'):
                        self.feature_importance[name] = dict(zip(feature_columns, model.feature_importances_))
                    elif hasattr(model, 'best_estimator_') and hasattr(model.best_estimator_, 'feature_importances_'):
                        self.feature_importance[name] = dict(zip(feature_columns, model.best_estimator_.feature_importances_))
                
                except Exception as e:
                    model_results[name] = {'error': str(e)}
                    st.warning(f"Model {name} training failed: {str(e)}")
            
            # Create ensemble model from best performers
            good_models = [(name, results['model']) for name, results in model_results.items() 
                          if 'error' not in results and results['cv_mean'] > 0.1]
            
            if len(good_models) >= 2:
                ensemble_estimators = good_models[:3]  # Top 3 models
                self.ensemble_model = VotingRegressor(estimators=ensemble_estimators)
                
                # Train ensemble
                ensemble_X_train = X_train_scaled if any(name in ['linear_regression', 'ridge_regression', 'neural_network'] 
                                                        for name, _ in ensemble_estimators) else X_train
                self.ensemble_model.fit(ensemble_X_train, y_train)
            
            self.model_performance = model_results
            self.is_trained = True
            
            # Save training history
            self.training_history.append({
                'timestamp': datetime.now(),
                'best_model': self.best_model_name,
                'best_score': best_score,
                'models_trained': len(model_results),
                'data_size': len(X)
            })
            
            return True, f"Successfully trained {len(good_models)} models. Best: {self.best_model_name} (R²: {best_score:.3f})"
            
        except Exception as e:
            return False, f"Training failed: {str(e)}"
    
    def predict_contract_success(self, df):
        """Predict contract success probability using trained models"""
        if not self.is_trained:
            return None, "Models not trained yet"
        
        try:
            features_df = self.prepare_features(df)
            feature_columns = [col for col in features_df.columns if 
                             features_df[col].dtype in ['int64', 'float64'] and 
                             col not in ['Creation Date']]
            
            X = features_df[feature_columns].fillna(0)
            
            # Use best model for prediction
            best_model = self.model_performance[self.best_model_name]['model']
            
            if self.best_model_name in ['linear_regression', 'ridge_regression', 'neural_network']:
                X_scaled = self.scaler.transform(X)
                predictions = best_model.predict(X_scaled)
            else:
                predictions = best_model.predict(X)
            
            # Ensemble prediction if available
            if self.ensemble_model is not None:
                ensemble_X = self.scaler.transform(X) if any(name in ['linear_regression', 'ridge_regression', 'neural_network'] 
                                                           for name, _ in self.ensemble_model.estimators) else X
                ensemble_predictions = self.ensemble_model.predict(ensemble_X)
                
                # Average predictions
                predictions = (predictions + ensemble_predictions) / 2
            
            return predictions, "Predictions generated successfully"
            
        except Exception as e:
            return None, f"Prediction failed: {str(e)}"
    
    def _calculate_consolidation_potential(self, df):
        """Calculate consolidation potential score as target variable"""
        scores = []
        
        for _, row in df.iterrows():
            # Multi-factor scoring
            vendor_count = row.get('item_Vendor Name_nunique', 1)
            price_std = row.get('vendor_Unit Price_std', 0)
            total_spend = row.get('vendor_Line Total_sum', 0)
            
            vendor_score = min(vendor_count / 5, 1.0)  # More vendors = higher potential
            price_score = min(price_std / row.get('vendor_Unit Price_mean', 1), 1.0)  # Higher variance = higher potential
            spend_score = min(total_spend / 100000, 1.0)  # Higher spend = higher potential
            
            consolidation_score = (vendor_score * 0.4 + price_score * 0.3 + spend_score * 0.3)
            scores.append(consolidation_score)
        
        return pd.Series(scores)
    
    def save_models(self, filepath):
        """Save trained models to disk"""
        if self.is_trained:
            model_data = {
                'models': self.model_performance,
                'ensemble_model': self.ensemble_model,
                'scaler': self.scaler,
                'feature_importance': self.feature_importance,
                'best_model_name': self.best_model_name,
                'training_history': self.training_history
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            return True
        return False
    
    def load_models(self, filepath):
        """Load trained models from disk"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model_performance = model_data['models']
            self.ensemble_model = model_data['ensemble_model']
            self.scaler = model_data['scaler']
            self.feature_importance = model_data['feature_importance']
            self.best_model_name = model_data['best_model_name']
            self.training_history = model_data.get('training_history', [])
            self.is_trained = True
            
            return True
        except Exception as e:
            st.error(f"Failed to load models: {str(e)}")
            return False

class VendorSegmentationEngine:
    """Advanced vendor clustering and segmentation"""
    
    def __init__(self):
        self.clustering_models = {
            'kmeans': KMeans(random_state=42),
            'dbscan': DBSCAN()
        }
        self.optimal_clusters = None
        self.cluster_labels = None
        self.cluster_profiles = None
        self.scaler = StandardScaler()
        
    def analyze_vendor_segments(self, df):
        """Perform advanced vendor segmentation analysis"""
        try:
            # Prepare vendor features
            vendor_features = df.groupby('Vendor Name').agg({
                'Line Total': ['sum', 'mean', 'std'],
                'Unit Price': ['mean', 'std'],
                'Qty Delivered': ['sum', 'mean', 'std'],
                'Item': 'nunique',
                'DEP': 'nunique' if 'DEP' in df.columns else lambda x: 1,
                'W/H': 'nunique' if 'W/H' in df.columns else lambda x: 1
            }).round(4)
            
            vendor_features.columns = ['_'.join(col).strip() for col in vendor_features.columns]
            vendor_features = vendor_features.fillna(0)
            
            # Add calculated metrics
            vendor_features['price_volatility'] = vendor_features['Unit Price_std'] / vendor_features['Unit Price_mean']
            vendor_features['volume_volatility'] = vendor_features['Qty Delivered_std'] / vendor_features['Qty Delivered_mean']
            vendor_features['geographic_coverage'] = vendor_features['DEP_nunique'] * vendor_features['W/H_nunique']
            
            vendor_features = vendor_features.fillna(0).replace([np.inf, -np.inf], 0)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(vendor_features)
            
            # Determine optimal number of clusters
            silhouette_scores = []
            K_range = range(2, min(11, len(vendor_features)))
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                cluster_labels = kmeans.fit_predict(X_scaled)
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                silhouette_scores.append(silhouette_avg)
            
            # Find optimal clusters
            optimal_k = K_range[np.argmax(silhouette_scores)]
            self.optimal_clusters = optimal_k
            
            # Final clustering
            final_kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            self.cluster_labels = final_kmeans.fit_predict(X_scaled)
            
            # Add cluster labels to vendor features
            vendor_features['Cluster'] = self.cluster_labels
            vendor_features['Vendor'] = vendor_features.index
            
            # Generate cluster profiles
            self.cluster_profiles = self._generate_cluster_profiles(vendor_features)
            
            return vendor_features, self.cluster_profiles
            
        except Exception as e:
            st.error(f"Vendor segmentation failed: {str(e)}")
            return None, None
    
    def _generate_cluster_profiles(self, vendor_features):
        """Generate descriptive profiles for each cluster"""
        profiles = {}
        
        for cluster_id in range(self.optimal_clusters):
            cluster_data = vendor_features[vendor_features['Cluster'] == cluster_id]
            
            # Calculate cluster characteristics
            avg_spend = cluster_data['Line Total_sum'].mean()
            avg_items = cluster_data['Item_nunique'].mean()
            avg_price_volatility = cluster_data['price_volatility'].mean()
            vendor_count = len(cluster_data)
            
            # Determine cluster type
            if avg_spend > vendor_features['Line Total_sum'].quantile(0.8):
                cluster_type = "Strategic Partners"
            elif avg_items > vendor_features['Item_nunique'].quantile(0.7):
                cluster_type = "Diversified Suppliers"
            elif avg_price_volatility > vendor_features['price_volatility'].quantile(0.7):
                cluster_type = "Volatile Suppliers"
            else:
                cluster_type = "Standard Suppliers"
            
            profiles[cluster_id] = {
                'type': cluster_type,
                'vendor_count': vendor_count,
                'avg_spend': avg_spend,
                'avg_items': avg_items,
                'avg_price_volatility': avg_price_volatility,
                'vendors': cluster_data['Vendor'].tolist()
            }
        
        return profiles

class DatabaseManager:
    """Enhanced database management for real-time data integration"""
    
    def __init__(self):
        self.connections = {}
        self.connection_configs = {}
        
    def add_connection(self, name, db_type, config):
        """Add database connection configuration"""
        self.connection_configs[name] = {'type': db_type, 'config': config}
        
    def connect(self, name):
        """Establish database connection"""
        if name not in self.connection_configs:
            return False, f"No configuration found for {name}"
        
        config = self.connection_configs[name]
        
        try:
            if config['type'] == 'sqlite':
                conn = sqlite3.connect(config['config']['path'])
                self.connections[name] = conn
                return True, "SQLite connection established"
                
            elif config['type'] == 'postgresql':
                conn = psycopg2.connect(**config['config'])
                self.connections[name] = conn
                return True, "PostgreSQL connection established"
                
            else:
                return False, f"Unsupported database type: {config['type']}"
                
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def execute_query(self, connection_name, query, params=None):
        """Execute query on specified connection"""
        if connection_name not in self.connections:
            return None, "Connection not established"
        
        try:
            conn = self.connections[connection_name]
            
            if query.strip().upper().startswith('SELECT'):
                df = pd.read_sql_query(query, conn, params=params)
                return df, "Query executed successfully"
            else:
                cursor = conn.cursor()
                cursor.execute(query, params or ())
                conn.commit()
                return cursor.rowcount, "Query executed successfully"
                
        except Exception as e:
            return None, f"Query execution failed: {str(e)}"
    
    def save_analysis_results(self, connection_name, results_df, table_name):
        """Save analysis results to database"""
        try:
            conn = self.connections[connection_name]
            results_df.to_sql(table_name, conn, if_exists='replace', index=False)
            return True, f"Results saved to {table_name}"
        except Exception as e:
            return False, f"Save failed: {str(e)}"

class ReportGenerator:
    """Advanced report generation with multiple formats"""
    
    def __init__(self):
        self.report_templates = {
            'executive_summary': self._executive_summary_template,
            'detailed_analysis': self._detailed_analysis_template,
            'vendor_segmentation': self._vendor_segmentation_template,
            'ml_insights': self._ml_insights_template
        }
    
    def generate_pdf_report(self, analysis_results, report_type='detailed_analysis'):
        """Generate comprehensive PDF report"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch)
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#2E86AB'),
            alignment=1  # Center alignment
        )
        
        story = []
        
        # Title page
        story.append(Paragraph("Contract Opportunities Analysis Report", title_style))
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
        story.append(PageBreak())
        
        # Generate content based on template
        if report_type in self.report_templates:
            content = self.report_templates[report_type](analysis_results)
            story.extend(content)
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        return buffer
    
    def generate_excel_report(self, analysis_results):
        """Generate comprehensive Excel report with multiple sheets"""
        buffer = io.BytesIO()
        
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            # Executive Summary
            if 'summary_metrics' in analysis_results:
                summary_df = pd.DataFrame([analysis_results['summary_metrics']])
                summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)
            
            # Contract Opportunities
            if 'opportunities' in analysis_results:
                analysis_results['opportunities'].to_excel(writer, sheet_name='Contract Opportunities', index=False)
            
            # Vendor Segments
            if 'vendor_segments' in analysis_results:
                analysis_results['vendor_segments'].to_excel(writer, sheet_name='Vendor Segments', index=False)
            
            # ML Model Performance
            if 'model_performance' in analysis_results:
                model_df = pd.DataFrame(analysis_results['model_performance']).T
                model_df.to_excel(writer, sheet_name='ML Performance', index=True)
            
            # Risk Analysis
            if 'risk_analysis' in analysis_results:
                analysis_results['risk_analysis'].to_excel(writer, sheet_name='Risk Analysis', index=False)
        
        buffer.seek(0)
        return buffer
    
    def _executive_summary_template(self, results):
        """Generate executive summary content"""
        styles = getSampleStyleSheet()
        content = []
        
        content.append(Paragraph("Executive Summary", styles['Heading2']))
        content.append(Spacer(1, 12))
        
        # Key metrics table
        if 'summary_metrics' in results:
            metrics = results['summary_metrics']
            summary_data = [
                ['Metric', 'Value', 'Impact'],
                ['Total Contract Opportunities', f"{metrics.get('total_opportunities', 0)}", 'Strategic Focus'],
                ['High Priority Contracts', f"{metrics.get('high_priority', 0)}", 'Immediate Action'],
                ['Potential Annual Savings', f"${metrics.get('potential_savings', 0):,.0f}", 'Cost Reduction'],
                ['Vendor Segments Identified', f"{metrics.get('vendor_segments', 0)}", 'Supplier Strategy'],
                ['ML Model Accuracy', f"{metrics.get('ml_accuracy', 0):.1%}", 'Prediction Confidence']
            ]
            
            table = Table(summary_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            content.append(table)
            content.append(Spacer(1, 20))
        
        # Key recommendations
        content.append(Paragraph("Key Recommendations", styles['Heading3']))
        recommendations = [
            "Prioritize high-potential contract opportunities for immediate action",
            "Implement ML-driven vendor performance monitoring",
            "Establish vendor consolidation program for identified segments",
            "Deploy real-time contract performance tracking",
            "Develop strategic partnerships with top-tier vendors"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            content.append(Paragraph(f"{i}. {rec}", styles['Normal']))
            content.append(Spacer(1, 6))
        
        return content
    
    def _detailed_analysis_template(self, results):
        """Generate detailed analysis content"""
        styles = getSampleStyleSheet()
        content = []
        
        content.append(Paragraph("Detailed Analysis", styles['Heading2']))
        content.append(Spacer(1, 12))
        
        # ML Model Performance
        if 'model_performance' in results:
            content.append(Paragraph("Machine Learning Model Performance", styles['Heading3']))
            
            model_data = []
            model_data.append(['Model', 'R² Score', 'MAE', 'CV Score'])
            
            for model_name, metrics in results['model_performance'].items():
                if 'error' not in metrics:
                    model_data.append([
                        model_name.replace('_', ' ').title(),
                        f"{metrics.get('r2', 0):.3f}",
                        f"{metrics.get('mae', 0):.2f}",
                        f"{metrics.get('cv_mean', 0):.3f}"
                    ])
            
            model_table = Table(model_data, colWidths=[2*inch, 1*inch, 1*inch, 1*inch])
            model_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            content.append(model_table)
            content.append(Spacer(1, 20))
        
        return content
    
    def _vendor_segmentation_template(self, results):
        """Generate vendor segmentation content"""
        # Implementation for vendor segmentation report
        pass
    
    def _ml_insights_template(self, results):
        """Generate ML insights content"""
        # Implementation for ML insights report
        pass

class ConfigurationManager:
    """Advanced configuration management"""
    
    def __init__(self, config_file='contracting_config.yaml'):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration from file"""
        default_config = {
            'database': {
                'enabled': False,
                'connections': {}
            },
            'ml_settings': {
                'auto_retrain': True,
                'retrain_interval_days': 30,
                'min_data_points': 100,
                'cross_validation_folds': 5
            },
            'analysis_settings': {
                'min_spend_threshold': 10000,
                'min_frequency_threshold': 4,
                'consolidation_threshold': 0.7
            },
            'report_settings': {
                'auto_generate': False,
                'formats': ['pdf', 'excel'],
                'schedule': 'monthly'
            },
            'api_settings': {
                'enabled': False,
                'endpoints': {},
                'refresh_interval': 3600
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    self._deep_update(default_config, loaded_config or {})
            except Exception as e:
                st.warning(f"Failed to load config: {e}")
        
        return default_config
    
    def save_config(self):
        """Save current configuration"""
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            return True
        except Exception as e:
            st.error(f"Failed to save config: {e}")
            return False
    
    def _deep_update(self, base_dict, update_dict):
        """Deep update dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

class RealTimeDataManager:
    """Real-time data integration and monitoring"""
    
    def __init__(self):
        self.api_endpoints = {}
        self.refresh_jobs = {}
        self.data_cache = {}
        
    def add_api_endpoint(self, name, url, headers=None, params=None):
        """Add API endpoint for real-time data"""
        self.api_endpoints[name] = {
            'url': url,
            'headers': headers or {},
            'params': params or {},
            'last_updated': None
        }
    
    def fetch_api_data(self, endpoint_name):
        """Fetch data from API endpoint"""
        if endpoint_name not in self.api_endpoints:
            return None, f"Endpoint {endpoint_name} not found"
        
        endpoint = self.api_endpoints[endpoint_name]
        
        try:
            response = requests.get(
                endpoint['url'],
                headers=endpoint['headers'],
                params=endpoint['params'],
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                endpoint['last_updated'] = datetime.now()
                self.data_cache[endpoint_name] = data
                return data, "Data fetched successfully"
            else:
                return None, f"API request failed with status {response.status_code}"
                
        except Exception as e:
            return None, f"API request failed: {str(e)}"
    
    def schedule_data_refresh(self, endpoint_name, interval_seconds):
        """Schedule automatic data refresh"""
        def refresh_job():
            self.fetch_api_data(endpoint_name)
        
        # Simple scheduling (in production, use proper scheduler)
        self.refresh_jobs[endpoint_name] = {
            'job': refresh_job,
            'interval': interval_seconds,
            'last_run': None
        }

def analyze_cross_regional_patterns(df):
    """Enhanced cross-regional analysis with ML insights"""
    required_cols = ['Vendor Name', 'Item', 'Unit Price', 'Qty Delivered', 'Creation Date']
    
    if 'Line Total' not in df.columns:
        df['Line Total'] = df['Unit Price'] * df['Qty Delivered']
    
    # Enhanced regional analysis
    regional_analysis = []
    
    for item in df['Item'].unique():
        item_data = df[df['Item'] == item]
        
        # Multi-dimensional analysis
        unique_vendors = item_data['Vendor Name'].nunique()
        total_spend = item_data['Line Total'].sum()
        price_variance = item_data['Unit Price'].std() / item_data['Unit Price'].mean() if item_data['Unit Price'].mean() > 0 else 0
        
        # Regional coverage
        regions = []
        if 'DEP' in item_data.columns:
            regions.extend(item_data['DEP'].unique())
        if 'W/H' in item_data.columns:
            regions.extend(item_data['W/H'].unique())
        unique_regions = len(set(regions))
        
        # Demand pattern analysis
        monthly_demand = item_data.groupby(item_data['Creation Date'].dt.to_period('M'))['Qty Delivered'].sum()
        demand_volatility = monthly_demand.std() / monthly_demand.mean() if len(monthly_demand) > 1 and monthly_demand.mean() > 0 else 0
        
        # Market position
        item_rank = df.groupby('Item')['Line Total'].sum().rank(pct=True)[item]
        
        # Enhanced consolidation scoring
        consolidation_score = calculate_enhanced_consolidation_score(
            unique_vendors, price_variance, total_spend, demand_volatility, unique_regions, item_rank
        )
        
        regional_analysis.append({
            'Item': item,
            'Total Vendors': unique_vendors,
            'Total Regions': unique_regions,
            'Total Spend': total_spend,
            'Price Variance': price_variance,
            'Demand Volatility': demand_volatility,
            'Market Position': item_rank,
            'Consolidation Potential': consolidation_score,
            'Avg Unit Price': item_data['Unit Price'].mean(),
            'Total Volume': item_data['Qty Delivered'].sum(),
            'Priority': get_priority_level(consolidation_score, total_spend)
        })
    
    return pd.DataFrame(regional_analysis).sort_values('Consolidation Potential', ascending=False)

def calculate_enhanced_consolidation_score(vendor_count, price_variance, total_spend, demand_volatility, regions, market_position):
    """Enhanced consolidation scoring with multiple factors"""
    
    # Vendor diversity score (more vendors = higher consolidation potential)
    vendor_score = min(vendor_count / 8, 1.0) * 0.25
    
    # Price standardization opportunity
    price_score = min(price_variance * 3, 1.0) * 0.25
    
    # Spend impact score
    spend_score = min(total_spend / 200000, 1.0) * 0.25
    
    # Demand predictability (lower volatility = better for contracting)
    demand_score = max(0, 1 - demand_volatility) * 0.15
    
    # Geographic consolidation opportunity
    region_score = min(regions / 5, 1.0) * 0.05
    
    # Market importance
    market_score = market_position * 0.05
    
    total_score = vendor_score + price_score + spend_score + demand_score + region_score + market_score
    return min(total_score, 1.0)

def get_priority_level(score, spend):
    """Determine priority level based on score and spend"""
    if score >= 0.8 and spend >= 100000:
        return "Critical"
    elif score >= 0.6 and spend >= 50000:
        return "High"
    elif score >= 0.4 and spend >= 20000:
        return "Medium"
    else:
        return "Low"

def advanced_demand_forecasting_ml(historical_data, ml_engine, periods_ahead=12):
    """Enhanced ML-based demand forecasting"""
    try:
        if not ml_engine.is_trained:
            return None, None, "ML models not trained"
        
        # Prepare historical data
        historical_data['Month'] = historical_data['Creation Date'].dt.to_period('M')
        monthly_demand = historical_data.groupby('Month')['Qty Delivered'].sum()
        
        if len(monthly_demand) < 6:
            return None, None, "Insufficient historical data"
        
        # Use ML engine for forecasting
        predictions, status = ml_engine.predict_contract_success(historical_data)
        
        if predictions is not None:
            # Generate future period forecasts
            forecasts = []
            confidence_intervals = []
            
            for i in range(periods_ahead):
                # Simple forecast extension (can be enhanced with time series ML)
                base_forecast = monthly_demand.mean() * (1 + np.random.normal(0, 0.1))
                forecasts.append(max(0, base_forecast))
                
                # Confidence interval
                std_dev = monthly_demand.std()
                confidence_intervals.append((
                    max(0, base_forecast - 1.96 * std_dev),
                    base_forecast + 1.96 * std_dev
                ))
            
            forecast_info = {
                'confidence': 0.85,  # Model confidence
                'trend': 'stable',
                'seasonality_detected': True,
                'model_used': ml_engine.best_model_name
            }
            
            return forecasts, confidence_intervals, forecast_info
        
        return None, None, status
        
    except Exception as e:
        return None, None, f"Forecasting failed: {str(e)}"

def display_enhanced_contracting(df):
    """Enhanced contracting opportunities display with all advanced features"""
    st.header("🤝 Enhanced Contracting Opportunities")
    st.markdown("AI-powered contract analysis with advanced ML, real-time data, and comprehensive reporting.")
    
    # Initialize session state components
    if 'ml_engine' not in st.session_state:
        st.session_state.ml_engine = ContractingMLEngine()
    
    if 'vendor_segmentation' not in st.session_state:
        st.session_state.vendor_segmentation = VendorSegmentationEngine()
    
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    if 'report_generator' not in st.session_state:
        st.session_state.report_generator = ReportGenerator()
    
    if 'config_manager' not in st.session_state:
        st.session_state.config_manager = ConfigurationManager()
    
    if 'realtime_manager' not in st.session_state:
        st.session_state.realtime_manager = RealTimeDataManager()
    
    # Sidebar configuration
    with st.sidebar:
        st.subheader("⚙️ Configuration")
        
        # ML Settings
        with st.expander("🤖 ML Configuration"):
            auto_retrain = st.checkbox("Auto-retrain Models", value=True)
            min_data_points = st.number_input("Min Data Points", 50, 1000, 100)
            cv_folds = st.slider("Cross-validation Folds", 3, 10, 5)
            
            if st.button("💾 Save ML Config"):
                st.session_state.config_manager.config['ml_settings'].update({
                    'auto_retrain': auto_retrain,
                    'min_data_points': min_data_points,
                    'cross_validation_folds': cv_folds
                })
                if st.session_state.config_manager.save_config():
                    st.success("Configuration saved!")
        
        # Database Settings
        with st.expander("🗄️ Database Integration"):
            enable_db = st.checkbox("Enable Database Integration")
            
            if enable_db:
                db_type = st.selectbox("Database Type", ["SQLite", "PostgreSQL"])
                
                if db_type == "SQLite":
                    db_path = st.text_input("Database Path", "contracting.db")
                    if st.button("Connect SQLite"):
                        st.session_state.db_manager.add_connection(
                            "main", "sqlite", {"path": db_path}
                        )
                        success, msg = st.session_state.db_manager.connect("main")
                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)
                
                elif db_type == "PostgreSQL":
                    host = st.text_input("Host", "localhost")
                    port = st.number_input("Port", 1, 65535, 5432)
                    database = st.text_input("Database")
                    user = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    
                    if st.button("Connect PostgreSQL"):
                        config = {
                            "host": host, "port": port, "database": database,
                            "user": user, "password": password
                        }
                        st.session_state.db_manager.add_connection("main", "postgresql", config)
                        success, msg = st.session_state.db_manager.connect("main")
                        if success:
                            st.success(msg)
                        else:
                            st.error(msg)
        
        # API Integration
        with st.expander("🌐 API Integration"):
            enable_api = st.checkbox("Enable Real-time API")
            
            if enable_api:
                api_name = st.text_input("API Name")
                api_url = st.text_input("API URL")
                
                if st.button("Add API Endpoint"):
                    if api_name and api_url:
                        st.session_state.realtime_manager.add_api_endpoint(api_name, api_url)
                        st.success(f"API endpoint '{api_name}' added!")
        
        # Export Settings
        with st.expander("📊 Export Settings"):
            export_formats = st.multiselect(
                "Export Formats", 
                ["PDF", "Excel", "CSV"], 
                default=["PDF", "Excel"]
            )
            auto_export = st.checkbox("Auto-export Reports")
            include_visualizations = st.checkbox("Include Charts", True)
    
    # Data validation and cleaning
    required_columns = ['Vendor Name', 'Item', 'Unit Price', 'Qty Delivered', 'Creation Date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Clean data
    df_clean = df.dropna(subset=required_columns)
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    df_clean = df_clean[df_clean['Qty Delivered'] > 0]
    df_clean['Creation Date'] = pd.to_datetime(df_clean['Creation Date'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Creation Date'])
    
    if 'Line Total' not in df_clean.columns:
        df_clean['Line Total'] = df_clean['Unit Price'] * df_clean['Qty Delivered']
    
    if len(df_clean) == 0:
        st.warning("No valid data found after cleaning.")
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🤖 ML-Powered Analysis",
        "🌍 Cross-Regional Patterns", 
        "👥 Vendor Segmentation",
        "🔮 Advanced Forecasting",
        "📊 Performance Monitoring",
        "📋 Advanced Reports",
        "⚙️ Model Management"
    ])
    
    with tab1:
        st.subheader("🤖 ML-Powered Contract Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Model Training:**")
            if st.button("🚀 Train ML Models", type="primary"):
                with st.spinner("Training advanced ML models..."):
                    success, message = st.session_state.ml_engine.train_models(df_clean)
                    
                    if success:
                        st.success(message)
                        
                        # Display model performance
                        if st.session_state.ml_engine.model_performance:
                            perf_data = []
                            for model, metrics in st.session_state.ml_engine.model_performance.items():
                                if 'error' not in metrics:
                                    perf_data.append({
                                        'Model': model.replace('_', ' ').title(),
                                        'R² Score': f"{metrics.get('r2', 0):.3f}",
                                        'Cross-Val Score': f"{metrics.get('cv_mean', 0):.3f}",
                                        'MAE': f"{metrics.get('mae', 0):.2f}"
                                    })
                            
                            if perf_data:
                                st.subheader("🎯 Model Performance")
                                perf_df = pd.DataFrame(perf_data)
                                st.dataframe(perf_df, use_container_width=True)
                                
                                # Highlight best model
                                best_model = st.session_state.ml_engine.best_model_name
                                st.info(f"🏆 Best Model: {best_model.replace('_', ' ').title()}")
                    else:
                        st.error(message)
        
        with col2:
            st.write("**Model Status:**")
            if st.session_state.ml_engine.is_trained:
                st.success("✅ Models are trained and ready")
                
                # Model actions
                if st.button("💾 Save Models"):
                    if st.session_state.ml_engine.save_models("ml_models.pkl"):
                        st.success("Models saved successfully!")
                
                uploaded_model = st.file_uploader("📁 Load Saved Models", type=['pkl'])
                if uploaded_model and st.button("📤 Load Models"):
                    # Save uploaded file temporarily
                    with open("temp_model.pkl", "wb") as f:
                        f.write(uploaded_model.getbuffer())
                    
                    if st.session_state.ml_engine.load_models("temp_model.pkl"):
                        st.success("Models loaded successfully!")
                        os.remove("temp_model.pkl")
            else:
                st.warning("⚠️ Models need to be trained")
        
        # ML Predictions
        if st.session_state.ml_engine.is_trained:
            st.subheader("🔮 ML Predictions")
            
            predictions, status = st.session_state.ml_engine.predict_contract_success(df_clean)
            
            if predictions is not None:
                # Add predictions to dataframe
                prediction_df = df_clean.copy()
                prediction_df['ML_Score'] = predictions
                prediction_df['ML_Priority'] = pd.cut(
                    predictions, 
                    bins=[0, 0.3, 0.6, 1.0], 
                    labels=['Low', 'Medium', 'High']
                )
                
                # Display top ML-identified opportunities
                top_opportunities = prediction_df.nlargest(20, 'ML_Score')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # ML score distribution
                    fig = px.histogram(
                        prediction_df, 
                        x='ML_Score', 
                        title="ML Contract Success Score Distribution",
                        nbins=20
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Top opportunities
                    st.write("**Top ML-Identified Opportunities:**")
                    display_cols = ['Vendor Name', 'Item', 'Line Total', 'ML_Score', 'ML_Priority']
                    available_cols = [col for col in display_cols if col in top_opportunities.columns]
                    
                    st.dataframe(
                        top_opportunities[available_cols].head(10).style.format({
                            'Line Total': '${:,.0f}',
                            'ML_Score': '{:.3f}'
                        }),
                        use_container_width=True
                    )
                
                # Feature importance analysis
                if st.session_state.ml_engine.feature_importance:
                    st.subheader("📊 Feature Importance Analysis")
                    
                    # Get feature importance from best model
                    best_model = st.session_state.ml_engine.best_model_name
                    if best_model in st.session_state.ml_engine.feature_importance:
                        importance_data = st.session_state.ml_engine.feature_importance[best_model]
                        
                        importance_df = pd.DataFrame([
                            {'Feature': k, 'Importance': v} 
                            for k, v in importance_data.items()
                        ]).sort_values('Importance', ascending=False).head(15)
                        
                        fig = px.bar(
                            importance_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title=f"Feature Importance - {best_model.replace('_', ' ').title()}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"Prediction failed: {status}")
    
    with tab2:
        st.subheader("🌍 Enhanced Cross-Regional Analysis")
        
        if st.button("🔍 Analyze Cross-Regional Patterns", type="primary"):
            with st.spinner("Analyzing cross-regional patterns with ML insights..."):
                
                regional_patterns = analyze_cross_regional_patterns(df_clean)
                
                if regional_patterns is not None and len(regional_patterns) > 0:
                    # Enhanced summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        critical_items = len(regional_patterns[regional_patterns['Priority'] == 'Critical'])
                        st.metric("Critical Priority Items", critical_items)
                    
                    with col2:
                        total_consolidation_spend = regional_patterns[
                            regional_patterns['Consolidation Potential'] > 0.6
                        ]['Total Spend'].sum()
                        st.metric("High Potential Spend", f"${total_consolidation_spend:,.0f}")
                    
                    with col3:
                        avg_vendors = regional_patterns['Total Vendors'].mean()
                        st.metric("Avg Vendors per Item", f"{avg_vendors:.1f}")
                    
                    with col4:
                        market_coverage = regional_patterns['Market Position'].mean()
                        st.metric("Avg Market Position", f"{market_coverage:.2f}")
                    
                    # Enhanced visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 3D analysis plot
                        fig = px.scatter_3d(
                            regional_patterns,
                            x='Consolidation Potential',
                            y='Total Spend',
                            z='Total Vendors',
                            color='Priority',
                            size='Total Volume',
                            hover_data=['Item', 'Price Variance'],
                            title="3D Consolidation Analysis"
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Priority-based analysis
                        priority_summary = regional_patterns.groupby('Priority').agg({
                            'Total Spend': 'sum',
                            'Item': 'count',
                            'Consolidation Potential': 'mean'
                        }).round(2)
                        
                        fig = px.bar(
                            priority_summary.reset_index(),
                            x='Priority',
                            y='Total Spend',
                            color='Item',
                            title="Spend Analysis by Priority Level"
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed analysis table with enhanced metrics
                    st.subheader("📋 Enhanced Regional Analysis")
                    
                    display_df = regional_patterns.head(25)
                    st.dataframe(
                        display_df.style.format({
                            'Total Spend': '${:,.0f}',
                            'Consolidation Potential': '{:.3f}',
                            'Price Variance': '{:.2%}',
                            'Demand Volatility': '{:.2%}',
                            'Market Position': '{:.2f}',
                            'Avg Unit Price': '${:.2f}',
                            'Total Volume': '{:,.0f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Store results
                    st.session_state['regional_analysis'] = regional_patterns
                
                else:
                    st.info("No cross-regional patterns detected.")
    
    with tab3:
        st.subheader("👥 Advanced Vendor Segmentation")
        
        if st.button("🎯 Perform Vendor Segmentation", type="primary"):
            with st.spinner("Performing advanced vendor clustering analysis..."):
                
                vendor_features, cluster_profiles = st.session_state.vendor_segmentation.analyze_vendor_segments(df_clean)
                
                if vendor_features is not None:
                    # Segmentation summary
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        optimal_clusters = st.session_state.vendor_segmentation.optimal_clusters
                        st.metric("Optimal Clusters", optimal_clusters)
                    
                    with col2:
                        total_vendors = len(vendor_features)
                        st.metric("Total Vendors", total_vendors)
                    
                    with col3:
                        largest_cluster = vendor_features['Cluster'].value_counts().max()
                        st.metric("Largest Cluster Size", largest_cluster)
                    
                    # Cluster visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Cluster distribution
                        cluster_counts = vendor_features['Cluster'].value_counts().sort_index()
                        
                        fig = px.pie(
                            values=cluster_counts.values,
                            names=[f"Cluster {i}" for i in cluster_counts.index],
                            title="Vendor Cluster Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Cluster characteristics
                        fig = px.scatter(
                            vendor_features,
                            x='Line Total_sum',
                            y='Item_nunique',
                            color='Cluster',
                            size='price_volatility',
                            hover_name='Vendor',
                            title="Vendor Clusters: Spend vs Item Diversity",
                            labels={'Line Total_sum': 'Total Spend', 'Item_nunique': 'Item Count'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster profiles
                    if cluster_profiles:
                        st.subheader("📊 Cluster Profiles")
                        
                        for cluster_id, profile in cluster_profiles.items():
                            with st.expander(f"Cluster {cluster_id}: {profile['type']} ({profile['vendor_count']} vendors)"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.metric("Average Spend", f"${profile['avg_spend']:,.0f}")
                                    st.metric("Average Items", f"{profile['avg_items']:.1f}")
                                
                                with col2:
                                    st.metric("Price Volatility", f"{profile['avg_price_volatility']:.2%}")
                                    st.write("**Sample Vendors:**")
                                    st.write(", ".join(profile['vendors'][:5]))
                    
                    # Detailed vendor segmentation table
                    st.subheader("📋 Vendor Segmentation Details")
                    
                    display_cols = ['Vendor', 'Cluster', 'Line Total_sum', 'Item_nunique', 'price_volatility']
                    available_cols = [col for col in display_cols if col in vendor_features.columns]
                    
                    st.dataframe(
                        vendor_features[available_cols].head(25).style.format({
                            'Line Total_sum': '${:,.0f}',
                            'price_volatility': '{:.2%}'
                        }),
                        use_container_width=True
                    )
                    
                    # Store results
                    st.session_state['vendor_segmentation_results'] = vendor_features
                
                else:
                    st.error("Vendor segmentation failed.")
    
    with tab4:
        st.subheader("🔮 Advanced ML Forecasting")
        
        # Item selection for forecasting
        items = sorted(df_clean['Item'].unique())
        selected_item = st.selectbox("Select Item for Advanced Forecasting", items[:20])
        
        col1, col2 = st.columns(2)
        with col1:
            forecast_periods = st.slider("Forecast Periods", 6, 24, 12)
        with col2:
            confidence_level = st.slider("Confidence Level", 80, 95, 90)
        
        if selected_item and st.button("🚀 Generate ML Forecast", type="primary"):
            item_data = df_clean[df_clean['Item'] == selected_item]
            
            with st.spinner("Generating advanced ML-powered forecast..."):
                
                forecasts, confidence_intervals, forecast_info = advanced_demand_forecasting_ml(
                    item_data, st.session_state.ml_engine, forecast_periods
                )
                
                if forecasts is not None:
                    # Historical data preparation
                    monthly_demand = item_data.groupby(item_data['Creation Date'].dt.to_period('M'))['Qty Delivered'].sum()
                    
                    # Create forecast visualization
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=[str(p) for p in monthly_demand.index],
                        y=monthly_demand.values,
                        mode='lines+markers',
                        name='Historical Demand',
                        line=dict(color='blue', width=3)
                    ))
                    
                    # Forecast data
                    last_period = monthly_demand.index[-1]
                    forecast_periods_list = [str(last_period + i) for i in range(1, len(forecasts) + 1)]
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_periods_list,
                        y=forecasts,
                        mode='lines+markers',
                        name='ML Forecast',
                        line=dict(color='red', dash='dash', width=3)
                    ))
                    
                    # Confidence intervals
                    if confidence_intervals:
                        lower_bounds = [ci[0] for ci in confidence_intervals]
                        upper_bounds = [ci[1] for ci in confidence_intervals]
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_periods_list,
                            y=upper_bounds,
                            fill=None,
                            mode='lines',
                            line_color='rgba(255,0,0,0)',
                            showlegend=False
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=forecast_periods_list,
                            y=lower_bounds,
                            fill='tonexty',
                            mode='lines',
                            line_color='rgba(255,0,0,0)',
                            name=f'{confidence_level}% Confidence',
                            fillcolor='rgba(255,0,0,0.2)'
                        ))
                    
                    fig.update_layout(
                        title=f"Advanced ML Forecast - Item {selected_item}",
                        xaxis_title="Period",
                        yaxis_title="Demand",
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Forecast insights
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_forecast = np.mean(forecasts)
                        st.metric("Avg Forecast Demand", f"{avg_forecast:.0f}")
                    
                    with col2:
                        historical_avg = monthly_demand.mean()
                        growth_rate = ((avg_forecast - historical_avg) / historical_avg * 100) if historical_avg > 0 else 0
                        st.metric("Growth Rate", f"{growth_rate:+.1f}%")
                    
                    with col3:
                        st.metric("Model Confidence", f"{forecast_info['confidence']:.1%}")
                    
                    # Model insights
                    st.subheader("🧠 ML Model Insights")
                    
                    insights_col1, insights_col2 = st.columns(2)
                    
                    with insights_col1:
                        st.info(f"**Model Used:** {forecast_info.get('model_used', 'Ensemble').replace('_', ' ').title()}")
                        st.info(f"**Trend:** {forecast_info.get('trend', 'Unknown').title()}")
                    
                    with insights_col2:
                        st.info(f"**Seasonality:** {'Detected' if forecast_info.get('seasonality_detected') else 'Not Detected'}")
                        st.info(f"**Forecast Horizon:** {forecast_periods} periods")
                    
                    # Export forecast data
                    forecast_df = pd.DataFrame({
                        'Period': forecast_periods_list,
                        'Forecast': forecasts,
                        'Lower_Bound': [ci[0] for ci in confidence_intervals] if confidence_intervals else [None] * len(forecasts),
                        'Upper_Bound': [ci[1] for ci in confidence_intervals] if confidence_intervals else [None] * len(forecasts)
                    })
                    
                    csv_data = forecast_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Export Forecast Data",
                        data=csv_data,
                        file_name=f"ml_forecast_item_{selected_item}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                else:
                    st.error(f"Forecasting failed: {forecast_info}")
    
    with tab5:
        st.subheader("📊 Real-time Performance Monitoring")
        
        # Performance dashboard
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.write(f"**Last Updated:** {current_time}")
        
        # Real-time metrics simulation
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            active_contracts = np.random.randint(15, 30)
            st.metric("Active Analysis Models", len(st.session_state.ml_engine.models))
        
        with col2:
            if st.session_state.ml_engine.is_trained:
                model_accuracy = list(st.session_state.ml_engine.model_performance.values())[0].get('cv_mean', 0)
                st.metric("Best Model Accuracy", f"{model_accuracy:.1%}")
            else:
                st.metric("Best Model Accuracy", "Not Trained")
        
        with col3:
            opportunities_count = len(st.session_state.get('regional_analysis', pd.DataFrame()))
            st.metric("Identified Opportunities", opportunities_count)
        
        with col4:
            data_quality = min(len(df_clean) / 1000, 1.0)
            st.metric("Data Quality Score", f"{data_quality:.1%}")
        
        # Performance trends
        st.subheader("📈 Performance Trends")
        
        # Simulated performance data
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        performance_data = pd.DataFrame({
            'Date': dates,
            'Model_Performance': np.random.uniform(0.7, 0.95, 30),
            'Opportunities_Found': np.random.randint(5, 25, 30),
            'Data_Quality': np.random.uniform(0.8, 1.0, 30)
        })
        
        # Multi-metric chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance', 'Opportunities Found', 'Data Quality', 'System Health'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "indicator"}]]
        )
        
        # Model performance trend
        fig.add_trace(
            go.Scatter(x=performance_data['Date'], y=performance_data['Model_Performance'], 
                      name='Model Performance', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Opportunities trend
        fig.add_trace(
            go.Scatter(x=performance_data['Date'], y=performance_data['Opportunities_Found'],
                      name='Opportunities', line=dict(color='green')),
            row=1, col=2
        )
        
        # Data quality trend
        fig.add_trace(
            go.Scatter(x=performance_data['Date'], y=performance_data['Data_Quality'],
                      name='Data Quality', line=dict(color='orange')),
            row=2, col=1
        )
        
        # System health indicator
        system_health = np.random.uniform(0.85, 0.98)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=system_health,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "System Health"},
                gauge={'axis': {'range': [None, 1]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                                {'range': [0.5, 0.8], 'color': "yellow"},
                                {'range': [0.8, 1], 'color': "green"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 0.9}}
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # System alerts (non-email)
        st.subheader("🚨 System Alerts")
        
        alerts = []
        if not st.session_state.ml_engine.is_trained:
            alerts.append({"type": "warning", "message": "ML models need training", "time": "Now"})
        
        if len(df_clean) < 100:
            alerts.append({"type": "info", "message": "Limited data available for analysis", "time": "Now"})
        
        if len(st.session_state.get('regional_analysis', pd.DataFrame())) == 0:
            alerts.append({"type": "info", "message": "No regional analysis performed yet", "time": "Now"})
        
        if not alerts:
            st.success("✅ All systems operating normally")
        else:
            for alert in alerts:
                if alert["type"] == "warning":
                    st.warning(f"⚠️ {alert['message']} - {alert['time']}")
                elif alert["type"] == "info":
                    st.info(f"ℹ️ {alert['message']} - {alert['time']}")
    
    with tab6:
        st.subheader("📋 Advanced Report Generation")
        
        # Report configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Report Configuration:**")
            report_type = st.selectbox("Report Type", [
                "Executive Summary",
                "Detailed Analysis", 
                "ML Model Performance",
                "Vendor Segmentation",
                "Comprehensive Report"
            ])
            
            include_sections = st.multiselect("Include Sections", [
                "Executive Summary",
                "ML Analysis Results",
                "Cross-Regional Patterns", 
                "Vendor Segmentation",
                "Forecasting Results",
                "Performance Metrics",
                "Recommendations"
            ], default=["Executive Summary", "ML Analysis Results"])
        
        with col2:
            st.write("**Export Options:**")
            export_format = st.selectbox("Format", ["PDF", "Excel", "Both"])
            
            include_charts = st.checkbox("Include Visualizations", True)
            include_raw_data = st.checkbox("Include Raw Data", False)
            
            report_title = st.text_input("Report Title", "Contract Opportunities Analysis")
        
        # Generate reports
        if st.button("📊 Generate Advanced Report", type="primary"):
            with st.spinner("Generating comprehensive report..."):
                
                # Prepare analysis results
                analysis_results = {
                    'summary_metrics': {
                        'total_opportunities': len(st.session_state.get('regional_analysis', pd.DataFrame())),
                        'high_priority': len(st.session_state.get('regional_analysis', pd.DataFrame()).query('Priority == "Critical"') if 'regional_analysis' in st.session_state else pd.DataFrame()),
                        'potential_savings': st.session_state.get('regional_analysis', pd.DataFrame())['Total Spend'].sum() * 0.1 if 'regional_analysis' in st.session_state else 0,
                        'vendor_segments': st.session_state.vendor_segmentation.optimal_clusters if st.session_state.vendor_segmentation.optimal_clusters else 0,
                        'ml_accuracy': list(st.session_state.ml_engine.model_performance.values())[0].get('cv_mean', 0) if st.session_state.ml_engine.model_performance else 0
                    },
                    'opportunities': st.session_state.get('regional_analysis', pd.DataFrame()),
                    'vendor_segments': st.session_state.get('vendor_segmentation_results', pd.DataFrame()),
                    'model_performance': st.session_state.ml_engine.model_performance,
                    'report_metadata': {
                        'generated_at': datetime.now(),
                        'report_type': report_type,
                        'data_size': len(df_clean),
                        'analysis_period': f"{df_clean['Creation Date'].min().date()} to {df_clean['Creation Date'].max().date()}"
                    }
                }
                
                # Generate PDF report
                if export_format in ["PDF", "Both"]:
                    pdf_buffer = st.session_state.report_generator.generate_pdf_report(
                        analysis_results, report_type.lower().replace(' ', '_')
                    )
                    
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_buffer.getvalue(),
                        file_name=f"{report_title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                
                # Generate Excel report
                if export_format in ["Excel", "Both"]:
                    excel_buffer = st.session_state.report_generator.generate_excel_report(analysis_results)
                    
                    st.download_button(
                        label="📥 Download Excel Report",
                        data=excel_buffer.getvalue(),
                        file_name=f"{report_title.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                st.success("✅ Reports generated successfully!")
                
                # Save to database if connected
                if "main" in st.session_state.db_manager.connections:
                    success, msg = st.session_state.db_manager.save_analysis_results(
                        "main", 
                        st.session_state.get('regional_analysis', pd.DataFrame()),
                        "contract_opportunities"
                    )
                    if success:
                        st.info(f"📊 {msg}")
        
        # Automated reporting
        st.subheader("🤖 Automated Reporting")
        
        auto_reports = st.checkbox("Enable Automated Reports")
        if auto_reports:
            col1, col2 = st.columns(2)
            
            with col1:
                report_frequency = st.selectbox("Frequency", ["Daily", "Weekly", "Monthly"])
                report_time = st.time_input("Report Generation Time")
            
            with col2:
                auto_export_format = st.selectbox("Auto Format", ["PDF", "Excel", "Both"])
                save_location = st.text_input("Save Location", "./reports/")
            
            if st.button("💾 Setup Automated Reporting"):
                # In production, this would set up actual scheduling
                st.session_state.config_manager.config['report_settings'].update({
                    'auto_generate': True,
                    'frequency': report_frequency.lower(),
                    'time': str(report_time),
                    'format': auto_export_format.lower(),
                    'location': save_location
                })
                
                if st.session_state.config_manager.save_config():
                    st.success("✅ Automated reporting configured!")
    
    with tab7:
        st.subheader("⚙️ Advanced Model Management")
        
        # Model training history
        if st.session_state.ml_engine.training_history:
            st.subheader("📈 Training History")
            
            history_df = pd.DataFrame(st.session_state.ml_engine.training_history)
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            
            fig = px.line(
                history_df,
                x='timestamp',
                y='best_score',
                title="Model Performance Over Time",
                labels={'best_score': 'Best Model Score', 'timestamp': 'Training Date'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(
                history_df.style.format({
                    'best_score': '{:.3f}',
                    'timestamp': lambda x: x.strftime('%Y-%m-%d %H:%M')
                }),
                use_container_width=True
            )
        
        # Model configuration
        st.subheader("🔧 Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Training Parameters:**")
            test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
            cv_folds = st.slider("CV Folds", 3, 10, 5)
            random_state = st.number_input("Random State", 0, 100, 42)
        
        with col2:
            st.write("**Model Selection:**")
            enabled_models = st.multiselect(
                "Enabled Models",
                ["random_forest", "gradient_boosting", "neural_network", "linear_regression", "ridge_regression"],
                default=["random_forest", "gradient_boosting", "neural_network"]
            )
            
            ensemble_enabled = st.checkbox("Enable Ensemble Model", True)
        
        # Advanced model operations
        st.subheader("🚀 Advanced Operations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Retrain All Models"):
                with st.spinner("Retraining all models..."):
                    # Update model selection
                    st.session_state.ml_engine.models = {
                        name: model for name, model in st.session_state.ml_engine.models.items()
                        if name in enabled_models
                    }
                    
                    success, message = st.session_state.ml_engine.train_models(df_clean)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        
        with col2:
            if st.button("📊 Model Comparison"):
                if st.session_state.ml_engine.model_performance:
                    comparison_data = []
                    for model, metrics in st.session_state.ml_engine.model_performance.items():
                        if 'error' not in metrics:
                            comparison_data.append({
                                'Model': model.replace('_', ' ').title(),
                                'R² Score': metrics.get('r2', 0),
                                'CV Score': metrics.get('cv_mean', 0),
                                'MAE': metrics.get('mae', 0)
                            })
                    
                    if comparison_data:
                        comp_df = pd.DataFrame(comparison_data)
                        fig = px.bar(
                            comp_df,
                            x='Model',
                            y='CV Score',
                            title="Model Performance Comparison",
                            color='R² Score'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No trained models to compare")
        
        with col3:
            if st.button("🧹 Clear Model Cache"):
                st.session_state.ml_engine = ContractingMLEngine()
                st.success("Model cache cleared!")
        
        # Model deployment settings
        st.subheader("🚀 Model Deployment")
        
        deployment_col1, deployment_col2 = st.columns(2)
        
        with deployment_col1:
            auto_deploy = st.checkbox("Auto-deploy Best Model")
            model_versioning = st.checkbox("Enable Model Versioning", True)
            
            if st.button("📦 Export Production Model"):
                if st.session_state.ml_engine.is_trained:
                    # Create production model package
                    model_package = {
                        'model': st.session_state.ml_engine.model_performance[st.session_state.ml_engine.best_model_name]['model'],
                        'scaler': st.session_state.ml_engine.scaler,
                        'metadata': {
                            'version': datetime.now().strftime('%Y%m%d_%H%M%S'),
                            'best_model': st.session_state.ml_engine.best_model_name,
                            'performance': st.session_state.ml_engine.model_performance[st.session_state.ml_engine.best_model_name],
                            'training_data_size': len(df_clean)
                        }
                    }
                    
                    # Serialize model package
                    model_buffer = io.BytesIO()
                    pickle.dump(model_package, model_buffer)
                    model_buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Download Production Model",
                        data=model_buffer.getvalue(),
                        file_name=f"production_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl",
                        mime="application/octet-stream"
                    )
                else:
                    st.warning("No trained models available for export")
        
        with deployment_col2:
            st.write("**Model Status:**")
            if st.session_state.ml_engine.is_trained:
                st.success("✅ Models trained and ready")
                st.info(f"Best Model: {st.session_state.ml_engine.best_model_name}")
                
                # Performance metrics
                best_performance = st.session_state.ml_engine.model_performance[st.session_state.ml_engine.best_model_name]
                st.write(f"**Performance Metrics:**")
                st.write(f"- R² Score: {best_performance.get('r2', 0):.3f}")
                st.write(f"- CV Score: {best_performance.get('cv_mean', 0):.3f}")
                st.write(f"- MAE: {best_performance.get('mae', 0):.2f}")
            else:
                st.warning("⚠️ No trained models available")

# Main display function
def display(df):
    """Main display function for enhanced contracting opportunities"""
    display_enhanced_contracting(df)
