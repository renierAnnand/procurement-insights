# Enhanced Contracting Opportunities Module with Advanced Currency Support
# 
# 🌟 NEW FEATURES:
# 
# 💱 COMPREHENSIVE CURRENCY SYSTEM:
# - Support for 50+ currencies including all peso currencies (COP, MXN, CLP, ARS, etc.)
# - Smart detection of peso and large-nominal currencies
# - Automatic validation and warning system for conversion errors
# - Post-conversion analysis with statistical validation
# - Conversion audit log for transparency
# 
# 🪙 PESO CURRENCY SPECIALISTS:
# - Colombian Peso (COP), Mexican Peso (MXN), Chilean Peso (CLP)
# - Argentine Peso (ARS), Philippine Peso (PHP), Dominican Peso (DOP)
# - Proper exchange rates and validation for large amounts
# - Smart detection based on amount patterns
# 
# 🔍 INTELLIGENT VALIDATION:
# - Pre-conversion warnings for suspected issues
# - Post-conversion statistical analysis
# - Unrealistic amount detection and correction suggestions
# - Peso-specific validation rules
# 
# ⚡ BULK SELECTION SYSTEM:
# - Select All/Deselect All buttons throughout interface
# - Priority-based smart selections (High Priority Only, Top Spenders, etc.)
# - Batch operations for negotiation planning
# - Quick action presets for common scenarios
# 
# 🎯 ENHANCED USER EXPERIENCE:
# - Visual indicators for currency types (🪙 for pesos, 📈 for large-nominal)
# - Comprehensive help guides and troubleshooting
# - Real-time validation feedback
# - Professional Arabic SAR formatting (ر.س)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, date
import calendar
import json
import re

# Currency conversion and formatting functions
def get_default_exchange_rates():
    """Get default exchange rates to SAR (Saudi Riyal) with comprehensive peso currencies"""
    # Updated with accurate rates as of 2024 - peso currencies included
    return {
        'SAR': 1.0,
        
        # Major currencies
        'USD': 3.75,  # 1 USD = 3.75 SAR
        'EUR': 4.10,  # 1 EUR = 4.10 SAR
        'GBP': 4.70,  # 1 GBP = 4.70 SAR
        'CHF': 4.20,  # 1 CHF = 4.20 SAR
        'JPY': 0.025, # 1 JPY = 0.025 SAR
        'CNY': 0.52,  # 1 CNY = 0.52 SAR
        
        # GCC and Middle East
        'AED': 1.02,  # 1 AED = 1.02 SAR
        'QAR': 1.03,  # 1 QAR = 1.03 SAR
        'KWD': 12.30, # 1 KWD = 12.30 SAR
        'BHD': 9.95,  # 1 BHD = 9.95 SAR
        'OMR': 9.75,  # 1 OMR = 9.75 SAR
        'EGP': 0.24,  # 1 EGP = 0.24 SAR
        'JOD': 5.29,  # 1 JOD = 5.29 SAR
        'LBP': 0.0025, # 1 LBP = 0.0025 SAR
        'ILS': 1.02,  # 1 ILS = 1.02 SAR
        'TRY': 0.22,  # 1 TRY = 0.22 SAR
        
        # PESO CURRENCIES (Critical for Latin America)
        'COP': 0.0009,  # 1 Colombian Peso = 0.0009 SAR (~4,500 COP = 1 USD)
        'MXN': 0.19,    # 1 Mexican Peso = 0.19 SAR (~20 MXN = 1 USD)
        'CLP': 0.0042,  # 1 Chilean Peso = 0.0042 SAR (~900 CLP = 1 USD)
        'ARS': 0.0095,  # 1 Argentine Peso = 0.0095 SAR (~400 ARS = 1 USD)
        'UYU': 0.096,   # 1 Uruguayan Peso = 0.096 SAR (~39 UYU = 1 USD)
        'PEN': 1.01,    # 1 Peruvian Sol = 1.01 SAR (~3.7 PEN = 1 USD)
        'BOB': 0.54,    # 1 Bolivian Boliviano = 0.54 SAR (~6.9 BOB = 1 USD)
        'PYG': 0.0005,  # 1 Paraguayan Guarani = 0.0005 SAR (~7,300 PYG = 1 USD)
        'DOP': 0.067,   # 1 Dominican Peso = 0.067 SAR (~56 DOP = 1 USD)
        'CUP': 0.156,   # 1 Cuban Peso = 0.156 SAR (~24 CUP = 1 USD)
        
        # Asian currencies
        'INR': 0.045,   # 1 INR = 0.045 SAR
        'PKR': 0.013,   # 1 PKR = 0.013 SAR
        'BDT': 0.034,   # 1 Bangladeshi Taka = 0.034 SAR
        'LKR': 0.012,   # 1 Sri Lankan Rupee = 0.012 SAR
        'NPR': 0.028,   # 1 Nepalese Rupee = 0.028 SAR
        'KRW': 0.0028,  # 1 Korean Won = 0.0028 SAR
        'THB': 0.11,    # 1 Thai Baht = 0.11 SAR
        'MYR': 0.85,    # 1 Malaysian Ringgit = 0.85 SAR
        'SGD': 2.80,    # 1 Singapore Dollar = 2.80 SAR
        'HKD': 0.48,    # 1 Hong Kong Dollar = 0.48 SAR
        'PHP': 0.067,   # 1 Philippine Peso = 0.067 SAR
        'IDR': 0.00025, # 1 Indonesian Rupiah = 0.00025 SAR (very small)
        'VND': 0.00015, # 1 Vietnamese Dong = 0.00015 SAR (very small)
        
        # Other major currencies
        'CAD': 2.80,    # 1 CAD = 2.80 SAR
        'AUD': 2.50,    # 1 AUD = 2.50 SAR
        'NZD': 2.30,    # 1 NZD = 2.30 SAR
        'SEK': 0.36,    # 1 SEK = 0.36 SAR
        'NOK': 0.35,    # 1 NOK = 0.35 SAR
        'DKK': 0.55,    # 1 DKK = 0.55 SAR
        'PLN': 0.93,    # 1 PLN = 0.93 SAR
        'CZK': 0.17,    # 1 CZK = 0.17 SAR
        'HUF': 0.010,   # 1 HUF = 0.010 SAR
        'ZAR': 0.21,    # 1 ZAR = 0.21 SAR
        'BRL': 0.75,    # 1 Brazilian Real = 0.75 SAR
        'RUB': 0.042,   # 1 RUB = 0.042 SAR
        
        # African currencies
        'NGN': 0.0025,  # 1 Nigerian Naira = 0.0025 SAR
        'GHS': 0.32,    # 1 Ghanaian Cedi = 0.32 SAR
        'KES': 0.025,   # 1 Kenyan Shilling = 0.025 SAR
        'UGX': 0.001,   # 1 Ugandan Shilling = 0.001 SAR
        'TZS': 0.0016,  # 1 Tanzanian Shilling = 0.0016 SAR
        'MAD': 0.37,    # 1 Moroccan Dirham = 0.37 SAR
        'TND': 1.22,    # 1 Tunisian Dinar = 1.22 SAR
        'DZD': 0.028,   # 1 Algerian Dinar = 0.028 SAR
    }

def get_currency_info():
    """Get currency information including warnings for high-value currencies"""
    return {
        # High-value currencies (small amounts typical)
        'high_value': ['USD', 'EUR', 'GBP', 'CHF', 'KWD', 'BHD', 'OMR', 'JOD'],
        
        # Medium-value currencies
        'medium_value': ['SAR', 'AED', 'QAR', 'CAD', 'AUD', 'SGD', 'NZD'],
        
        # Large-nominal currencies (large amounts typical)
        'large_nominal': {
            'COP': {'name': 'Colombian Peso', 'typical_range': '1,000 - 100,000', 'warning_threshold': 1000000},
            'CLP': {'name': 'Chilean Peso', 'typical_range': '500 - 50,000', 'warning_threshold': 500000},
            'ARS': {'name': 'Argentine Peso', 'typical_range': '100 - 50,000', 'warning_threshold': 200000},
            'KRW': {'name': 'Korean Won', 'typical_range': '1,000 - 500,000', 'warning_threshold': 2000000},
            'IDR': {'name': 'Indonesian Rupiah', 'typical_range': '10,000 - 1,000,000', 'warning_threshold': 5000000},
            'VND': {'name': 'Vietnamese Dong', 'typical_range': '20,000 - 2,000,000', 'warning_threshold': 10000000},
            'PYG': {'name': 'Paraguayan Guarani', 'typical_range': '5,000 - 500,000', 'warning_threshold': 2000000},
            'UGX': {'name': 'Ugandan Shilling', 'typical_range': '1,000 - 200,000', 'warning_threshold': 1000000},
            'TZS': {'name': 'Tanzanian Shilling', 'typical_range': '1,000 - 500,000', 'warning_threshold': 2000000},
            'LBP': {'name': 'Lebanese Pound', 'typical_range': '5,000 - 500,000', 'warning_threshold': 2000000},
            'NGN': {'name': 'Nigerian Naira', 'typical_range': '500 - 200,000', 'warning_threshold': 1000000}
        },
        
        # Peso family specifically
        'peso_currencies': ['COP', 'MXN', 'CLP', 'ARS', 'UYU', 'DOP', 'CUP', 'PHP']
    }

def detect_currency_from_data(df):
    """Detect currencies present in the dataset with enhanced peso detection"""
    currencies_found = set()
    
    # Check if there's a currency column
    currency_columns = ['Currency', 'Curr', 'CCY', 'Currency Code', 'Moneda', 'Divisa']
    currency_col = None
    
    for col in currency_columns:
        if col in df.columns:
            currency_col = col
            break
    
    if currency_col:
        unique_currencies = df[currency_col].dropna().unique()
        currencies_found.update(unique_currencies)
    
    # Enhanced currency detection from price/amount columns
    price_columns = [col for col in df.columns if any(word in col.lower() for word in ['price', 'amount', 'cost', 'value', 'total', 'precio', 'costo', 'valor'])]
    
    for col in price_columns:
        if df[col].dtype == 'object':  # String column might contain currency symbols
            sample_values = df[col].dropna().astype(str).head(200)  # Increased sample size
            for value in sample_values:
                value_str = str(value).upper()
                
                # Extract currency codes (3 letters)
                currency_matches = re.findall(r'\b[A-Z]{3}\b', value_str)
                currencies_found.update(currency_matches)
                
                # Look for peso indicators
                peso_patterns = [
                    r'\bPESO\b', r'\bPSO\b', r'\bCOP\b', r'\bMXN\b', r'\bCLP\b', 
                    r'\bARS\b', r'\bUYU\b', r'\bDOP\b', r'\bPHP\b', r'\$\s*\d+[,.]?\d*\s*(COP|MXN|CLP|ARS)'
                ]
                
                for pattern in peso_patterns:
                    if re.search(pattern, value_str):
                        if 'COP' in value_str or 'COLOMBIAN' in value_str:
                            currencies_found.add('COP')
                        elif 'MXN' in value_str or 'MEXICAN' in value_str:
                            currencies_found.add('MXN')
                        elif 'CLP' in value_str or 'CHILEAN' in value_str:
                            currencies_found.add('CLP')
                        elif 'ARS' in value_str or 'ARGENTINIAN' in value_str or 'ARGENTINE' in value_str:
                            currencies_found.add('ARS')
                        elif 'PHP' in value_str or 'PHILIPPINE' in value_str:
                            currencies_found.add('PHP')
                        break
                
                # Look for other large-nominal currency indicators
                if re.search(r'\b(RUPIAH|IDR)\b', value_str):
                    currencies_found.add('IDR')
                if re.search(r'\b(DONG|VND)\b', value_str):
                    currencies_found.add('VND')
                if re.search(r'\b(WON|KRW)\b', value_str):
                    currencies_found.add('KRW')
    
    # Intelligent currency guessing based on amount patterns
    if not currencies_found and len(price_columns) > 0:
        # Analyze amount patterns to guess currency
        first_price_col = price_columns[0]
        numeric_values = []
        
        for value in df[first_price_col].dropna().head(100):
            try:
                if isinstance(value, str):
                    # Extract numeric part
                    numeric_part = re.sub(r'[^\d.,]', '', str(value))
                    if numeric_part:
                        numeric_part = numeric_part.replace(',', '')
                        numeric_values.append(float(numeric_part))
                else:
                    numeric_values.append(float(value))
            except:
                continue
        
        if numeric_values:
            avg_value = sum(numeric_values) / len(numeric_values)
            max_value = max(numeric_values)
            
            # Currency guessing based on typical amount ranges
            if avg_value > 100000:  # Very large amounts
                if max_value > 10000000:
                    currencies_found.add('IDR')  # Indonesian Rupiah
                elif max_value > 1000000:
                    currencies_found.add('COP')  # Colombian Peso likely
                else:
                    currencies_found.add('KRW')  # Korean Won
            elif avg_value > 10000:
                currencies_found.add('CLP')  # Chilean Peso likely
            elif avg_value > 1000:
                currencies_found.add('MXN')  # Mexican Peso or other medium peso
            else:
                currencies_found.add('SAR')  # Assume SAR for reasonable amounts
    
    # If still no currencies found, assume SAR
    if not currencies_found:
        currencies_found.add('SAR')
    
    return list(currencies_found)

def validate_currency_conversion(df, currencies_detected, exchange_rates):
    """Validate currency conversion and warn about potential issues"""
    warnings = []
    recommendations = []
    
    currency_info = get_currency_info()
    
    # Check for large-nominal currencies
    for currency in currencies_detected:
        if currency in currency_info['large_nominal']:
            currency_data = currency_info['large_nominal'][currency]
            warnings.append({
                'type': 'large_nominal',
                'currency': currency,
                'name': currency_data['name'],
                'message': f"{currency_data['name']} typically has large amounts ({currency_data['typical_range']})"
            })
    
    # Analyze amount distributions after conversion
    if 'Unit Price' in df.columns:
        converted_prices = []
        for idx, row in df.head(100).iterrows():  # Sample first 100 rows
            try:
                amount = extract_numeric_value(row['Unit Price'])
                if amount > 0:
                    # Assume first detected currency for validation
                    currency = currencies_detected[0] if currencies_detected else 'SAR'
                    converted_amount = convert_to_sar(amount, currency, exchange_rates)
                    converted_prices.append(converted_amount)
            except:
                continue
        
        if converted_prices:
            avg_sar_price = sum(converted_prices) / len(converted_prices)
            max_sar_price = max(converted_prices)
            
            # Warning thresholds
            if avg_sar_price > 100000:  # Average > 100K SAR
                warnings.append({
                    'type': 'high_average',
                    'message': f"Average unit price after conversion: {format_sar_amount(avg_sar_price)}",
                    'severity': 'high'
                })
                recommendations.append("Consider checking if amounts were already in SAR or if exchange rates are correct")
            
            if max_sar_price > 1000000:  # Max > 1M SAR
                warnings.append({
                    'type': 'very_high_max',
                    'message': f"Highest unit price after conversion: {format_sar_amount(max_sar_price)}",
                    'severity': 'critical'
                })
                recommendations.append("Very high amounts detected - likely conversion error")
    
    # Check for suspicious exchange rates
    for currency, rate in exchange_rates.items():
        if currency in currency_info['large_nominal'] and rate > 0.1:
            warnings.append({
                'type': 'suspicious_rate',
                'currency': currency,
                'rate': rate,
                'message': f"{currency} rate ({rate}) seems too high for a large-nominal currency",
                'severity': 'medium'
            })
    
    return warnings, recommendations

def smart_currency_detection_interface(df):
    """Enhanced currency detection interface with validation"""
    st.write("### 🕵️ Smart Currency Detection")
    
    # Detect currencies
    detected_currencies = detect_currency_from_data(df)
    currency_info = get_currency_info()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🔍 Detected Currencies:**")
        for curr in detected_currencies:
            # Add context for peso currencies
            if curr in currency_info['peso_currencies']:
                st.write(f"• {curr} 🪙 (Peso Currency - Large Amounts Expected)")
            elif curr in currency_info['large_nominal']:
                currency_name = currency_info['large_nominal'][curr]['name']
                st.write(f"• {curr} 📈 ({currency_name} - Large Nominal)")
            else:
                st.write(f"• {curr}")
    
    with col2:
        st.write("**⚠️ Currency Warnings:**")
        
        peso_count = len([c for c in detected_currencies if c in currency_info['peso_currencies']])
        large_nominal_count = len([c for c in detected_currencies if c in currency_info['large_nominal']])
        
        if peso_count > 0:
            st.warning(f"🪙 {peso_count} peso currency(ies) detected - amounts will be large")
        
        if large_nominal_count > 0:
            st.info(f"📊 {large_nominal_count} large-nominal currency(ies) detected")
        
        if not any(c in currency_info['high_value'] for c in detected_currencies):
            st.success("✅ No high-value currencies detected")
    
    return detected_currencies

def extract_numeric_value(value_str):
    """Extract numeric value from string that might contain currency symbols"""
    if pd.isna(value_str):
        return 0
    
    # Convert to string and remove common currency symbols and letters
    clean_str = re.sub(r'[A-Za-z$€£¥₹₽﷼,\s]', '', str(value_str))
    
    try:
        return float(clean_str)
    except:
        return 0

def convert_to_sar(amount, from_currency, exchange_rates):
    """Convert amount from given currency to SAR"""
    if pd.isna(amount) or amount == 0:
        return 0
    
    from_currency = str(from_currency).upper()
    
    if from_currency not in exchange_rates:
        st.warning(f"Exchange rate not found for {from_currency}. Using SAR rate.")
        return float(amount)
    
    return float(amount) * exchange_rates[from_currency]

def format_sar_amount(amount):
    """Format amount in Saudi Riyal with proper formatting"""
    if pd.isna(amount) or amount == 0:
        return "ر.س 0"
    
    # Format with thousands separator and 2 decimal places
    formatted = f"ر.س {amount:,.2f}"
    return formatted

def setup_currency_conversion(df):
    """Setup currency conversion interface with enhanced peso support and validation"""
    st.subheader("💱 Enhanced Currency Configuration")
    
    # Smart currency detection
    detected_currencies = smart_currency_detection_interface(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Configuration Options:**")
        
        # Currency mapping interface
        currency_source = st.selectbox(
            "Currency Data Source",
            ["Auto-detect from Price Column", "Dedicated Currency Column", "Single Currency (All SAR)", "Manual Currency Assignment"]
        )
        
        if currency_source == "Dedicated Currency Column":
            currency_columns = [col for col in df.columns if any(word in col.lower() for word in ['curr', 'ccy', 'moneda', 'divisa'])]
            if currency_columns:
                currency_column = st.selectbox("Select Currency Column", currency_columns)
            else:
                st.warning("No currency column found. Using auto-detection.")
                currency_source = "Auto-detect from Price Column"
        
        elif currency_source == "Manual Currency Assignment":
            manual_currency = st.selectbox("Assign Single Currency to All Data", 
                                         options=list(get_default_exchange_rates().keys()),
                                         index=list(get_default_exchange_rates().keys()).index('SAR'))
            detected_currencies = [manual_currency]
    
    with col2:
        # Data preview for validation
        st.write("**Data Preview:**")
        if 'Unit Price' in df.columns:
            sample_prices = df['Unit Price'].head(5)
            for i, price in enumerate(sample_prices):
                st.write(f"• Row {i+1}: {price}")
    
    # Exchange rate configuration with peso-specific warnings
    st.write("**Exchange Rates to SAR (Saudi Riyal):**")
    
    default_rates = get_default_exchange_rates()
    currency_info = get_currency_info()
    exchange_rates = {}
    
    # Create enhanced exchange rate input interface
    rate_cols = st.columns(min(4, len(detected_currencies)))
    
    for i, currency in enumerate(detected_currencies):
        with rate_cols[i % len(rate_cols)]:
            # Special handling for peso currencies
            if currency in currency_info['peso_currencies']:
                st.write(f"🪙 **{currency}** (Peso Currency)")
                if currency in currency_info['large_nominal']:
                    typical_range = currency_info['large_nominal'][currency]['typical_range']
                    st.caption(f"Typical amounts: {typical_range}")
            elif currency in currency_info['large_nominal']:
                currency_name = currency_info['large_nominal'][currency]['name']
                st.write(f"📈 **{currency}** ({currency_name})")
                typical_range = currency_info['large_nominal'][currency]['typical_range']
                st.caption(f"Typical amounts: {typical_range}")
            else:
                st.write(f"**{currency}**")
            
            if currency in default_rates:
                default_rate = default_rates[currency]
                
                # Show rate with context
                if currency in currency_info['large_nominal']:
                    st.caption(f"💡 Very small rate expected for {currency}")
                
                rate = st.number_input(
                    f"1 {currency} = ? SAR",
                    value=default_rate,
                    step=0.0001 if default_rate < 0.01 else 0.01,
                    format="%.6f" if default_rate < 0.01 else "%.4f",
                    key=f"rate_{currency}"
                )
            else:
                rate = st.number_input(
                    f"1 {currency} = ? SAR",
                    value=1.0,
                    step=0.01,
                    format="%.4f",
                    key=f"rate_{currency}"
                )
            exchange_rates[currency] = rate
    
    # Pre-conversion validation
    if len(detected_currencies) > 0:
        st.write("### 🔍 Pre-Conversion Validation")
        
        warnings, recommendations = validate_currency_conversion(df, detected_currencies, exchange_rates)
        
        if warnings:
            for warning in warnings:
                if warning.get('severity') == 'critical':
                    st.error(f"🚨 **Critical**: {warning['message']}")
                elif warning.get('severity') == 'high':
                    st.warning(f"⚠️ **Warning**: {warning['message']}")
                else:
                    st.info(f"ℹ️ **Info**: {warning['message']}")
        
        if recommendations:
            st.write("**💡 Recommendations:**")
            for rec in recommendations:
                st.write(f"• {rec}")
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        
        # Peso-specific settings
        st.write("**Peso Currency Settings:**")
        peso_currencies_found = [c for c in detected_currencies if c in currency_info['peso_currencies']]
        
        if peso_currencies_found:
            st.info(f"Found peso currencies: {', '.join(peso_currencies_found)}")
            
            peso_validation = st.checkbox("Enable peso amount validation", value=True)
            if peso_validation:
                st.write("Will validate that peso amounts are in expected ranges")
        
        # Large amount threshold
        large_amount_threshold = st.number_input(
            "Alert threshold for unit prices (SAR)", 
            value=50000, 
            step=10000,
            help="Alert if unit prices exceed this amount after conversion"
        )
        
        # Auto-correction option
        auto_correction = st.checkbox(
            "Enable smart correction suggestions",
            value=True,
            help="Suggest corrections for obviously wrong conversions"
        )
    
    # Exchange rate update options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Update Exchange Rates", help="Refresh with latest default rates"):
            st.success("Exchange rates updated!")
            exchange_rates.update(default_rates)
    
    with col2:
        if st.button("🪙 Peso Rate Helper", help="Set recommended rates for peso currencies"):
            peso_rates = {
                'COP': 0.0009, 'MXN': 0.19, 'CLP': 0.0042, 'ARS': 0.0095,
                'UYU': 0.096, 'DOP': 0.067, 'PHP': 0.067
            }
            for peso_curr in peso_currencies_found:
                if peso_curr in peso_rates:
                    exchange_rates[peso_curr] = peso_rates[peso_curr]
            st.success("Peso rates updated!")
    
    with col3:
        if st.button("🔧 Reset to Defaults", help="Reset all rates to default values"):
            exchange_rates.update(default_rates)
            st.success("All rates reset to defaults!")
    
    return exchange_rates, currency_source

def post_conversion_validation(df_converted, original_df, exchange_rates):
    """Validate converted amounts and provide warnings/suggestions"""
    
    st.write("### 📊 Post-Conversion Analysis")
    
    validation_results = {
        'warnings': [],
        'recommendations': [],
        'statistics': {}
    }
    
    if 'Unit Price' in df_converted.columns:
        # Calculate statistics
        converted_prices = df_converted['Unit Price'].dropna()
        
        if len(converted_prices) > 0:
            validation_results['statistics'] = {
                'count': len(converted_prices),
                'min': converted_prices.min(),
                'max': converted_prices.max(),
                'mean': converted_prices.mean(),
                'median': converted_prices.median(),
                'std': converted_prices.std()
            }
            
            # Analysis columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", f"{validation_results['statistics']['count']:,}")
            with col2:
                st.metric("Min Price", format_sar_amount(validation_results['statistics']['min']))
            with col3:
                st.metric("Max Price", format_sar_amount(validation_results['statistics']['max']))
            with col4:
                st.metric("Avg Price", format_sar_amount(validation_results['statistics']['mean']))
            
            # Warning checks
            warnings = []
            recommendations = []
            
            # Check for extremely high values
            if validation_results['statistics']['max'] > 5000000:  # > 5M SAR
                warnings.append(f"🚨 Extremely high unit price detected: {format_sar_amount(validation_results['statistics']['max'])}")
                recommendations.append("Check if this is a bulk order or if currency conversion is incorrect")
            
            # Check for very high average
            if validation_results['statistics']['mean'] > 100000:  # > 100K SAR average
                warnings.append(f"⚠️ Very high average unit price: {format_sar_amount(validation_results['statistics']['mean'])}")
                recommendations.append("Consider if amounts were already in SAR or if peso/large-nominal currency rates are wrong")
            
            # Check for unrealistic distributions
            if validation_results['statistics']['std'] > validation_results['statistics']['mean'] * 10:
                warnings.append("📊 Extremely high price variance detected")
                recommendations.append("Data may contain mixed currencies or scales")
            
            # Check for suspiciously round numbers (indicating possible pre-conversion)
            round_number_count = len(converted_prices[converted_prices % 1000 == 0])
            if round_number_count > len(converted_prices) * 0.5:
                warnings.append("🔢 Many round numbers detected - amounts may already be in SAR")
                recommendations.append("Consider using 'Single Currency (All SAR)' option")
            
            # Display warnings and recommendations
            if warnings:
                st.write("**⚠️ Validation Warnings:**")
                for warning in warnings:
                    st.warning(warning)
            
            if recommendations:
                st.write("**💡 Recommendations:**")
                for rec in recommendations:
                    st.info(f"• {rec}")
            
            # Show sample of highest values for manual review
            st.write("**🔍 Highest Unit Prices (Manual Review):**")
            if len(df_converted) > 0:
                top_prices = df_converted.nlargest(5, 'Unit Price')[['Vendor Name', 'Item', 'Unit Price']]
                if 'Item Description' in df_converted.columns:
                    top_prices['Item Description'] = df_converted.nlargest(5, 'Unit Price')['Item Description']
                
                st.dataframe(
                    top_prices.style.format({
                        'Unit Price': lambda x: format_sar_amount(x)
                    }),
                    use_container_width=True
                )
            
            # Quick fix suggestions
            st.write("**🔧 Quick Fix Options:**")
            
            fix_col1, fix_col2, fix_col3 = st.columns(3)
            
            with fix_col1:
                if st.button("🔄 Try All SAR Mode", key="fix_all_sar"):
                    st.info("💡 Suggestion: Reset conversion and select 'Single Currency (All SAR)'")
            
            with fix_col2:
                if st.button("🪙 Fix Peso Rates", key="fix_peso_rates"):
                    currency_info = get_currency_info()
                    peso_suggestions = []
                    for curr, rate in exchange_rates.items():
                        if curr in currency_info['peso_currencies'] and rate > 0.1:
                            peso_suggestions.append(f"{curr}: Try {get_default_exchange_rates().get(curr, 0.01):.6f}")
                    
                    if peso_suggestions:
                        st.info("💡 Peso rate suggestions:")
                        for suggestion in peso_suggestions:
                            st.write(f"• {suggestion}")
                    else:
                        st.success("✅ Peso rates look reasonable")
            
            with fix_col3:
                if st.button("📊 Show Distribution", key="show_distribution"):
                    # Create price distribution chart
                    import plotly.express as px
                    
                    # Sample data for histogram (limit to 1000 points for performance)
                    sample_prices = converted_prices.sample(min(1000, len(converted_prices)))
                    
                    fig = px.histogram(
                        x=sample_prices,
                        nbins=50,
                        title="Unit Price Distribution (SAR)",
                        labels={'x': 'Unit Price (SAR)', 'y': 'Frequency'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
            
            # Export validation report
            if st.button("📥 Export Validation Report", key="export_validation"):
                validation_report = {
                    'Validation_Type': ['Total Records', 'Min Price (SAR)', 'Max Price (SAR)', 'Avg Price (SAR)', 'Median Price (SAR)'],
                    'Value': [
                        validation_results['statistics']['count'],
                        validation_results['statistics']['min'],
                        validation_results['statistics']['max'],
                        validation_results['statistics']['mean'],
                        validation_results['statistics']['median']
                    ],
                    'Status': ['✅ OK', '✅ OK', '⚠️ Check' if validation_results['statistics']['max'] > 1000000 else '✅ OK',
                              '⚠️ Check' if validation_results['statistics']['mean'] > 50000 else '✅ OK', '✅ OK']
                }
                
                report_df = pd.DataFrame(validation_report)
                csv = report_df.to_csv(index=False)
                
                st.download_button(
                    label="Download Validation Report",
                    data=csv,
                    file_name=f"currency_validation_report_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    return validation_results

def convert_dataframe_to_sar(df, exchange_rates, currency_source="Auto-detect from Price Column"):
    """Convert all monetary columns in dataframe to SAR with enhanced validation"""
    df_converted = df.copy()
    
    # Identify monetary columns
    monetary_columns = []
    for col in df.columns:
        if any(word in col.lower() for word in ['price', 'amount', 'cost', 'value', 'total', 'spend', 'precio', 'costo', 'valor']):
            monetary_columns.append(col)
    
    conversion_log = []
    
    # Convert each monetary column
    for col in monetary_columns:
        if col in df_converted.columns:
            converted_values = []
            
            for idx, row in df_converted.iterrows():
                try:
                    # Extract numeric value
                    if df_converted[col].dtype == 'object':
                        amount = extract_numeric_value(row[col])
                    else:
                        amount = float(row[col]) if not pd.isna(row[col]) else 0
                    
                    # Determine currency
                    if currency_source == "Single Currency (All SAR)":
                        currency = "SAR"
                    elif currency_source == "Manual Currency Assignment":
                        # Use the first currency in exchange_rates as the assigned currency
                        currency = list(exchange_rates.keys())[0] if exchange_rates else "SAR"
                    elif currency_source == "Dedicated Currency Column":
                        currency_col = None
                        for c in ['Currency', 'Curr', 'CCY', 'Currency Code', 'Moneda', 'Divisa']:
                            if c in df_converted.columns:
                                currency_col = c
                                break
                        currency = row[currency_col] if currency_col else "SAR"
                    else:  # Auto-detect
                        # Try to extract currency from the value string
                        if df_converted[col].dtype == 'object':
                            value_str = str(row[col]).upper()
                            currency_match = re.search(r'\b([A-Z]{3})\b', value_str)
                            
                            # Check for peso indicators
                            if 'COP' in value_str or 'COLOMBIAN' in value_str:
                                currency = 'COP'
                            elif 'MXN' in value_str or 'MEXICAN' in value_str:
                                currency = 'MXN'
                            elif 'CLP' in value_str or 'CHILEAN' in value_str:
                                currency = 'CLP'
                            elif 'ARS' in value_str or 'ARGENTIN' in value_str:
                                currency = 'ARS'
                            elif 'PHP' in value_str or 'PHILIPPINE' in value_str:
                                currency = 'PHP'
                            elif currency_match:
                                currency = currency_match.group(1)
                            else:
                                # Smart guessing based on amount range
                                if amount > 100000:
                                    currency = 'COP'  # Likely Colombian Peso
                                elif amount > 10000:
                                    currency = 'CLP'  # Likely Chilean Peso
                                elif amount > 1000:
                                    currency = 'MXN'  # Likely Mexican Peso
                                else:
                                    currency = "SAR"
                        else:
                            # For numeric columns, use the most common currency from exchange_rates
                            currency = list(exchange_rates.keys())[0] if exchange_rates else "SAR"
                    
                    # Convert to SAR
                    sar_amount = convert_to_sar(amount, currency, exchange_rates)
                    converted_values.append(sar_amount)
                    
                    # Log conversion for audit
                    if idx < 10:  # Log first 10 conversions
                        conversion_log.append({
                            'row': idx,
                            'column': col,
                            'original_amount': amount,
                            'currency': currency,
                            'sar_amount': sar_amount,
                            'rate_used': exchange_rates.get(currency, 1.0)
                        })
                
                except Exception as e:
                    converted_values.append(0)
                    if idx < 5:  # Log first 5 errors
                        conversion_log.append({
                            'row': idx,
                            'column': col,
                            'error': str(e),
                            'original_value': row[col]
                        })
            
            # Update the column with converted values
            df_converted[col] = converted_values
    
    # Store conversion log in session state for debugging
    st.session_state['conversion_log'] = conversion_log
    
    return df_converted
    """Convert all monetary columns in dataframe to SAR"""
    df_converted = df.copy()
    
    # Identify monetary columns
    monetary_columns = []
    for col in df.columns:
        if any(word in col.lower() for word in ['price', 'amount', 'cost', 'value', 'total', 'spend']):
            monetary_columns.append(col)
    
    # Convert each monetary column
    for col in monetary_columns:
        if col in df_converted.columns:
            converted_values = []
            
            for idx, row in df_converted.iterrows():
                try:
                    # Extract numeric value
                    if df_converted[col].dtype == 'object':
                        amount = extract_numeric_value(row[col])
                    else:
                        amount = float(row[col]) if not pd.isna(row[col]) else 0
                    
                    # Determine currency
                    if currency_source == "Single Currency (All SAR)":
                        currency = "SAR"
                    elif currency_source == "Dedicated Currency Column":
                        currency_col = None
                        for c in ['Currency', 'Curr', 'CCY', 'Currency Code']:
                            if c in df_converted.columns:
                                currency_col = c
                                break
                        currency = row[currency_col] if currency_col else "SAR"
                    else:  # Auto-detect
                        # Try to extract currency from the value string
                        if df_converted[col].dtype == 'object':
                            currency_match = re.search(r'\b([A-Z]{3})\b', str(row[col]))
                            currency = currency_match.group(1) if currency_match else "SAR"
                        else:
                            currency = "SAR"
                    
                    # Convert to SAR
                    sar_amount = convert_to_sar(amount, currency, exchange_rates)
                    converted_values.append(sar_amount)
                
                except Exception as e:
                    converted_values.append(0)
            
            # Update the column with converted values
            df_converted[col] = converted_values
    
    return df_converted

def calculate_vendor_performance_score(vendor_data):
    """Calculate comprehensive vendor performance score"""
    metrics = {}
    
    # Ensure vendor_data is a DataFrame
    if not isinstance(vendor_data, pd.DataFrame) or len(vendor_data) == 0:
        return {
            'overall_score': 0.5,
            'volume_consistency': 0.5,
            'price_stability': 0.5,
            'lead_time_consistency': 0.5,
            'delivery_performance': 0.5,
            'quality_score': 0.5
        }
    
    # Volume consistency (coefficient of variation of monthly orders)
    try:
        monthly_orders = vendor_data.groupby(vendor_data['Creation Date'].dt.to_period('M')).size()
        if isinstance(monthly_orders, pd.Series) and len(monthly_orders) > 1:
            metrics['volume_consistency'] = 1 - (monthly_orders.std() / monthly_orders.mean()) if monthly_orders.mean() > 0 else 0
        else:
            metrics['volume_consistency'] = 0
    except:
        metrics['volume_consistency'] = 0
    
    # Price stability (1 - coefficient of variation of unit prices)
    if 'Unit Price' in vendor_data.columns:
        price_cv = vendor_data['Unit Price'].std() / vendor_data['Unit Price'].mean() if vendor_data['Unit Price'].mean() > 0 else 0
        metrics['price_stability'] = max(0, 1 - price_cv)
    else:
        metrics['price_stability'] = 0
    
    # Lead time consistency (if available)
    if 'lead_time_days' in vendor_data.columns:
        lead_times = vendor_data['lead_time_days'].dropna()
        if len(lead_times) > 0:
            lt_cv = lead_times.std() / lead_times.mean() if lead_times.mean() > 0 else 0
            metrics['lead_time_consistency'] = max(0, 1 - lt_cv)
        else:
            metrics['lead_time_consistency'] = 0.5
    else:
        metrics['lead_time_consistency'] = 0.5
    
    # Delivery performance (if available)
    if 'delivery_delay_days' in vendor_data.columns:
        delays = vendor_data['delivery_delay_days'].dropna()
        if len(delays) > 0:
            on_time_rate = len(delays[delays <= 0]) / len(delays)
            metrics['delivery_performance'] = on_time_rate
        else:
            metrics['delivery_performance'] = 0.5
    else:
        metrics['delivery_performance'] = 0.5
    
    # Quality score (based on rejection rates if available)
    if 'Qty Rejected' in vendor_data.columns and 'Qty Delivered' in vendor_data.columns:
        total_delivered = vendor_data['Qty Delivered'].sum()
        total_rejected = vendor_data['Qty Rejected'].sum()
        if total_delivered > 0:
            rejection_rate = total_rejected / (total_delivered + total_rejected)
            metrics['quality_score'] = 1 - rejection_rate
        else:
            metrics['quality_score'] = 0.5
    else:
        metrics['quality_score'] = 0.5
    
    # Calculate weighted overall score
    weights = {
        'volume_consistency': 0.25,
        'price_stability': 0.25,
        'lead_time_consistency': 0.20,
        'delivery_performance': 0.20,
        'quality_score': 0.10
    }
    
    overall_score = sum(metrics[metric] * weights[metric] for metric in weights)
    
    return {
        'overall_score': overall_score,
        **metrics
    }

def calculate_negotiation_potential_index(vendor_data):
    """Calculate negotiation potential based on historical patterns"""
    
    # Ensure vendor_data is a DataFrame
    if not isinstance(vendor_data, pd.DataFrame) or len(vendor_data) == 0:
        return {
            'negotiation_index': 0.5,
            'negotiation_class': "Unknown",
            'price_flexibility': 0.5,
            'discount_behavior': 0,
            'order_frequency_score': 0,
            'volume_consistency': 0.5
        }
    
    try:
        # Price variance over time (higher = more flexible)
        if 'Unit Price' in vendor_data.columns and len(vendor_data['Unit Price'].dropna()) > 0:
            price_variance = vendor_data['Unit Price'].std() / vendor_data['Unit Price'].mean() if vendor_data['Unit Price'].mean() > 0 else 0
            price_flexibility = min(price_variance * 2, 1.0)  # Normalize to 0-1
        else:
            price_flexibility = 0
        
        # Discount behavior (if price has decreased over time)
        discount_behavior = 0
        if len(vendor_data) > 3:
            try:
                recent_prices = vendor_data.nlargest(3, 'Creation Date')['Unit Price'].mean()
                older_prices = vendor_data.nsmallest(3, 'Creation Date')['Unit Price'].mean()
                if older_prices > 0:
                    discount_behavior = max(0, (older_prices - recent_prices) / older_prices)
            except:
                discount_behavior = 0
        
        # Order frequency (regular customers might get better treatment)
        order_frequency_score = min(len(vendor_data) / 50, 1.0)  # Normalize to 0-1
        
        # Volume consistency (consistent buyers have more leverage)
        if 'Qty Delivered' in vendor_data.columns and len(vendor_data['Qty Delivered'].dropna()) > 0:
            qty_mean = vendor_data['Qty Delivered'].mean()
            if qty_mean > 0:
                volume_consistency = 1 - (vendor_data['Qty Delivered'].std() / qty_mean)
                volume_consistency = max(0, min(volume_consistency, 1.0))
            else:
                volume_consistency = 0
        else:
            volume_consistency = 0
        
        # Combine metrics
        negotiation_index = (
            price_flexibility * 0.3 + 
            discount_behavior * 0.3 + 
            order_frequency_score * 0.2 + 
            volume_consistency * 0.2
        )
        
        # Classify negotiation potential
        if negotiation_index >= 0.7:
            negotiation_class = "Open to Negotiation"
        elif negotiation_index >= 0.4:
            negotiation_class = "Moderate"
        else:
            negotiation_class = "Rigid"
        
        return {
            'negotiation_index': negotiation_index,
            'negotiation_class': negotiation_class,
            'price_flexibility': price_flexibility,
            'discount_behavior': discount_behavior,
            'order_frequency_score': order_frequency_score,
            'volume_consistency': volume_consistency
        }
    
    except Exception as e:
        # Return default values in case of any error
        return {
            'negotiation_index': 0.5,
            'negotiation_class': "Unknown",
            'price_flexibility': 0.5,
            'discount_behavior': 0,
            'order_frequency_score': 0,
            'volume_consistency': 0.5
        }

def calculate_supplier_risk_index(vendor_data):
    """Calculate supplier risk index"""
    
    # Ensure vendor_data is a DataFrame
    if not isinstance(vendor_data, pd.DataFrame) or len(vendor_data) == 0:
        return {
            'overall_risk': 0.5,
            'risk_class': "Medium Risk",
            'delivery_risk': 0.5,
            'quality_risk': 0.1,
            'price_volatility_risk': 0.5,
            'continuity_risk': 0.5
        }
    
    risk_factors = {}
    
    try:
        # Delivery reliability
        if 'delivery_delay_days' in vendor_data.columns:
            delays = vendor_data['delivery_delay_days'].dropna()
            if isinstance(delays, pd.Series) and len(delays) > 0:
                on_time_rate = len(delays[delays <= 0]) / len(delays)
                risk_factors['delivery_risk'] = 1 - on_time_rate
            else:
                risk_factors['delivery_risk'] = 0.5
        else:
            risk_factors['delivery_risk'] = 0.5
    except:
        risk_factors['delivery_risk'] = 0.5
    
    try:
        # Quality risk
        if 'Qty Rejected' in vendor_data.columns and 'Qty Delivered' in vendor_data.columns:
            total_delivered = vendor_data['Qty Delivered'].sum()
            total_rejected = vendor_data['Qty Rejected'].sum()
            if total_delivered > 0:
                rejection_rate = total_rejected / (total_delivered + total_rejected)
                risk_factors['quality_risk'] = rejection_rate
            else:
                risk_factors['quality_risk'] = 0.1
        else:
            risk_factors['quality_risk'] = 0.1
    except:
        risk_factors['quality_risk'] = 0.1
    
    try:
        # Price volatility risk
        if 'Unit Price' in vendor_data.columns and len(vendor_data['Unit Price'].dropna()) > 0:
            price_cv = vendor_data['Unit Price'].std() / vendor_data['Unit Price'].mean() if vendor_data['Unit Price'].mean() > 0 else 0
            risk_factors['price_volatility_risk'] = min(price_cv, 1.0)
        else:
            risk_factors['price_volatility_risk'] = 0.5
    except:
        risk_factors['price_volatility_risk'] = 0.5
    
    try:
        # Supply continuity risk (based on order gaps)
        if len(vendor_data) > 1:
            vendor_data_sorted = vendor_data.sort_values('Creation Date')
            date_gaps = vendor_data_sorted['Creation Date'].diff().dt.days.dropna()
            if len(date_gaps) > 0:
                avg_gap = date_gaps.mean()
                max_gap = date_gaps.max()
                if avg_gap > 0:
                    continuity_risk = min((max_gap - avg_gap) / avg_gap, 1.0)
                else:
                    continuity_risk = 0
                risk_factors['continuity_risk'] = max(0, continuity_risk)
            else:
                risk_factors['continuity_risk'] = 0.5
        else:
            risk_factors['continuity_risk'] = 0.5
    except:
        risk_factors['continuity_risk'] = 0.5
    
    # Calculate overall risk index
    weights = {
        'delivery_risk': 0.3,
        'quality_risk': 0.3,
        'price_volatility_risk': 0.2,
        'continuity_risk': 0.2
    }
    
    overall_risk = sum(risk_factors[factor] * weights[factor] for factor in weights)
    
    # Risk classification
    if overall_risk <= 0.3:
        risk_class = "Low Risk"
    elif overall_risk <= 0.6:
        risk_class = "Medium Risk"
    else:
        risk_class = "High Risk"
    
    return {
        'overall_risk': overall_risk,
        'risk_class': risk_class,
        **risk_factors
    }

def recommend_contract_type(spend_data, vendor_performance, negotiation_potential):
    """Recommend optimal contract type based on multiple factors"""
    
    annual_spend = spend_data['total_spend']
    order_frequency = spend_data['order_frequency']
    demand_predictability = spend_data['demand_predictability']
    
    recommendations = []
    
    # Fixed Price Contract
    if demand_predictability > 0.7 and vendor_performance['price_stability'] > 0.6:
        score = (demand_predictability * 0.4 + vendor_performance['price_stability'] * 0.4 + 
                min(annual_spend / 100000, 1.0) * 0.2)
        recommendations.append({
            'type': 'Fixed Price Contract',
            'score': score,
            'reasoning': 'High demand predictability and price stability make this ideal for fixed pricing'
        })
    
    # Volume Commitment
    if annual_spend > 50000 and order_frequency > 6:
        score = (min(annual_spend / 200000, 1.0) * 0.5 + 
                min(order_frequency / 24, 1.0) * 0.3 + 
                negotiation_potential['negotiation_index'] * 0.2)
        recommendations.append({
            'type': 'Volume Commitment',
            'score': score,
            'reasoning': 'High spend and frequency provide leverage for volume-based discounts'
        })
    
    # Blanket Purchase Order
    if order_frequency > 12 and demand_predictability > 0.5:
        score = (min(order_frequency / 36, 1.0) * 0.5 + 
                demand_predictability * 0.3 + 
                vendor_performance['overall_score'] * 0.2)
        recommendations.append({
            'type': 'Blanket Purchase Order',
            'score': score,
            'reasoning': 'Frequent orders and predictable demand suit blanket PO arrangements'
        })
    
    # Requirements Contract
    if demand_predictability < 0.5 and annual_spend > 25000:
        score = ((1 - demand_predictability) * 0.4 + 
                min(annual_spend / 100000, 1.0) * 0.4 + 
                vendor_performance['delivery_performance'] * 0.2)
        recommendations.append({
            'type': 'Requirements Contract',
            'score': score,
            'reasoning': 'Uncertain demand requires flexible volume arrangements'
        })
    
    # Sort by score and return top recommendation
    if recommendations:
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[0]
    else:
        return {
            'type': 'Spot Purchase',
            'score': 0.5,
            'reasoning': 'Current purchasing pattern is optimal for spot buying'
        }

def analyze_multi_vendor_aggregation(df):
    """Identify opportunities for vendor consolidation"""
    
    # Ensure df is valid
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return pd.DataFrame()
    
    # Group by item and analyze vendor distribution
    item_vendor_analysis = []
    
    try:
        for item in df['Item'].unique():
            try:
                item_data = df[df['Item'] == item]
                vendors = item_data['Vendor Name'].unique()
                
                if len(vendors) > 1:  # Multiple vendors for same item
                    total_spend = (item_data['Unit Price'] * item_data['Qty Delivered']).sum()
                    
                    vendor_spend = item_data.groupby('Vendor Name').agg({
                        'Unit Price': 'mean',
                        'Qty Delivered': 'sum',
                        'Creation Date': 'count'
                    }).reset_index()
                    vendor_spend['Total Spend'] = vendor_spend['Unit Price'] * vendor_spend['Qty Delivered']
                    vendor_spend = vendor_spend.sort_values('Total Spend', ascending=False)
                    
                    if len(vendor_spend) == 0:
                        continue
                    
                    # Calculate consolidation potential
                    primary_vendor = vendor_spend.iloc[0]
                    secondary_spend = vendor_spend.iloc[1:]['Total Spend'].sum() if len(vendor_spend) > 1 else 0
                    
                    # Potential savings from consolidating to primary vendor
                    avg_primary_price = primary_vendor['Unit Price']
                    secondary_data = item_data[item_data['Vendor Name'] != primary_vendor['Vendor Name']]
                    secondary_volume = secondary_data['Qty Delivered'].sum() if len(secondary_data) > 0 else 0
                    
                    potential_savings = 0
                    if len(secondary_data) > 0:
                        avg_secondary_price = secondary_data['Unit Price'].mean()
                        if avg_primary_price < avg_secondary_price:
                            potential_savings = (avg_secondary_price - avg_primary_price) * secondary_volume
                    
                    item_desc = item_data['Item Description'].iloc[0] if 'Item Description' in item_data.columns else f"Item {item}"
                    
                    item_vendor_analysis.append({
                        'Item': item,
                        'Item Description': str(item_desc)[:50] + "..." if len(str(item_desc)) > 50 else str(item_desc),
                        'Vendor Count': len(vendors),
                        'Total Spend': total_spend,
                        'Primary Vendor': primary_vendor['Vendor Name'],
                        'Primary Vendor Share': (primary_vendor['Total Spend'] / total_spend) * 100 if total_spend > 0 else 0,
                        'Secondary Spend': secondary_spend,
                        'Potential Savings': potential_savings,
                        'Consolidation Priority': 'High' if potential_savings > 5000 and len(vendors) > 2 else 'Medium' if potential_savings > 1000 else 'Low'
                    })
            
            except Exception as e:
                # Skip this item if there's an error
                continue
    
    except Exception as e:
        st.error(f"Error in vendor consolidation analysis: {str(e)}")
        return pd.DataFrame()
    
    return pd.DataFrame(item_vendor_analysis).sort_values('Potential Savings', ascending=False) if item_vendor_analysis else pd.DataFrame()

def generate_procurement_calendar(opportunities_df, current_date=None):
    """Generate interactive procurement calendar with key dates"""
    
    if current_date is None:
        current_date = datetime.now()
    
    calendar_events = []
    
    # Contract renewal dates (simulate based on creation patterns)
    for _, opp in opportunities_df.iterrows():
        if opp['Contract Priority'] in ['High Priority', 'Medium Priority']:
            # Simulate contract start date (6 months from now for high priority)
            if opp['Contract Priority'] == 'High Priority':
                start_date = current_date + timedelta(days=30)  # Start in 1 month
                duration_months = 18
            else:
                start_date = current_date + timedelta(days=90)  # Start in 3 months
                duration_months = 12
            
            end_date = start_date + timedelta(days=duration_months * 30)
            renewal_notice_date = end_date - timedelta(days=90)  # 3 months before expiry
            
            calendar_events.append({
                'Event': f'Contract Start - {opp["Vendor Name"]}',
                'Date': start_date,
                'Type': 'Contract Start',
                'Vendor': opp['Vendor Name'],
                'Item': opp['Item'],
                'Priority': opp['Contract Priority'],
                'Description': f'Begin contract for {opp["Item Description"]}'
            })
            
            calendar_events.append({
                'Event': f'Renewal Notice - {opp["Vendor Name"]}',
                'Date': renewal_notice_date,
                'Type': 'Renewal Notice',
                'Vendor': opp['Vendor Name'],
                'Item': opp['Item'],
                'Priority': opp['Contract Priority'],
                'Description': f'Send renewal notice for {opp["Item Description"]}'
            })
            
            calendar_events.append({
                'Event': f'Contract Expiry - {opp["Vendor Name"]}',
                'Date': end_date,
                'Type': 'Contract Expiry',
                'Vendor': opp['Vendor Name'],
                'Item': opp['Item'],
                'Priority': opp['Contract Priority'],
                'Description': f'Contract expires for {opp["Item Description"]}'
            })
    
    return pd.DataFrame(calendar_events).sort_values('Date')

def analyze_contract_suitability(item_vendor_data, min_spend_threshold=10000, min_frequency_threshold=4):
    """Analyze suitability for contracting based on spend and frequency"""
    
    # Ensure item_vendor_data is a DataFrame
    if not isinstance(item_vendor_data, pd.DataFrame) or len(item_vendor_data) == 0:
        return {
            'total_spend': 0,
            'order_frequency': 0,
            'monthly_frequency': 0,
            'demand_predictability': 0,
            'suitability_score': 0,
            'recommendation': "Not Suitable",
            'months_span': 1
        }
    
    try:
        # Calculate key metrics
        total_spend = (item_vendor_data['Unit Price'] * item_vendor_data['Qty Delivered']).sum()
        order_frequency = len(item_vendor_data)
        
        # Calculate time span
        date_range = item_vendor_data['Creation Date'].max() - item_vendor_data['Creation Date'].min()
        months_span = max(date_range.days / 30, 1) if date_range.days > 0 else 1
        monthly_frequency = order_frequency / months_span
        
        # Demand predictability
        monthly_demand = item_vendor_data.groupby(item_vendor_data['Creation Date'].dt.to_period('M'))['Qty Delivered'].sum()
        if isinstance(monthly_demand, pd.Series) and len(monthly_demand) > 1 and monthly_demand.mean() > 0:
            demand_cv = monthly_demand.std() / monthly_demand.mean()
            demand_predictability = max(0, 1 - demand_cv)
        else:
            demand_predictability = 0
        
        # Contract suitability score
        spend_score = min(total_spend / min_spend_threshold, 1.0) if min_spend_threshold > 0 else 1.0
        frequency_score = min(monthly_frequency / (min_frequency_threshold / 12), 1.0)
        
        suitability_score = (spend_score * 0.4 + frequency_score * 0.3 + demand_predictability * 0.3)
        
        # Contract recommendation
        if suitability_score >= 0.7 and total_spend >= min_spend_threshold:
            recommendation = "High Priority"
        elif suitability_score >= 0.5 and total_spend >= min_spend_threshold * 0.5:
            recommendation = "Medium Priority"
        elif suitability_score >= 0.3:
            recommendation = "Low Priority"
        else:
            recommendation = "Not Suitable"
        
        return {
            'total_spend': total_spend,
            'order_frequency': order_frequency,
            'monthly_frequency': monthly_frequency,
            'demand_predictability': demand_predictability,
            'suitability_score': suitability_score,
            'recommendation': recommendation,
            'months_span': months_span
        }
    
    except Exception as e:
        # Return default values in case of any error
        return {
            'total_spend': 0,
            'order_frequency': 0,
            'monthly_frequency': 0,
            'demand_predictability': 0,
            'suitability_score': 0,
            'recommendation': "Not Suitable",
            'months_span': 1
        }

def simulate_multi_year_contract(historical_data, contract_terms, years=3):
    """Simulate multi-year contract performance"""
    
    current_annual_spend = (historical_data['Unit Price'] * historical_data['Qty Delivered']).sum()
    current_annual_volume = historical_data['Qty Delivered'].sum()
    
    simulation_results = []
    
    for year in range(1, years + 1):
        # Assume 5% volume growth per year
        projected_volume = current_annual_volume * (1.05 ** year)
        
        # Base scenario (no contract)
        base_price = historical_data['Unit Price'].mean()
        # Assume 3% annual inflation
        inflated_price = base_price * (1.03 ** year)
        base_cost = projected_volume * inflated_price
        
        # Contract scenario
        contract_price = base_price * (1 - contract_terms.get('volume_discount', 0))
        # Contract price escalation (usually lower than inflation)
        escalated_contract_price = contract_price * (1.02 ** year)
        contract_cost = projected_volume * escalated_contract_price
        
        # Admin savings
        annual_admin_savings = contract_terms.get('admin_savings', 0) * 12  # Monthly savings
        
        # Calculate savings
        total_savings = (base_cost - contract_cost) + annual_admin_savings
        cumulative_savings = sum([
            simulate_multi_year_contract(historical_data, contract_terms, y)['total_savings'] 
            for y in range(1, year + 1)
        ]) if year > 1 else total_savings
        
        simulation_results.append({
            'year': year,
            'projected_volume': projected_volume,
            'base_cost': base_cost,
            'contract_cost': contract_cost,
            'annual_savings': total_savings,
            'cumulative_savings': cumulative_savings,
            'roi_percent': (cumulative_savings / (contract_terms.get('setup_cost', 5000))) * 100 if contract_terms.get('setup_cost', 5000) > 0 else 0
        })
    
    return simulation_results

def display(df):
    st.header("🤝 Enhanced Contracting Opportunities")
    st.markdown("Advanced AI-powered procurement intelligence platform for optimal contracting strategies.")
    
    # Quick help for bulk selection features and currency
    with st.expander("💡 Bulk Selection & Currency Features Guide", expanded=False):
        tab1, tab2 = st.tabs(["🎯 Bulk Selection", "💱 Currency Guide"])
        
        with tab1:
            st.markdown("""
            **New Bulk Selection Features Available:**
            
            **🎯 Smart Identification Tab:**
            - Quick configuration presets (High Value Focus, Standard Analysis, etc.)
            
            **📊 Vendor Performance Tab:**
            - ✅ Select All Vendors / ❌ Deselect All buttons
            
            **🤝 Negotiation Intelligence Tab:**
            - ✅ Select All Open (for negotiation)
            - 🎯 Top 5 Spenders selection
            - ⚡ Quick Wins (Open + Low Risk)
            - 📊 Batch reporting and scheduling
            
            **🔄 Multi-Vendor Consolidation Tab:**
            - ✅ Select All Items / ❌ Deselect All
            - 🔴 High Priority Only
            - 🟡 Medium+ Priority  
            - 💰 Top 10 Savings selection
            
            **🎪 Contract Simulation Tab:**
            - 🎯 Highest Spend vendor selection
            - 🤝 Best Negotiable vendor selection
            
            **📅 Procurement Calendar Tab:**
            - ✅ All Events / ❌ Clear Events
            - ✅ All Priorities / 🔴 High Only
            
            **👥 Collaboration Hub Tab:**
            - 🔴 Highest Priority contract
            - 💰 Highest Spend contract selection
            
            **💡 Pro Tip:** Use these bulk selection features to quickly focus on your most important contracts and vendors!
            """)
        
        with tab2:
            st.markdown("""
            **💱 Enhanced Currency Conversion Guide**
            
            **🪙 Peso Currencies (Large Amounts Expected):**
            - **Colombian Peso (COP)**: 1,000-100,000 COP typical
            - **Mexican Peso (MXN)**: 20-2,000 MXN typical  
            - **Chilean Peso (CLP)**: 500-50,000 CLP typical
            - **Argentine Peso (ARS)**: 100-50,000 ARS typical
            - **Philippine Peso (PHP)**: 50-5,000 PHP typical
            
            **📈 Other Large-Nominal Currencies:**
            - **Korean Won (KRW)**: 1,000-500,000 KRW
            - **Indonesian Rupiah (IDR)**: 10,000-1,000,000 IDR
            - **Vietnamese Dong (VND)**: 20,000-2,000,000 VND
            
            **🔧 Troubleshooting High Amounts:**
            1. **Check original data**: Were amounts already in SAR?
            2. **Verify currency detection**: Did system detect peso currencies correctly?
            3. **Validate exchange rates**: Are peso rates very small (0.001-0.2)?
            4. **Use validation tools**: Click "📊 Validate Conversion" in sidebar
            5. **Try manual assignment**: Use "Manual Currency Assignment" option
            
            **⚠️ Common Issues:**
            - **23 billion SAR total**: Likely peso conversion error
            - **Round numbers**: Data might already be in SAR
            - **Extreme variance**: Mixed currencies detected
            
            **💡 Quick Fixes:**
            - Use "🪙 Peso Rate Helper" button
            - Try "Single Currency (All SAR)" if data already converted
            - Check conversion log for audit trail
            - Use post-conversion validation tools
            """)
    
    # Data validation
    required_columns = ['Vendor Name', 'Item', 'Unit Price', 'Qty Delivered', 'Creation Date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Missing required columns: {', '.join(missing_columns)}")
        return
    
    # Currency conversion setup
    st.sidebar.header("💱 Currency Settings")
    
    with st.sidebar.expander("Currency Configuration", expanded=True):
        exchange_rates, currency_source = setup_currency_conversion(df)
        
        if st.button("Apply Currency Conversion", type="primary"):
            with st.spinner("Converting all amounts to SAR..."):
                df = convert_dataframe_to_sar(df, exchange_rates, currency_source)
                st.success("✅ All amounts converted to Saudi Riyal (SAR)")
                st.session_state['currency_converted'] = True
                st.session_state['converted_df'] = df
                st.session_state['exchange_rates'] = exchange_rates
                st.rerun()  # Refresh to show the updated display
    
    # Additional currency information
    if 'currency_converted' in st.session_state and st.session_state['currency_converted']:
        with st.sidebar.expander("💰 SAR Summary"):
            converted_df = st.session_state['converted_df']
            if 'Unit Price' in converted_df.columns and 'Qty Delivered' in converted_df.columns:
                total_value_sar = (converted_df['Unit Price'] * converted_df['Qty Delivered']).sum()
                st.metric("Total Portfolio Value", format_sar_amount(total_value_sar))
                
                avg_order_value = (converted_df['Unit Price'] * converted_df['Qty Delivered']).mean()
                st.metric("Avg Order Value", format_sar_amount(avg_order_value))
                
                # Currency distribution before conversion
                detected_currencies = detect_currency_from_data(df)
                st.write(f"**Currencies Processed:** {len(detected_currencies)}")
                for curr in detected_currencies[:5]:  # Show top 5
                    st.write(f"• {curr}")
        
        # Enhanced sidebar with bulk action summary
        if 'currency_converted' in st.session_state and st.session_state['currency_converted']:
            with st.sidebar.expander("🎯 Quick Actions Summary"):
                st.write("**Active Selections:**")
                
                # Show active selections across tabs
                selections_summary = []
                
                if 'performance_vendors' in st.session_state and st.session_state['performance_vendors']:
                    selections_summary.append(f"📊 {len(st.session_state['performance_vendors'])} vendors (Performance)")
                
                if 'consolidation_items' in st.session_state and st.session_state['consolidation_items']:
                    selections_summary.append(f"🔄 {len(st.session_state['consolidation_items'])} items (Consolidation)")
                
                if 'selected_negotiation_vendors' in st.session_state and st.session_state['selected_negotiation_vendors']:
                    selections_summary.append(f"🤝 {len(st.session_state['selected_negotiation_vendors'])} vendors (Negotiation)")
                
                if selections_summary:
                    for summary in selections_summary:
                        st.write(f"• {summary}")
                    
                    # Global clear button
                    if st.button("🗑️ Clear All Selections", key="global_clear_selections"):
                        # Clear all selection states
                        selection_keys = ['performance_vendors', 'consolidation_items', 'selected_negotiation_vendors', 
                                        'calendar_event_filter', 'calendar_priority_filter']
                        for key in selection_keys:
                            if key in st.session_state:
                                if key.endswith('_filter'):
                                    st.session_state[key] = []  # Clear filters
                                else:
                                    st.session_state[key] = []  # Clear selections
                        st.success("✅ All selections cleared!")
                        st.rerun()
                else:
                    st.write("No active selections")
                
                # Quick stats
                if 'enhanced_opportunities' in st.session_state:
                    opportunities_df = st.session_state['enhanced_opportunities']
                    st.write("**Portfolio Quick Stats:**")
                    st.write(f"• Total Opportunities: {len(opportunities_df)}")
                    st.write(f"• High Priority: {len(opportunities_df[opportunities_df['Contract Priority'] == 'High Priority'])}")
                    st.write(f"• Open to Negotiation: {len(opportunities_df[opportunities_df['Negotiation Class'] == 'Open to Negotiation'])}")
            
            # Keyboard shortcuts info
            with st.sidebar.expander("⌨️ Keyboard Tips"):
                st.markdown("""
                **Efficiency Tips:**
                - Use bulk selection buttons to quickly select multiple items
                - Try configuration presets for common analysis scenarios
                - Use priority-based selections for focused analysis
                - Export selected data for offline analysis
                - Check the Quick Actions Summary to track your selections
                """)
        
        if st.sidebar.button("🔄 Reset Currency Conversion"):
            if 'currency_converted' in st.session_state:
                del st.session_state['currency_converted']
            if 'converted_df' in st.session_state:
                del st.session_state['converted_df']
            if 'exchange_rates' in st.session_state:
                del st.session_state['exchange_rates']
            st.rerun()
    
    # Use converted dataframe if available
    if 'currency_converted' in st.session_state and st.session_state['currency_converted']:
        df = st.session_state['converted_df']
        
        # Display conversion summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("💱 Currency Conversion Applied")
        with col2:
            total_records = len(df)
            st.info(f"📊 {total_records:,} records converted")
        with col3:
            st.info("🇸🇦 All amounts in Saudi Riyal (ر.س)")
        
        # Display exchange rates used
        with st.expander("📋 Exchange Rates Applied"):
            if 'exchange_rates' in st.session_state:
                rates = st.session_state.get('exchange_rates', get_default_exchange_rates())
                rate_cols = st.columns(4)
                for i, (currency, rate) in enumerate(rates.items()):
                    with rate_cols[i % 4]:
                        st.write(f"**1 {currency}** = ر.س {rate:.4f}")
    else:
        st.warning("⚠️ Currency conversion not applied. Please configure and apply currency settings in the sidebar.")
    
    # Clean data
    df_clean = df.dropna(subset=required_columns)
    df_clean = df_clean[df_clean['Unit Price'] > 0]
    df_clean = df_clean[df_clean['Qty Delivered'] > 0]
    df_clean['Creation Date'] = pd.to_datetime(df_clean['Creation Date'], errors='coerce')
    df_clean = df_clean.dropna(subset=['Creation Date'])
    
    if len(df_clean) == 0:
        st.warning("No valid data found after cleaning.")
        return
    
    # Enhanced tabs with new features
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🎯 Smart Identification", 
        "🤝 Negotiation Intelligence", 
        "🔄 Multi-Vendor Insights",
        "⚠️ Risk Assessment",
        "🎪 Contract Simulation",
        "📅 Procurement Calendar",
        "👥 Collaboration Hub",
        "📊 Executive Dashboard"
    ])
    
    with tab1:
        st.subheader("🎯 Smart Contract Identification")
        
        # Enhanced configuration with bulk selection helpers
        st.write("### ⚙️ Analysis Configuration")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            min_spend = st.number_input("Min Annual Spend", min_value=0, value=50000, step=10000)
        with col2:
            min_frequency = st.number_input("Min Order Frequency", min_value=1, value=6, step=1)
        with col3:
            analysis_period = st.selectbox("Analysis Period", ["All Data", "Last 12 Months", "Last 6 Months"])
        with col4:
            region_options = ["All Regions"] + list(df_clean.get('W/H', pd.Series()).unique()) if 'W/H' in df_clean.columns else ["All Regions"]
            region_filter = st.selectbox("Region Filter", region_options)
        
        # Quick configuration presets
        st.write("**Quick Configuration Presets:**")
        preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)
        
        with preset_col1:
            if st.button("🎯 High Value Focus", key="preset_high_value"):
                st.session_state['config_min_spend'] = 100000
                st.session_state['config_min_frequency'] = 12
                st.rerun()
        
        with preset_col2:
            if st.button("📊 Standard Analysis", key="preset_standard"):
                st.session_state['config_min_spend'] = 50000
                st.session_state['config_min_frequency'] = 6
                st.rerun()
        
        with preset_col3:
            if st.button("🔍 Comprehensive Scan", key="preset_comprehensive"):
                st.session_state['config_min_spend'] = 10000
                st.session_state['config_min_frequency'] = 3
                st.rerun()
        
        with preset_col4:
            if st.button("⚡ Quick Wins", key="preset_quick_wins"):
                st.session_state['config_min_spend'] = 25000
                st.session_state['config_min_frequency'] = 8
                st.rerun()
        
        # Apply preset values if set
        if 'config_min_spend' in st.session_state:
            min_spend = st.session_state['config_min_spend']
        if 'config_min_frequency' in st.session_state:
            min_frequency = st.session_state['config_min_frequency']
        
        # Filter data
        analysis_df = df_clean.copy()
        
        if analysis_period == "Last 12 Months":
            cutoff_date = df_clean['Creation Date'].max() - timedelta(days=365)
            analysis_df = analysis_df[analysis_df['Creation Date'] >= cutoff_date]
        elif analysis_period == "Last 6 Months":
            cutoff_date = df_clean['Creation Date'].max() - timedelta(days=180)
            analysis_df = analysis_df[analysis_df['Creation Date'] >= cutoff_date]
        
        if region_filter != "All Regions" and 'W/H' in analysis_df.columns:
            analysis_df = analysis_df[analysis_df['W/H'] == region_filter]
        
        if st.button("🚀 Run Smart Analysis", type="primary"):
            with st.spinner("Running advanced contract analysis..."):
                contract_opportunities = []
                
                # Enhanced analysis with new metrics
                vendor_item_combinations = analysis_df.groupby(['Vendor Name', 'Item'])
                
                for (vendor, item), group_data in vendor_item_combinations:
                    try:
                        # Ensure group_data is valid
                        if not isinstance(group_data, pd.DataFrame) or len(group_data) == 0:
                            continue
                            
                        suitability_analysis = analyze_contract_suitability(group_data, min_spend, min_frequency)
                        
                        if suitability_analysis['recommendation'] != "Not Suitable":
                            vendor_performance = calculate_vendor_performance_score(group_data)
                            negotiation_potential = calculate_negotiation_potential_index(group_data)
                            risk_assessment = calculate_supplier_risk_index(group_data)
                            contract_recommendation = recommend_contract_type(
                                suitability_analysis, vendor_performance, negotiation_potential
                            )
                            
                            item_desc = group_data['Item Description'].iloc[0] if 'Item Description' in group_data.columns else f"Item {item}"
                            
                            contract_opportunities.append({
                                'Vendor Name': vendor,
                                'Item': item,
                                'Item Description': item_desc[:40] + "..." if len(str(item_desc)) > 40 else str(item_desc),
                                'Annual Spend': suitability_analysis['total_spend'],
                                'Order Frequency': suitability_analysis['order_frequency'],
                                'Demand Predictability': suitability_analysis['demand_predictability'],
                                'Vendor Performance': vendor_performance['overall_score'],
                                'Negotiation Potential': negotiation_potential['negotiation_index'],
                                'Negotiation Class': negotiation_potential['negotiation_class'],
                                'Risk Level': risk_assessment['risk_class'],
                                'Risk Score': risk_assessment['overall_risk'],
                                'Recommended Contract': contract_recommendation['type'],
                                'Contract Reasoning': contract_recommendation['reasoning'],
                                'Suitability Score': suitability_analysis['suitability_score'],
                                'Contract Priority': suitability_analysis['recommendation'],
                                'Avg Unit Price': group_data['Unit Price'].mean(),
                                'Price Stability': vendor_performance['price_stability']
                            })
                    
                    except Exception as e:
                        # Log error and continue with next vendor-item combination
                        st.warning(f"Error processing {vendor} - Item {item}: {str(e)}")
                        continue
                
                if contract_opportunities:
                    opportunities_df = pd.DataFrame(contract_opportunities)
                    opportunities_df = opportunities_df.sort_values(['Suitability Score', 'Annual Spend'], ascending=[False, False])
                    
                    # Enhanced summary metrics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Total Opportunities", len(opportunities_df))
                    with col2:
                        high_priority = len(opportunities_df[opportunities_df['Contract Priority'] == 'High Priority'])
                        st.metric("High Priority", high_priority)
                    with col3:
                        total_spend = opportunities_df['Annual Spend'].sum()
                        st.metric("Total Spend", format_sar_amount(total_spend))
                    with col4:
                        open_negotiations = len(opportunities_df[opportunities_df['Negotiation Class'] == 'Open to Negotiation'])
                        st.metric("Open to Negotiation", open_negotiations)
                    with col5:
                        low_risk = len(opportunities_df[opportunities_df['Risk Level'] == 'Low Risk'])
                        st.metric("Low Risk Vendors", low_risk)
                    
                    # Enhanced visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Contract type recommendations
                        contract_types = opportunities_df['Recommended Contract'].value_counts()
                        fig = px.pie(values=contract_types.values, names=contract_types.index,
                                    title="Recommended Contract Types")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Risk vs Negotiation potential matrix
                        fig = px.scatter(opportunities_df, 
                                       x='Negotiation Potential', y='Risk Score',
                                       size='Annual Spend', color='Contract Priority',
                                       hover_data=['Vendor Name', 'Recommended Contract'],
                                       title="Risk vs Negotiation Potential Matrix",
                                       labels={'Annual Spend': 'Annual Spend (SAR)'})
                        
                        # Update hover template to show SAR formatting
                        fig.update_traces(
                            hovertemplate="<b>%{customdata[0]}</b><br>" +
                                        "Negotiation Potential: %{x:.2f}<br>" +
                                        "Risk Score: %{y:.2f}<br>" +
                                        "Annual Spend: ر.س %{marker.size:,.0f}<br>" +
                                        "Contract: %{customdata[1]}<br>" +
                                        "<extra></extra>"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Enhanced results table
                    st.subheader("📊 Enhanced Contract Analysis Results")
                    
                    display_df = opportunities_df[[
                        'Vendor Name', 'Item Description', 'Annual Spend', 'Contract Priority',
                        'Recommended Contract', 'Negotiation Class', 'Risk Level',
                        'Suitability Score', 'Vendor Performance'
                    ]]
                    
                    st.dataframe(
                        display_df.style.format({
                            'Annual Spend': lambda x: format_sar_amount(x),
                            'Suitability Score': '{:.2f}',
                            'Vendor Performance': '{:.2f}'
                        }),
                        use_container_width=True
                    )
                    
                    # Store enhanced results
                    st.session_state['enhanced_opportunities'] = opportunities_df
                    
                    # Export enhanced results
                    csv = opportunities_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Export Enhanced Analysis",
                        data=csv,
                        file_name=f"enhanced_contract_opportunities_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                else:
                    st.info("No contract opportunities found with current criteria.")
    
    with tab2:
        st.subheader("🤝 Negotiation Intelligence Dashboard")
        
        if 'enhanced_opportunities' in st.session_state:
            opportunities_df = st.session_state['enhanced_opportunities']
            
            # Negotiation potential analysis
            st.write("### 📈 Vendor Negotiation Profiles")
            
            nego_summary = opportunities_df.groupby('Negotiation Class').agg({
                'Annual Spend': 'sum',
                'Vendor Name': 'count'
            }).reset_index()
            nego_summary.columns = ['Negotiation Class', 'Total Spend', 'Vendor Count']
            
            col1, col2, col3 = st.columns(3)
            for i, (_, row) in enumerate(nego_summary.iterrows()):
                with [col1, col2, col3][i]:
                    color = "🟢" if row['Negotiation Class'] == "Open to Negotiation" else "🟡" if row['Negotiation Class'] == "Moderate" else "🔴"
                    st.metric(
                        f"{color} {row['Negotiation Class']}", 
                        f"{row['Vendor Count']} vendors",
                        f"${row['Total Spend']:,.0f}"
                    )
            
            # Detailed negotiation insights
            st.write("### 🎯 Priority Negotiation Targets")
            
            high_value_negotiations = opportunities_df[
                (opportunities_df['Negotiation Class'].isin(['Open to Negotiation', 'Moderate'])) &
                (opportunities_df['Annual Spend'] > opportunities_df['Annual Spend'].quantile(0.7))
            ].sort_values('Annual Spend', ascending=False)
            
            if len(high_value_negotiations) > 0:
                
                # Bulk selection for negotiation targets
                st.write("**Bulk Selection for Negotiation Planning:**")
                nego_col1, nego_col2, nego_col3, nego_col4 = st.columns(4)
                
                with nego_col1:
                    if st.button("✅ Select All Open", key="select_all_open_nego"):
                        open_vendors = opportunities_df[opportunities_df['Negotiation Class'] == 'Open to Negotiation']['Vendor Name'].tolist()
                        if 'selected_negotiation_vendors' not in st.session_state:
                            st.session_state['selected_negotiation_vendors'] = []
                        st.session_state['selected_negotiation_vendors'].extend(open_vendors)
                        st.session_state['selected_negotiation_vendors'] = list(set(st.session_state['selected_negotiation_vendors']))  # Remove duplicates
                        st.rerun()
                
                with nego_col2:
                    if st.button("🎯 Top 5 Spenders", key="select_top_5_spenders"):
                        top_spenders = opportunities_df.nlargest(5, 'Annual Spend')['Vendor Name'].tolist()
                        st.session_state['selected_negotiation_vendors'] = top_spenders
                        st.rerun()
                
                with nego_col3:
                    if st.button("⚡ Quick Wins", key="select_quick_wins_nego"):
                        quick_wins = opportunities_df[
                            (opportunities_df['Negotiation Class'] == 'Open to Negotiation') & 
                            (opportunities_df['Risk Level'] == 'Low Risk')
                        ]['Vendor Name'].tolist()
                        st.session_state['selected_negotiation_vendors'] = quick_wins
                        st.rerun()
                
                with nego_col4:
                    if st.button("❌ Clear Selection", key="clear_negotiation_selection"):
                        st.session_state['selected_negotiation_vendors'] = []
                        st.rerun()
                
                # Display selected vendors for batch operations
                if 'selected_negotiation_vendors' in st.session_state and st.session_state['selected_negotiation_vendors']:
                    st.success(f"📋 **Selected for Negotiation:** {len(st.session_state['selected_negotiation_vendors'])} vendors")
                    
                    # Bulk actions for selected vendors
                    batch_col1, batch_col2, batch_col3 = st.columns(3)
                    with batch_col1:
                        if st.button("📊 Generate Batch Report", key="batch_nego_report"):
                            selected_vendor_data = opportunities_df[opportunities_df['Vendor Name'].isin(st.session_state['selected_negotiation_vendors'])]
                            total_batch_spend = selected_vendor_data['Annual Spend'].sum()
                            avg_negotiation_score = selected_vendor_data['Negotiation Potential'].mean()
                            
                            st.info(f"""
                            **Batch Analysis Results:**
                            • Total Spend: {format_sar_amount(total_batch_spend)}
                            • Average Negotiation Score: {avg_negotiation_score:.2f}
                            • Vendors Selected: {len(st.session_state['selected_negotiation_vendors'])}
                            • Estimated Savings Potential: {format_sar_amount(total_batch_spend * 0.06)}
                            """)
                    
                    with batch_col2:
                        if st.button("📅 Create Negotiation Schedule", key="create_nego_schedule"):
                            st.success("📅 Negotiation schedule created for selected vendors!")
                            st.write("**Suggested Schedule:**")
                            for i, vendor in enumerate(st.session_state['selected_negotiation_vendors'][:5]):  # Show first 5
                                week = i + 1
                                st.write(f"• Week {week}: {vendor}")
                    
                    with batch_col3:
                        if st.button("📄 Export Vendor List", key="export_nego_vendors"):
                            selected_data = opportunities_df[opportunities_df['Vendor Name'].isin(st.session_state['selected_negotiation_vendors'])]
                            csv = selected_data.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Selected Vendors",
                                data=csv,
                                file_name=f"selected_negotiation_vendors_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                for _, vendor in high_value_negotiations.head(5).iterrows():
                    with st.expander(f"🎯 {vendor['Vendor Name']} - ${vendor['Annual Spend']:,.0f}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Negotiation Class:** {vendor['Negotiation Class']}")
                            st.write(f"**Risk Level:** {vendor['Risk Level']}")
                            st.write(f"**Recommended Contract:** {vendor['Recommended Contract']}")
                            st.write(f"**Performance Score:** {vendor['Vendor Performance']:.2f}")
                        
                        with col2:
                            st.write("**Negotiation Strategy:**")
                            if vendor['Negotiation Class'] == 'Open to Negotiation':
                                st.write("• Leverage high spend for volume discounts")
                                st.write("• Propose multi-year agreements")
                                st.write("• Request price freeze or escalation caps")
                            else:
                                st.write("• Focus on service improvements")
                                st.write("• Negotiate payment terms")
                                st.write("• Explore value-added services")
                        
                        # Display annual spend in SAR
                        st.write(f"**Annual Spend:** {format_sar_amount(vendor['Annual Spend'])}")
                        
                        if vendor['Negotiation Class'] == 'Open to Negotiation':
                            potential_savings = vendor['Annual Spend'] * 0.08  # 8% potential savings
                            st.write(f"**Potential Savings:** {format_sar_amount(potential_savings)}")
                        else:
                            potential_savings = vendor['Annual Spend'] * 0.03  # 3% potential savings
                            st.write(f"**Potential Savings:** {format_sar_amount(potential_savings)}")
            
            # Negotiation timeline planner
            st.write("### 📅 Negotiation Timeline Planner")
            
            timeline_df = high_value_negotiations.head(10).copy()
            timeline_df['Negotiation Start'] = pd.Timestamp.now() + pd.to_timedelta(range(len(timeline_df)), unit='W')
            timeline_df['Negotiation End'] = timeline_df['Negotiation Start'] + pd.Timedelta(weeks=4)
            
            fig = px.timeline(
                timeline_df,
                x_start='Negotiation Start',
                x_end='Negotiation End',
                y='Vendor Name',
                color='Negotiation Class',
                title="Proposed Negotiation Schedule"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Run Smart Identification first to see negotiation intelligence.")
    
    with tab3:
        st.subheader("🔄 Multi-Vendor Consolidation Insights")
        
        # Analyze consolidation opportunities
        consolidation_analysis = analyze_multi_vendor_aggregation(df_clean)
        
        if len(consolidation_analysis) > 0:
            # Summary metrics
            total_consolidation_savings = consolidation_analysis['Potential Savings'].sum()
            high_priority_consolidations = len(consolidation_analysis[consolidation_analysis['Consolidation Priority'] == 'High'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Consolidation Opportunities", len(consolidation_analysis))
            with col2:
                st.metric("High Priority Items", high_priority_consolidations)
            with col3:
                st.metric("Total Savings Potential", format_sar_amount(total_consolidation_savings))
            
            # Consolidation visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Top consolidation opportunities
                top_consolidations = consolidation_analysis.nlargest(10, 'Potential Savings')
                fig = px.bar(top_consolidations, x='Potential Savings', y='Item',
                           orientation='h', title="Top Consolidation Opportunities")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Vendor distribution analysis
                vendor_counts = consolidation_analysis['Vendor Count'].value_counts().sort_index()
                fig = px.bar(x=vendor_counts.index, y=vendor_counts.values,
                           title="Items by Number of Vendors",
                           labels={'x': 'Number of Vendors', 'y': 'Number of Items'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed consolidation table
            st.write("### 📊 Consolidation Opportunities Details")
            
            st.dataframe(
                consolidation_analysis.style.format({
                    'Total Spend': lambda x: format_sar_amount(x),
                    'Primary Vendor Share': '{:.1f}%',
                    'Secondary Spend': lambda x: format_sar_amount(x),
                    'Potential Savings': lambda x: format_sar_amount(x)
                }),
                use_container_width=True
            )
            
            # Interactive consolidation planner with bulk selection
            st.write("### 🎯 Consolidation Action Planner")
            
            # Bulk selection buttons for consolidation
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write("**Select items for consolidation planning:**")
            with col2:
                if st.button("✅ Select All Items", key="select_all_consolidation"):
                    st.session_state['consolidation_items'] = consolidation_analysis['Item'].tolist()
                    st.rerun()
            with col3:
                if st.button("❌ Deselect All", key="deselect_all_consolidation"):
                    st.session_state['consolidation_items'] = []
                    st.rerun()
            
            # Quick selection buttons for priorities
            priority_col1, priority_col2, priority_col3 = st.columns(3)
            with priority_col1:
                if st.button("🔴 High Priority Only", key="select_high_priority"):
                    high_priority_items = consolidation_analysis[consolidation_analysis['Consolidation Priority'] == 'High']['Item'].tolist()
                    st.session_state['consolidation_items'] = high_priority_items
                    st.rerun()
            with priority_col2:
                if st.button("🟡 Medium+ Priority", key="select_medium_priority"):
                    medium_plus_items = consolidation_analysis[consolidation_analysis['Consolidation Priority'].isin(['High', 'Medium'])]['Item'].tolist()
                    st.session_state['consolidation_items'] = medium_plus_items
                    st.rerun()
            with priority_col3:
                if st.button("💰 Top 10 Savings", key="select_top_savings"):
                    top_savings_items = consolidation_analysis.nlargest(10, 'Potential Savings')['Item'].tolist()
                    st.session_state['consolidation_items'] = top_savings_items
                    st.rerun()
            
            # Get current selection from session state
            current_consolidation_selection = st.session_state.get('consolidation_items', 
                consolidation_analysis.nlargest(3, 'Potential Savings')['Item'].tolist())
            
            selected_items = st.multiselect(
                "Selected consolidation items:",
                consolidation_analysis['Item'].tolist(),
                default=current_consolidation_selection,
                key="consolidation_items"
            )
            
            if selected_items:
                # Show selection status
                st.info(f"🎯 **Selected:** {len(selected_items)} of {len(consolidation_analysis)} items for consolidation planning")
                
                consolidation_plan = consolidation_analysis[consolidation_analysis['Item'].isin(selected_items)]
                total_plan_savings = consolidation_plan['Potential Savings'].sum()
                
                st.success(f"💰 Selected consolidation plan potential savings: {format_sar_amount(total_plan_savings)}")
                
                # Implementation timeline
                st.write("**Recommended Implementation Timeline:**")
                for i, (_, item) in enumerate(consolidation_plan.iterrows()):
                    phase = f"Phase {i+1}"
                    timeline = f"Month {i*2+1}-{i*2+2}"
                    st.write(f"• {phase} ({timeline}): Consolidate {item['Item']} with {item['Primary Vendor']} - {format_sar_amount(item['Potential Savings'])} savings")
            else:
                st.warning("⚠️ No items selected for consolidation planning. Please select items above.")
        
        else:
            st.info("No multi-vendor opportunities found in current dataset.")
    
    with tab4:
        st.subheader("⚠️ Comprehensive Risk Assessment")
        
        if 'enhanced_opportunities' in st.session_state:
            opportunities_df = st.session_state['enhanced_opportunities']
            
            # Risk dashboard
            st.write("### 🎛️ Risk Dashboard")
            
            risk_summary = opportunities_df.groupby('Risk Level').agg({
                'Annual Spend': 'sum',
                'Vendor Name': 'count'
            }).reset_index()
            
            col1, col2, col3 = st.columns(3)
            for i, (_, row) in enumerate(risk_summary.iterrows()):
                risk_color = "🟢" if row['Risk Level'] == "Low Risk" else "🟡" if row['Risk Level'] == "Medium Risk" else "🔴"
                with [col1, col2, col3][i % 3]:
                    st.metric(
                        f"{risk_color} {row['Risk Level']}", 
                        f"{row['Vendor Name']} vendors",
                        format_sar_amount(row['Annual Spend'])
                    )
            
            # Risk matrix visualization
            fig = px.scatter(opportunities_df, 
                           x='Vendor Performance', y='Risk Score',
                           size='Annual Spend', color='Risk Level',
                           hover_data=['Vendor Name', 'Negotiation Class'],
                           title="Vendor Risk vs Performance Matrix")
            st.plotly_chart(fig, use_container_width=True)
            
            # High risk vendor analysis
            st.write("### 🚨 High Risk Vendor Focus")
            
            high_risk_vendors = opportunities_df[opportunities_df['Risk Level'] == 'High Risk']
            
            if len(high_risk_vendors) > 0:
                st.warning(f"⚠️ {len(high_risk_vendors)} high-risk vendors identified requiring immediate attention:")
                
                for _, vendor in high_risk_vendors.iterrows():
                    with st.expander(f"🚨 {vendor['Vendor Name']} - Risk Score: {vendor['Risk Score']:.2f}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Risk Factors:**")
                            st.write(f"• Annual Spend: {format_sar_amount(vendor['Annual Spend'])}")
                            st.write(f"• Performance Score: {vendor['Vendor Performance']:.2f}")
                            st.write(f"• Contract Priority: {vendor['Contract Priority']}")
                        
                        with col2:
                            st.write("**Mitigation Actions:**")
                            st.write("• Develop backup supplier relationships")
                            st.write("• Implement performance monitoring")
                            st.write("• Consider shorter contract terms")
                            st.write("• Increase inventory buffers")
            else:
                st.success("✅ No high-risk vendors identified in current portfolio!")
            
            # Risk mitigation recommendations
            st.write("### 💡 Risk Mitigation Recommendations")
            
            risk_recommendations = {
                'Portfolio Diversification': f"Reduce concentration - top vendor represents {opportunities_df.nlargest(1, 'Annual Spend')['Annual Spend'].iloc[0] / opportunities_df['Annual Spend'].sum() * 100:.1f}% of spend",
                'Performance Monitoring': f"Implement SLAs for {len(opportunities_df[opportunities_df['Vendor Performance'] < 0.7])} underperforming vendors",
                'Backup Suppliers': f"Develop alternatives for {len(opportunities_df[opportunities_df['Risk Level'].isin(['High Risk', 'Medium Risk'])])} risky vendors",
                'Contract Terms': "Include force majeure and business continuity clauses"
            }
            
            for recommendation, description in risk_recommendations.items():
                st.write(f"**{recommendation}:** {description}")
        
        else:
            st.info("Run Smart Identification first to see risk assessment.")
    
    with tab5:
        st.subheader("🎪 Multi-Year Contract Simulation")
        
        if 'enhanced_opportunities' in st.session_state:
            opportunities_df = st.session_state['enhanced_opportunities']
            
            # Contract simulation interface
            st.write("### ⚙️ Simulation Parameters")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                simulation_years = st.selectbox("Simulation Period", [1, 2, 3, 5], index=2)
            with col2:
                volume_growth = st.slider("Annual Volume Growth (%)", 0, 20, 5) / 100
            with col3:
                inflation_rate = st.slider("Annual Inflation (%)", 0, 10, 3) / 100
            with col4:
                contract_escalation = st.slider("Contract Escalation (%)", 0, 5, 2) / 100
            
            # Select vendor for simulation with bulk actions
            vendor_options = opportunities_df['Vendor Name'].unique()
            
            # Vendor selection interface
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write("**Select Vendor for Multi-Year Simulation:**")
            with col2:
                if st.button("🎯 Highest Spend", key="select_highest_spend_vendor"):
                    highest_spend_vendor = opportunities_df.loc[opportunities_df['Annual Spend'].idxmax(), 'Vendor Name']
                    st.session_state['simulation_vendor'] = highest_spend_vendor
                    st.rerun()
            with col3:
                if st.button("🤝 Best Negotiable", key="select_best_negotiable_vendor"):
                    negotiable_vendors = opportunities_df[opportunities_df['Negotiation Class'] == 'Open to Negotiation']
                    if len(negotiable_vendors) > 0:
                        best_negotiable = negotiable_vendors.loc[negotiable_vendors['Annual Spend'].idxmax(), 'Vendor Name']
                        st.session_state['simulation_vendor'] = best_negotiable
                    else:
                        st.session_state['simulation_vendor'] = vendor_options[0]
                    st.rerun()
            
            # Get current selection
            current_simulation_vendor = st.session_state.get('simulation_vendor', vendor_options[0])
            selected_vendor = st.selectbox(
                "Vendor for detailed simulation:",
                vendor_options,
                index=list(vendor_options).index(current_simulation_vendor) if current_simulation_vendor in vendor_options else 0,
                key="simulation_vendor"
            )
            
            if selected_vendor:
                vendor_opportunities = opportunities_df[opportunities_df['Vendor Name'] == selected_vendor]
                
                # Contract terms configuration
                st.write("### 📋 Contract Terms Configuration")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    volume_discount = st.slider("Volume Discount (%)", 0, 25, 8) / 100
                with col2:
                    setup_cost = st.number_input("Contract Setup Cost", 0, 50000, 10000)
                with col3:
                    admin_savings_monthly = st.number_input("Monthly Admin Savings", 0, 2000, 500)
                
                # Run simulation
                if st.button("🚀 Run Multi-Year Simulation"):
                    
                    # Get historical data for selected vendor
                    vendor_historical = df_clean[df_clean['Vendor Name'] == selected_vendor]
                    
                    contract_terms = {
                        'volume_discount': volume_discount,
                        'setup_cost': setup_cost,
                        'admin_savings': admin_savings_monthly
                    }
                    
                    # Run simulation (simplified version for demo)
                    simulation_data = []
                    cumulative_savings = 0
                    
                    base_annual_spend = vendor_opportunities['Annual Spend'].sum()
                    
                    for year in range(1, simulation_years + 1):
                        # Project volumes and costs
                        projected_volume_factor = (1 + volume_growth) ** year
                        inflation_factor = (1 + inflation_rate) ** year
                        contract_escalation_factor = (1 + contract_escalation) ** year
                        
                        # No-contract scenario
                        no_contract_cost = base_annual_spend * projected_volume_factor * inflation_factor
                        
                        # Contract scenario
                        contract_price_factor = (1 - volume_discount) * contract_escalation_factor
                        contract_cost = base_annual_spend * projected_volume_factor * contract_price_factor
                        
                        # Annual savings
                        price_savings = no_contract_cost - contract_cost
                        admin_savings = admin_savings_monthly * 12
                        total_annual_savings = price_savings + admin_savings
                        
                        cumulative_savings += total_annual_savings
                        
                        # ROI calculation
                        roi = ((cumulative_savings - setup_cost) / setup_cost) * 100 if setup_cost > 0 else 0
                        
                        simulation_data.append({
                            'Year': year,
                            'No Contract Cost': no_contract_cost,
                            'Contract Cost': contract_cost,
                            'Annual Savings': total_annual_savings,
                            'Cumulative Savings': cumulative_savings,
                            'ROI %': roi
                        })
                    
                    simulation_df = pd.DataFrame(simulation_data)
                    
                    # Display results
                    st.write("### 📊 Simulation Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        total_savings = simulation_df['Cumulative Savings'].iloc[-1]
                        st.metric("Total Savings", format_sar_amount(total_savings))
                    with col2:
                        final_roi = simulation_df['ROI %'].iloc[-1]
                        st.metric("Final ROI", f"{final_roi:.1f}%")
                    with col3:
                        payback_period = setup_cost / simulation_df['Annual Savings'].mean() if simulation_df['Annual Savings'].mean() > 0 else 0
                        st.metric("Payback Period", f"{payback_period:.1f} years")
                    
                    # Visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.line(simulation_df, x='Year', y=['No Contract Cost', 'Contract Cost'],
                                     title="Cost Comparison Over Time")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.bar(simulation_df, x='Year', y='Annual Savings',
                                   title="Annual Savings by Year")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed table
                    st.dataframe(
                        simulation_df.style.format({
                            'No Contract Cost': lambda x: format_sar_amount(x),
                            'Contract Cost': lambda x: format_sar_amount(x),
                            'Annual Savings': lambda x: format_sar_amount(x),
                            'Cumulative Savings': lambda x: format_sar_amount(x),
                            'ROI %': '{:.1f}%'
                        }),
                        use_container_width=True
                    )
        
        else:
            st.info("Run Smart Identification first to access simulation features.")
    
    with tab6:
        st.subheader("📅 Interactive Procurement Calendar")
        
        if 'enhanced_opportunities' in st.session_state:
            opportunities_df = st.session_state['enhanced_opportunities']
            
            # Generate calendar events
            calendar_events = generate_procurement_calendar(opportunities_df)
            
            # Calendar view options with bulk selection
            col1, col2, col3 = st.columns(3)
            with col1:
                view_type = st.selectbox("Calendar View", ["Timeline", "Monthly", "Quarterly"])
            with col2:
                st.write("**Event Types:**")
                
                # Bulk buttons for event types
                event_col1, event_col2 = st.columns(2)
                with event_col1:
                    if st.button("✅ All Events", key="select_all_events"):
                        st.session_state['calendar_event_filter'] = ["Contract Start", "Renewal Notice", "Contract Expiry"]
                        st.rerun()
                with event_col2:
                    if st.button("❌ Clear Events", key="clear_all_events"):
                        st.session_state['calendar_event_filter'] = []
                        st.rerun()
                
                current_event_selection = st.session_state.get('calendar_event_filter', ["Contract Start", "Renewal Notice", "Contract Expiry"])
                event_filter = st.multiselect("Select Event Types", 
                                            ["Contract Start", "Renewal Notice", "Contract Expiry"],
                                            default=current_event_selection,
                                            key="calendar_event_filter")
            with col3:
                st.write("**Priority Levels:**")
                
                # Bulk buttons for priority filter
                priority_col1, priority_col2 = st.columns(2)
                with priority_col1:
                    if st.button("✅ All Priorities", key="select_all_priorities"):
                        st.session_state['calendar_priority_filter'] = ["High Priority", "Medium Priority", "Low Priority"]
                        st.rerun()
                with priority_col2:
                    if st.button("🔴 High Only", key="select_high_only"):
                        st.session_state['calendar_priority_filter'] = ["High Priority"]
                        st.rerun()
                
                current_priority_selection = st.session_state.get('calendar_priority_filter', ["High Priority", "Medium Priority"])
                priority_filter = st.multiselect("Select Priority Levels",
                                                ["High Priority", "Medium Priority", "Low Priority"],
                                                default=current_priority_selection,
                                                key="calendar_priority_filter")
            
            # Filter events
            filtered_events = calendar_events[
                (calendar_events['Type'].isin(event_filter)) &
                (calendar_events['Priority'].isin(priority_filter))
            ]
            
            if view_type == "Timeline":
                # Timeline visualization
                fig = px.timeline(
                    filtered_events,
                    x_start='Date',
                    x_end='Date',
                    y='Vendor',
                    color='Type',
                    title="Procurement Timeline",
                    hover_data=['Event', 'Description']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            elif view_type == "Monthly":
                # Monthly calendar view
                current_month = datetime.now().month
                current_year = datetime.now().year
                
                month_events = filtered_events[
                    (filtered_events['Date'].dt.month == current_month) &
                    (filtered_events['Date'].dt.year == current_year)
                ]
                
                if len(month_events) > 0:
                    st.write(f"### 📅 {calendar.month_name[current_month]} {current_year} Events")
                    
                    for _, event in month_events.iterrows():
                        event_color = "🟢" if event['Type'] == "Contract Start" else "🟡" if event['Type'] == "Renewal Notice" else "🔴"
                        st.write(f"{event_color} **{event['Date'].strftime('%d')}**: {event['Event']} - {event['Description']}")
                else:
                    st.info(f"No events scheduled for {calendar.month_name[current_month]} {current_year}")
            
            # Upcoming events summary
            st.write("### ⏰ Upcoming Events (Next 30 Days)")
            
            next_30_days = datetime.now() + timedelta(days=30)
            upcoming_events = filtered_events[
                (filtered_events['Date'] >= datetime.now()) &
                (filtered_events['Date'] <= next_30_days)
            ].sort_values('Date')
            
            if len(upcoming_events) > 0:
                for _, event in upcoming_events.iterrows():
                    days_until = (event['Date'] - datetime.now()).days
                    priority_color = "🔴" if days_until <= 7 else "🟡" if days_until <= 14 else "🟢"
                    
                    st.write(f"{priority_color} **{event['Date'].strftime('%Y-%m-%d')}** ({days_until} days): {event['Event']}")
            else:
                st.success("✅ No critical events in the next 30 days!")
            
            # Export calendar
            if st.button("📥 Export Calendar Events"):
                csv = calendar_events.to_csv(index=False)
                st.download_button(
                    label="Download Calendar CSV",
                    data=csv,
                    file_name=f"procurement_calendar_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        else:
            st.info("Run Smart Identification first to generate procurement calendar.")
    
    with tab7:
        st.subheader("👥 Stakeholder Collaboration Hub")
        
        # Collaboration features
        st.write("### 💬 Contract Collaboration Center")
        
        if 'enhanced_opportunities' in st.session_state:
            opportunities_df = st.session_state['enhanced_opportunities']
            
            # Contract selection for collaboration with quick selection
            contract_options = [f"{row['Vendor Name']} - {row['Item Description']}" for _, row in opportunities_df.iterrows()]
            
            # Quick selection buttons
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write("**Select Contract for Collaboration:**")
            with col2:
                if st.button("🔴 Highest Priority", key="select_highest_priority_contract"):
                    high_priority = opportunities_df[opportunities_df['Contract Priority'] == 'High Priority']
                    if len(high_priority) > 0:
                        highest_priority_contract = f"{high_priority.iloc[0]['Vendor Name']} - {high_priority.iloc[0]['Item Description']}"
                        st.session_state['collaboration_contract'] = highest_priority_contract
                        st.rerun()
            with col3:
                if st.button("💰 Highest Spend", key="select_highest_spend_contract"):
                    highest_spend = opportunities_df.loc[opportunities_df['Annual Spend'].idxmax()]
                    highest_spend_contract = f"{highest_spend['Vendor Name']} - {highest_spend['Item Description']}"
                    st.session_state['collaboration_contract'] = highest_spend_contract
                    st.rerun()
            
            # Get current selection
            current_collaboration_contract = st.session_state.get('collaboration_contract', contract_options[0])
            selected_contract = st.selectbox(
                "Contract for collaboration:",
                contract_options,
                index=contract_options.index(current_collaboration_contract) if current_collaboration_contract in contract_options else 0,
                key="collaboration_contract"
            )
            
            if selected_contract:
                vendor_name = selected_contract.split(' - ')[0]
                
                # Initialize session state for comments if not exists
                if 'contract_comments' not in st.session_state:
                    st.session_state.contract_comments = {}
                
                if selected_contract not in st.session_state.contract_comments:
                    st.session_state.contract_comments[selected_contract] = []
                
                # Comment system
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    new_comment = st.text_area("Add Comment/Note", placeholder="Add your thoughts, concerns, or recommendations...")
                with col2:
                    st.write("**Stakeholders:**")
                    stakeholder = st.selectbox("Your Role", ["Procurement Manager", "Legal Review", "Finance Approval", "Category Manager", "Supplier Manager"])
                    priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
                
                if st.button("💬 Add Comment"):
                    if new_comment:
                        comment_data = {
                            'timestamp': datetime.now(),
                            'stakeholder': stakeholder,
                            'priority': priority,
                            'comment': new_comment
                        }
                        st.session_state.contract_comments[selected_contract].append(comment_data)
                        st.success("Comment added successfully!")
                
                # Display existing comments
                st.write("### 📋 Collaboration History")
                
                if st.session_state.contract_comments[selected_contract]:
                    for comment in reversed(st.session_state.contract_comments[selected_contract]):
                        priority_color = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}[comment['priority']]
                        
                        with st.expander(f"{priority_color} {comment['stakeholder']} - {comment['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                            st.write(comment['comment'])
                else:
                    st.info("No comments yet. Be the first to add feedback!")
                
                # Approval workflow
                st.write("### ✅ Approval Workflow")
                
                workflow_stages = [
                    {"Stage": "Procurement Review", "Status": "Complete", "Assignee": "Procurement Manager"},
                    {"Stage": "Legal Review", "Status": "In Progress", "Assignee": "Legal Team"},
                    {"Stage": "Finance Approval", "Status": "Pending", "Assignee": "Finance Director"},
                    {"Stage": "Final Approval", "Status": "Pending", "Assignee": "CPO"}
                ]
                
                for stage in workflow_stages:
                    status_color = "✅" if stage["Status"] == "Complete" else "🔄" if stage["Status"] == "In Progress" else "⏳"
                    st.write(f"{status_color} **{stage['Stage']}** - {stage['Status']} (Assignee: {stage['Assignee']})")
                
                # Document upload simulation
                st.write("### 📎 Document Management")
                
                uploaded_files = st.file_uploader(
                    "Upload Contract Documents",
                    accept_multiple_files=True,
                    type=['pdf', 'docx', 'xlsx'],
                    help="Upload RFPs, proposals, comparison sheets, etc."
                )
                
                if uploaded_files:
                    st.success(f"✅ {len(uploaded_files)} document(s) uploaded successfully!")
                    for file in uploaded_files:
                        st.write(f"📄 {file.name}")
        
        else:
            st.info("Run Smart Identification first to access collaboration features.")
    
    with tab8:
        st.subheader("📊 Executive Dashboard")
        
        if 'enhanced_opportunities' in st.session_state:
            opportunities_df = st.session_state['enhanced_opportunities']
            
            # Executive summary metrics
            st.write("### 🎯 Executive Summary")
            
            total_spend = opportunities_df['Annual Spend'].sum()
            high_priority_spend = opportunities_df[opportunities_df['Contract Priority'] == 'High Priority']['Annual Spend'].sum()
            potential_savings = total_spend * 0.08  # Assume 8% average savings
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Contract Spend", format_sar_amount(total_spend))
            with col2:
                st.metric("High Priority Spend", format_sar_amount(high_priority_spend))
            with col3:
                st.metric("Potential Annual Savings", format_sar_amount(potential_savings))
            with col4:
                savings_percentage = (potential_savings / total_spend) * 100
                st.metric("Savings %", f"{savings_percentage:.1f}%")
            
            # Strategic overview charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Contract priority distribution by spend
                priority_spend = opportunities_df.groupby('Contract Priority')['Annual Spend'].sum().reset_index()
                fig = px.pie(priority_spend, values='Annual Spend', names='Contract Priority',
                           title="Spend Distribution by Priority")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Risk vs Spend analysis
                fig = px.scatter(opportunities_df, x='Risk Score', y='Annual Spend',
                               color='Contract Priority', size='Vendor Performance',
                               title="Risk vs Spend Portfolio View",
                               labels={'Risk Score': 'Risk Score', 'Annual Spend': 'Annual Spend ($)'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Key insights and recommendations
            st.write("### 💡 Strategic Insights & Recommendations")
            
            insights = []
            
            # Generate insights
            if len(opportunities_df[opportunities_df['Negotiation Class'] == 'Open to Negotiation']) > 0:
                open_nego_spend = opportunities_df[opportunities_df['Negotiation Class'] == 'Open to Negotiation']['Annual Spend'].sum()
                insights.append(f"🎯 **Immediate Action**: {format_sar_amount(open_nego_spend)} in spend with vendors open to negotiation")
            
            if len(opportunities_df[opportunities_df['Risk Level'] == 'High Risk']) > 0:
                high_risk_count = len(opportunities_df[opportunities_df['Risk Level'] == 'High Risk'])
                insights.append(f"⚠️ **Risk Mitigation**: {high_risk_count} high-risk vendors require backup supplier development")
            
            contract_types = opportunities_df['Recommended Contract'].value_counts()
            if len(contract_types) > 0:
                top_contract_type = contract_types.index[0]
                insights.append(f"📋 **Contract Strategy**: {top_contract_type} is recommended for most opportunities")
            
            # Display insights
            for insight in insights:
                st.write(insight)
            
            # Implementation roadmap summary
            st.write("### 🗺️ Implementation Roadmap")
            
            roadmap_summary = {
                "Q1 2024": {"Contracts": 8, "Spend": high_priority_spend * 0.4, "Focus": "High Priority Negotiations"},
                "Q2 2024": {"Contracts": 6, "Spend": high_priority_spend * 0.3, "Focus": "Risk Mitigation"},
                "Q3 2024": {"Contracts": 4, "Spend": high_priority_spend * 0.2, "Focus": "Consolidation"},
                "Q4 2024": {"Contracts": 3, "Spend": high_priority_spend * 0.1, "Focus": "Optimization"}
            }
            
            roadmap_df = pd.DataFrame(roadmap_summary).T.reset_index()
            roadmap_df.columns = ['Quarter', 'Contracts', 'Spend', 'Focus']
            
            fig = px.bar(roadmap_df, x='Quarter', y='Spend', text='Contracts',
                        title="Quarterly Implementation Plan")
            fig.update_traces(texttemplate='%{text} contracts', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
            
            # Export executive summary
            if st.button("📥 Export Executive Summary"):
                summary_data = {
                    'Metric': ['Total Contract Spend', 'High Priority Spend', 'Potential Annual Savings', 'Savings Percentage'],
                    'Value': [format_sar_amount(total_spend), format_sar_amount(high_priority_spend), format_sar_amount(potential_savings), f"{savings_percentage:.1f}%"],
                    'Strategic_Focus': ['Contract Portfolio', 'Immediate Action', 'Financial Impact', 'Performance Target']
                }
                
                summary_df = pd.DataFrame(summary_data)
                csv = summary_df.to_csv(index=False)
                
                st.download_button(
                    label="Download Executive Summary",
                    data=csv,
                    file_name=f"executive_summary_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        else:
            st.info("Run Smart Identification first to view executive dashboard.")
