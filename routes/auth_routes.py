# routes/auth_routes.py
"""
Authentication routes module.
Handles login, logout, and access data endpoints.
"""

import logging
from flask import Blueprint, request, jsonify, render_template, session, redirect, url_for
from modules.auth import (
    load_access_data, 
    validate_designation,
    set_user_session,
    clear_user_session,
    require_login
)

logger = logging.getLogger(__name__)

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/')
def index():
    """Redirect to login or chatbot based on session"""
    if 'user_designation' in session:
        return redirect(url_for('auth.chatbot'))
    return redirect(url_for('auth.login'))


@auth_bp.route('/login')
def login():
    """Serve the login page"""
    try:
        return render_template('login_new.html')
    except Exception as e:
        return f"Template error: {e}", 500


@auth_bp.route('/chatbot')
@require_login
def chatbot():
    """Serve the main chatbot interface"""
    try:
        return render_template('chatbot_new.html')
    except Exception as e:
        return f"Template error: {e}", 500


@auth_bp.route('/api/login', methods=['POST'])
def api_login():
    """Handle user login"""
    try:
        data = request.get_json()
        if not data or 'designation' not in data:
            return jsonify({'error': 'No designation provided'}), 400
        
        designation = data['designation'].strip()
        if not designation:
            return jsonify({'error': 'Empty designation'}), 400
        
        if not validate_designation(designation):
            return jsonify({'error': 'Invalid designation'}), 400
        
        set_user_session(designation)
        
        return jsonify({
            'message': 'Login successful',
            'designation': designation,
            'redirect_url': url_for('auth.chatbot')
        })
        
    except Exception as e:
        logger.error(f"Login API error: {e}")
        return jsonify({'error': 'Login failed', 'message': str(e)}), 500


@auth_bp.route('/api/logout', methods=['POST'])
def api_logout():
    """Handle user logout"""
    try:
        clear_user_session()
        return jsonify({'message': 'Logged out successfully', 'redirect_url': url_for('auth.login')})
    except Exception as e:
        logger.error(f"Logout API error: {e}")
        return jsonify({'error': 'Logout failed', 'message': str(e)}), 500


@auth_bp.route('/api/access-data')
def api_access_data():
    """Get access data for login page"""
    try:
        access_data = load_access_data()
        return jsonify({'access_data': access_data})
    except Exception as e:
        logger.error(f"Access data API error: {e}")
        return jsonify({'error': 'Failed to load access data', 'message': str(e)}), 500