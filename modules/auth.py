# modules/auth.py
"""
Authentication and access control module.
Handles user login, session management, and document access permissions.
"""

import os
import csv
import logging
from pathlib import Path
from functools import wraps
from flask import session, redirect, url_for, jsonify

logger = logging.getLogger(__name__)
current_dir = Path(__file__).parent.parent


def load_access_data():
    """Load access permissions from CSV file"""
    access_data = {}
    csv_path = os.path.join(current_dir, 'access.csv')
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                designation = row['Designation']
                access_data[designation] = {}
                for doc_name, access in row.items():
                    if doc_name != 'Designation':
                        access_data[designation][doc_name] = int(access)
        return access_data
    except Exception as e:
        logger.error(f"Failed to load access data: {e}")
        return {}


def get_user_access(designation):
    """Get list of documents user has access to"""
    access_data = load_access_data()
    if designation not in access_data:
        return []
    
    user_access = []
    for doc_name, has_access in access_data[designation].items():
        if has_access == 1:
            user_access.append(doc_name)
    
    return user_access


def require_login(f):
    """No-op decorator — authentication disabled, all routes are public."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        return f(*args, **kwargs)
    return decorated_function


def validate_designation(designation):
    """Validate if designation exists in access data"""
    access_data = load_access_data()
    return designation in access_data


def check_document_access(designation, document_names):
    """Check if user has access to specified documents"""
    user_access = get_user_access(designation)
    unauthorized = [doc for doc in document_names if doc not in user_access]
    return len(unauthorized) == 0, unauthorized


def set_user_session(designation):
    """Set user session data"""
    session['user_designation'] = designation
    session.permanent = True


def clear_user_session():
    """Clear user session data"""
    session.clear()