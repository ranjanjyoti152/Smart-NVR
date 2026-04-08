import smtplib
import os
import json
import threading
import time
import requests
from html import escape
from urllib.parse import quote_plus
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime, timedelta
from flask import current_app
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

# Track if we've already logged that emails are disabled
_email_disabled_logged = False

# Email rate limiting
_last_email_time = {}  # Dictionary to track last email time for each ROI
_email_cooldown = 10  # Seconds between emails for the same ROI (10 seconds)
_email_queue = []  # Queue for emails to be sent
_email_thread = None  # Thread for sending emails asynchronously
_email_thread_running = False
_email_thread_lock = threading.Lock()
# Track objects that have already triggered notifications for each ROI
_tracked_objects = {}  # Format: {roi_id: {'class_name': timestamp}}
_object_expiry = 30  # Seconds before an object is considered "new" again (5 minutes)

def load_config():
    """Load email configuration from settings file"""
    config_file = os.path.join('config', 'settings.json')
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            # First try to get email config from 'email' key (new format)
            email_config = config.get('email', {})
            
            # If empty, try to get from 'notifications' key (current format in your settings.json)
            if not email_config:
                notifications = config.get('notifications', {})
                if notifications:
                    # Convert notifications structure to expected email config structure
                    email_config = {
                        'enabled': notifications.get('email_enabled', False),
                        'smtp_server': notifications.get('smtp_server'),
                        'smtp_port': notifications.get('smtp_port', 587),
                        'smtp_username': notifications.get('smtp_username'),
                        'smtp_password': notifications.get('smtp_password'),
                        'from_email': notifications.get('from_email'),
                        # Convert email_to to recipients list if it's a string
                        'recipients': [notifications.get('email_to')] if notifications.get('email_to') else []
                    }
                    
                    # Log whether email notifications are enabled or disabled
                    if email_config['enabled']:
                        logger.info("Email notifications are enabled in configuration")
                    else:
                        logger.info("Email notifications are disabled in configuration")
            
            # Load detection settings for Gemini AI
            detection_config = config.get('detection', {})
            email_config['gemini_enabled'] = detection_config.get('enable_gemini_ai', False)
            email_config['gemini_api_key'] = detection_config.get('gemini_api_key', '')
            email_config['gemini_model'] = detection_config.get('gemini_model', 'gemini-1.5-flash')
                    
            return email_config
        except Exception as e:
            logger.error(f"Error loading email config: {str(e)}")
    return {}

def _get_alert_theme(roi_name_lower):
    """Return email theme metadata based on ROI keyword."""
    default_theme = {
        'badge': 'Detection Alert',
        'accent': '#1167D8',
        'surface': '#EEF5FF',
        'title': 'SmartNVR Detection Alert',
        'summary': 'An event was detected in a monitored zone.',
        'actions': [
            'Verify the event in the attached image.',
            'Review recent camera playback for context.',
            'Escalate to the relevant team if needed.'
        ]
    }

    if not roi_name_lower:
        return default_theme

    themed_rules = [
        (('intrusion',), {
            'badge': 'Urgent Security',
            'accent': '#DE2F57',
            'surface': '#FFF0F4',
            'title': 'Potential Intrusion Detected',
            'summary': 'A possible unauthorized entry event was detected.',
            'actions': [
                'Validate the alert immediately using live and recorded feed.',
                'Notify on-site security personnel.',
                'Preserve evidence snapshots and relevant recordings.',
                'Inspect adjacent camera zones for linked activity.'
            ]
        }),
        (('fire',), {
            'badge': 'Emergency',
            'accent': '#D84A1B',
            'surface': '#FFF4EE',
            'title': 'Fire Hazard Alert',
            'summary': 'A possible fire event was detected.',
            'actions': [
                'Trigger emergency response protocol.',
                'Contact fire services immediately.',
                'Coordinate area evacuation if risk is confirmed.'
            ]
        }),
        (('smoke',), {
            'badge': 'Warning',
            'accent': '#FF8B43',
            'surface': '#FFF7F1',
            'title': 'Smoke Detection Warning',
            'summary': 'Smoke-like activity was identified in the monitored area.',
            'actions': [
                'Inspect the source of smoke right away.',
                'Prepare for fire protocol escalation.',
                'Inform safety staff and document findings.'
            ]
        }),
        (('helmet', 'safety'), {
            'badge': 'Safety',
            'accent': '#D99A12',
            'surface': '#FFF9E8',
            'title': 'Safety Compliance Alert',
            'summary': 'A potential safety compliance issue was detected.',
            'actions': [
                'Verify PPE compliance in the affected zone.',
                'Record the incident for safety tracking.',
                'Follow your safety SOP for corrective action.'
            ]
        }),
        (('loitering',), {
            'badge': 'Behavior',
            'accent': '#6A7B92',
            'surface': '#F2F5F9',
            'title': 'Loitering Pattern Detected',
            'summary': 'Potential loitering was observed in the monitored zone.',
            'actions': [
                'Observe live feed to confirm behavior.',
                'Escalate if the pattern appears suspicious.',
                'Log time window and location details.'
            ]
        }),
        (('parking',), {
            'badge': 'Parking',
            'accent': '#4E63C9',
            'surface': '#EEF1FF',
            'title': 'Parking Violation Alert',
            'summary': 'A likely parking policy violation was detected.',
            'actions': [
                'Confirm violation status via camera context.',
                'Document vehicle and timestamp details.',
                'Notify enforcement or operations as required.'
            ]
        }),
        (('restricted',), {
            'badge': 'Restricted Zone',
            'accent': '#CC2B75',
            'surface': '#FFF0F8',
            'title': 'Restricted Zone Violation',
            'summary': 'Potential unauthorized zone entry was detected.',
            'actions': [
                'Validate object type and access permissions.',
                'Notify security team for intervention.',
                'Preserve relevant footage for audit.'
            ]
        }),
        (('vehicle',), {
            'badge': 'Vehicle',
            'accent': '#1F88D9',
            'surface': '#EEF8FF',
            'title': 'Vehicle Activity Alert',
            'summary': 'Vehicle movement was detected in the monitored area.',
            'actions': [
                'Verify whether vehicle presence is expected.',
                'Review trajectory and dwell time.',
                'Record details if follow-up is needed.'
            ]
        }),
        (('package',), {
            'badge': 'Package',
            'accent': '#7E5E44',
            'surface': '#F8F4F1',
            'title': 'Package Activity Detected',
            'summary': 'Package-related activity was detected.',
            'actions': [
                'Confirm delivery or pickup event.',
                'Secure package if unattended.',
                'Archive footage for delivery records.'
            ]
        }),
        (('animal',), {
            'badge': 'Wildlife',
            'accent': '#2D9B65',
            'surface': '#EEF9F3',
            'title': 'Animal Detected',
            'summary': 'Animal presence was detected in the monitored area.',
            'actions': [
                'Identify potential risk to people or assets.',
                'Follow local policy for animal encounters.',
                'Notify operations if intervention is required.'
            ]
        }),
        (('fall',), {
            'badge': 'Assistance Needed',
            'accent': '#9A3DBA',
            'surface': '#F7EFFB',
            'title': 'Potential Fall Event',
            'summary': 'A potential fall was detected and may need urgent assistance.',
            'actions': [
                'Verify incident immediately on live feed.',
                'Contact emergency support if confirmed.',
                'Notify designated contacts and preserve evidence.'
            ]
        }),
    ]

    for keywords, theme in themed_rules:
        if any(keyword in roi_name_lower for keyword in keywords):
            return theme

    return default_theme

def generate_gemini_ai_description(detection_data, roi_data=None, camera_data=None):
    """
    Generate an AI-enhanced description of a detection event using Gemini AI
    
    Args:
        detection_data (dict): Dict containing detection information
        roi_data (dict): Dict containing ROI information
        camera_data (dict): Dict containing camera information
        
    Returns:
        str: AI-generated description or None if failed
    """
    # Load config to check if Gemini is enabled
    config = load_config()
    if not config.get('gemini_enabled', False):
        logger.debug("Gemini AI is disabled, skipping enhanced description")
        return None
    
    # Check for API key (UI setting or env var fallback)
    api_key = config.get('gemini_api_key', '') or os.getenv('SMARTNVR_GEMINI_API_KEY', '')
    if not api_key:
        logger.warning("Gemini AI is enabled but no API key is configured")
        return None
    
    # Format timestamp if it's a datetime object
    timestamp_str = detection_data['timestamp'].strftime('%I:%M %p') if isinstance(detection_data['timestamp'], datetime) else str(detection_data['timestamp'])
    
    # Construct prompt for Gemini
    prompt = f"""
    Generate a short, professional, human-friendly notification for a security camera detection:
    
    Details:
    - Object detected: {detection_data['class_name']}
    - Confidence: {detection_data['confidence']:.1%}
    - Time: {timestamp_str}
    - Camera: {camera_data['name'] if camera_data else 'Unknown'}
    """
    
    # Add ROI information if available
    if roi_data:
        prompt += f"- Detection Zone: {roi_data['name']}\n"
    
    prompt += """
    Generate a concise (1-2 sentences), informative, alert message suitable for a security notification.
    Focus on clarity, professionalism, and actionability. The response should be in this format:
    "Security Alert: A [object] was detected at [location] at [time]."
    
    Do not mention confidence percentages in the notification.
    Keep your response very brief and direct - no introduction or explanation needed.
    """
    
    # Prepare API request
    try:
        model = config.get('gemini_model', 'gemini-1.5-flash')
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 100,
                "topP": 0.8,
                "topK": 40
            }
        }
        
        # Make the API request with a tiny retry/backoff for transient issues
        timeouts = [5, 8]
        for idx, t in enumerate(timeouts):
            try:
                response = requests.post(url, headers=headers, json=data, timeout=t)

                if response.status_code == 200:
                    response_data = response.json()
                    if 'candidates' in response_data and response_data['candidates']:
                        generated_text = response_data['candidates'][0]['content']['parts'][0]['text']
                        generated_text = generated_text.strip().strip('"\'')
                        logger.info(f"Generated Gemini AI description: {generated_text}")
                        return generated_text
                    else:
                        logger.warning(f"Gemini API returned empty response: {response_data}")
                        break
                else:
                    logger.warning(f"Gemini API request failed with status {response.status_code}: {response.text}")
                    break

            except requests.exceptions.Timeout:
                logger.warning(f"Gemini description timeout on attempt {idx+1}/{len(timeouts)} after {t}s")
                if idx < len(timeouts) - 1:
                    time.sleep(0.5)
                    continue
                else:
                    break
            except requests.exceptions.RequestException as re:
                logger.error(f"Gemini description request error: {re}")
                break

    except Exception as e:
        logger.error(f"Error generating Gemini AI description: {str(e)}")
    
    # Return None if we couldn't generate a description
    return None

def _email_worker():
    """Background worker thread that processes the email queue"""
    global _email_thread_running
    
    logger.info("Email worker thread started")
    
    while _email_thread_running:
        try:
            # Check if there are any emails to send
            if _email_queue:
                # Get the next email job
                with _email_thread_lock:
                    if _email_queue:
                        email_job = _email_queue.pop(0)
                    else:
                        email_job = None
                
                if email_job:
                    _send_email_internal(email_job)
            
            # Sleep to avoid tight loop
            time.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error in email worker thread: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            time.sleep(1)  # Prevent rapid retries if there's an error
    
    logger.info("Email worker thread stopped")

def _start_email_worker():
    """Start the email worker thread if it's not already running"""
    global _email_thread, _email_thread_running
    
    if _email_thread is None or not _email_thread.is_alive():
        _email_thread_running = True
        _email_thread = threading.Thread(target=_email_worker)
        _email_thread.daemon = True
        _email_thread.start()
        logger.info("Started email worker thread")

def send_detection_email(camera, detection, roi=None):
    """Queue an email notification for a detection event
    
    Args:
        camera: Camera object from database
        detection: Detection object from database
        roi: Optional ROI object if detection was within a specific region
    """
    global _email_disabled_logged, _last_email_time, _tracked_objects
    
    # Start the email worker thread if not running
    _start_email_worker()
    
    # Load email config
    email_config = load_config()
    
    # Quick check if email notifications are enabled
    if not email_config.get('enabled', False):
        if not _email_disabled_logged:
            logger.info("Email notifications are disabled in configuration")
            _email_disabled_logged = True
        return False
    
    # If no ROI or ROI doesn't have email notifications enabled, don't send
    if roi is None or not getattr(roi, 'email_notifications', False):
        logger.debug(f"Email notification skipped - no ROI or ROI email not enabled")
        return False
    
    # Check if detection class is included in the ROI's detection classes
    if roi and hasattr(roi, 'detection_classes') and roi.detection_classes:
        try:
            # First, convert to a list regardless of format (string JSON or list)
            if isinstance(roi.detection_classes, str) and roi.detection_classes.strip():
                # Try to parse JSON string
                try:
                    roi_classes = json.loads(roi.detection_classes)
                except json.JSONDecodeError:
                    # If it's not valid JSON, try comma-separated format
                    roi_classes = [c.strip() for c in roi.detection_classes.split(',')]
            elif isinstance(roi.detection_classes, (list, tuple)):
                roi_classes = roi.detection_classes
            else:
                roi_classes = []
            
            # If ROI has specified classes (not empty list) and the detected class isn't in the list, skip
            if roi_classes:
                # Convert class IDs to class names if needed
                if all(isinstance(c, int) or (isinstance(c, str) and c.isdigit()) for c in roi_classes):
                    # These are class IDs - detection.class_name is the actual name, need to check by ID
                    # This is a simplification - you might need to look up class ID to name mapping
                    # in your actual implementation
                    logger.debug(f"ROI classes specified by ID: {roi_classes}, detected class: {detection.class_name}")
                    
                    # If detection has class_id attribute, use it for comparison
                    if hasattr(detection, 'class_id'):
                        detection_class_id = detection.class_id
                        match_found = str(detection_class_id) in [str(c) for c in roi_classes]
                    else:
                        # Try to infer class ID from name using COCO classes (if applicable)
                        # This is just an example, adjust based on your actual class mapping
                        coco_classes = {
                            'person': 0, 'bicycle': 1, 'car': 2, 'motorcycle': 3, 'airplane': 4, 
                            'bus': 5, 'train': 6, 'truck': 7, 'boat': 8, 'traffic light': 9,
                            'fire hydrant': 10, 'stop sign': 11, 'parking meter': 12, 'bench': 13, 
                            'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17, 'sheep': 18, 'cow': 19, 
                            'elephant': 20, 'bear': 21, 'zebra': 22, 'giraffe': 23, 'backpack': 24, 
                            'umbrella': 25, 'handbag': 26, 'tie': 27, 'suitcase': 28, 'frisbee': 29, 
                            'skis': 30, 'snowboard': 31, 'sports ball': 32, 'kite': 33, 'baseball bat': 34, 
                            'baseball glove': 35, 'skateboard': 36, 'surfboard': 37, 'tennis racket': 38, 
                            'bottle': 39, 'wine glass': 40, 'cup': 41, 'fork': 42, 'knife': 43, 
                            'spoon': 44, 'bowl': 45, 'banana': 46, 'apple': 47, 'sandwich': 48, 
                            'orange': 49, 'broccoli': 50, 'carrot': 51, 'hot dog': 52, 'pizza': 53, 
                            'donut': 54, 'cake': 55, 'chair': 56, 'couch': 57, 'potted plant': 58, 
                            'bed': 59, 'dining table': 60, 'toilet': 61, 'tv': 62, 'laptop': 63, 
                            'mouse': 64, 'remote': 65, 'keyboard': 66, 'cell phone': 67, 'microwave': 68, 
                            'oven': 69, 'toaster': 70, 'sink': 71, 'refrigerator': 72, 'book': 73, 
                            'clock': 74, 'vase': 75, 'scissors': 76, 'teddy bear': 77, 'hair drier': 78, 
                            'toothbrush': 79
                        }
                        # Try direct lookup, fall back to lower case if needed
                        detection_class_id = coco_classes.get(detection.class_name, 
                                                                coco_classes.get(detection.class_name.lower(), -1))
                        # Check if detected class ID is in the ROI's class list
                        match_found = str(detection_class_id) in [str(c) for c in roi_classes]
                else:
                    # These are class names, direct comparison
                    match_found = detection.class_name in roi_classes or detection.class_name.lower() in [c.lower() for c in roi_classes]
                
                if not match_found:
                    logger.info(f"Email notification skipped - {detection.class_name} not in ROI's detection classes: {roi_classes}")
                    return False
            else:
                # Empty detection_classes list means "all classes allowed" - don't skip
                logger.debug(f"ROI {roi.name} has empty detection_classes list - allowing all detections")
                    
        except Exception as e:
            # If there's any error parsing the classes, log the error but allow notification
            logger.warning(f"Error parsing detection classes for ROI {roi.id}: {str(e)} - allowing notification")
    else:
        # No detection_classes specified means "all classes allowed" - don't skip
        logger.debug(f"ROI {roi.name if roi else 'None'} has no detection_classes specified - allowing all detections")
    
    # Apply rate limiting for this ROI and object
    roi_id = getattr(roi, 'id', 'unknown')
    now = datetime.now()
    
    # Check if the object has already triggered a notification recently
    tracked_objects = _tracked_objects.get(roi_id, {})
    object_last_seen = tracked_objects.get(detection.class_name)
    
    if object_last_seen and (now - object_last_seen).total_seconds() < _object_expiry:
        logger.info(f"Email notification for {detection.class_name} in ROI {roi.name} (id={roi_id}) skipped due to duplicate object tracking")
        return False
    
    # Update the last email time for this ROI
    _last_email_time[roi_id] = now
    
    # Update tracked objects for this ROI
    tracked_objects[detection.class_name] = now
    _tracked_objects[roi_id] = tracked_objects
    
    # Extract only the necessary attributes from the database objects to avoid
    # SQLAlchemy detached instance errors in the background thread
    email_data = {
        'camera': {
            'id': camera.id,
            'name': camera.name
        },
        'detection': {
            'class_name': detection.class_name,
            'confidence': detection.confidence,
            'timestamp': detection.timestamp,
            'image_path': detection.image_path,
            'video_path': detection.video_path,
            'id': detection.id
        },
        'roi': {
            'id': roi.id,
            'name': roi.name
        } if roi else None
    }
    
    # Add to email queue for async processing
    with _email_thread_lock:
        _email_queue.append(email_data)
    
    logger.info(f"Queued email notification for {detection.class_name} in ROI {roi.name if roi else 'unknown'}")
    return True

def _send_email_internal(email_data):
    """Actually send the email (called from the worker thread)
    
    Args:
        email_data: Dictionary containing camera, detection and ROI information
    """
    # Load email config
    email_config = load_config()
    
    # Extract data from the email_data dictionary
    camera_data = email_data['camera']
    detection_data = email_data['detection']
    roi_data = email_data['roi']
    
    # Generate AI-enhanced description if Gemini is enabled
    ai_description = None
    if email_config.get('gemini_enabled', False):
        ai_description = generate_gemini_ai_description(detection_data, roi_data, camera_data)
        if ai_description:
            logger.info(f"Using Gemini AI description: {ai_description}")
    
    # Check required parameters
    smtp_server = email_config.get('smtp_server')
    smtp_port = email_config.get('smtp_port', 587)
    smtp_username = email_config.get('smtp_username')
    smtp_password = email_config.get('smtp_password')
    from_email = email_config.get('from_email')
    recipients = email_config.get('recipients', [])
    
    if not all([smtp_server, smtp_username, smtp_password, from_email, recipients]):
        missing = []
        if not smtp_server: missing.append("smtp_server")
        if not smtp_username: missing.append("smtp_username")
        if not smtp_password: missing.append("smtp_password")
        if not from_email: missing.append("from_email")
        if not recipients: missing.append("recipients")
        logger.warning(f"Incomplete email configuration: missing {', '.join(missing)}")
        return False
    
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = from_email
        msg['To'] = ', '.join(recipients)
        
        # Get ROI name (lowercase for case-insensitive comparison)
        roi_name_lower = roi_data['name'].lower() if roi_data else ""
        
        # Customize subject based on ROI name
        if roi_data:
            # Add urgency level for specific ROIs
            if "intrusion" in roi_name_lower:
                msg['Subject'] = f"🚨 URGENT: Intrusion Detected - {detection_data['class_name']} in {roi_data['name']} on {camera_data['name']}"
            elif "fire" in roi_name_lower:
                msg['Subject'] = f"🔥 EMERGENCY: Fire Detection Alert - {detection_data['class_name']} in {roi_data['name']} on {camera_data['name']}"
            elif "smoke" in roi_name_lower:
                msg['Subject'] = f"⚠️ WARNING: Smoke Detected - {detection_data['class_name']} in {roi_data['name']} on {camera_data['name']}"
            elif "helmet" in roi_name_lower or "safety" in roi_name_lower:
                msg['Subject'] = f"👷 Safety Alert: {detection_data['class_name']} detected in {roi_data['name']} on {camera_data['name']}"
            elif "loitering" in roi_name_lower:
                msg['Subject'] = f"🚶‍♂️ Loitering Alert: {detection_data['class_name']} in {roi_data['name']} on {camera_data['name']}"
            elif "parking" in roi_name_lower:
                msg['Subject'] = f"🅿️ Parking Violation: {detection_data['class_name']} in {roi_data['name']} on {camera_data['name']}"
            elif "vehicle" in roi_name_lower:
                msg['Subject'] = f"🚗 Vehicle Alert: {detection_data['class_name']} in {roi_data['name']} on {camera_data['name']}"
            elif "restricted" in roi_name_lower:
                msg['Subject'] = f"⛔ Restricted Zone Alert: {detection_data['class_name']} in {roi_data['name']} on {camera_data['name']}"
            elif "package" in roi_name_lower:
                msg['Subject'] = f"📦 Package Alert: {detection_data['class_name']} in {roi_data['name']} on {camera_data['name']}"
            elif "animal" in roi_name_lower:
                msg['Subject'] = f"🐾 Animal Detection: {detection_data['class_name']} in {roi_data['name']} on {camera_data['name']}"
            elif "fall" in roi_name_lower:
                msg['Subject'] = f"❗ Fall Detected: {detection_data['class_name']} in {roi_data['name']} on {camera_data['name']}"
            else:
                msg['Subject'] = f"SmartNVR Alert: {detection_data['class_name']} detected in {roi_data['name']} on {camera_data['name']}"
        else:
            msg['Subject'] = f"SmartNVR Alert: {detection_data['class_name']} detected on {camera_data['name']}"
        
        # Format timestamp if it's a datetime object
        timestamp_str = detection_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(detection_data['timestamp'], datetime) else str(detection_data['timestamp'])
        
        theme = _get_alert_theme(roi_name_lower)
        camera_name = escape(str(camera_data.get('name', 'Unknown Camera')))
        camera_id = camera_data.get('id', '')
        object_name = escape(str(detection_data.get('class_name', 'Unknown')))
        roi_name = escape(str(roi_data.get('name', 'Not specified'))) if roi_data else 'Not specified'
        ai_description_safe = escape(str(ai_description)) if ai_description else ''

        try:
            confidence_percentage = float(detection_data.get('confidence', 0.0)) * 100.0
        except (TypeError, ValueError):
            confidence_percentage = 0.0

        details_rows = f"""
            <tr>
                <td style="padding:10px 12px; color:#73829D; font-size:13px; border-bottom:1px solid #E4EBF6;">Camera</td>
                <td style="padding:10px 12px; color:#0F1F34; font-size:13px; font-weight:600; border-bottom:1px solid #E4EBF6;">{camera_name}</td>
            </tr>
            <tr>
                <td style="padding:10px 12px; color:#73829D; font-size:13px; border-bottom:1px solid #E4EBF6;">Object</td>
                <td style="padding:10px 12px; color:#0F1F34; font-size:13px; font-weight:600; border-bottom:1px solid #E4EBF6;">{object_name}</td>
            </tr>
            <tr>
                <td style="padding:10px 12px; color:#73829D; font-size:13px; border-bottom:1px solid #E4EBF6;">Confidence</td>
                <td style="padding:10px 12px; color:#0F1F34; font-size:13px; font-weight:600; border-bottom:1px solid #E4EBF6;">{confidence_percentage:.2f}%</td>
            </tr>
            <tr>
                <td style="padding:10px 12px; color:#73829D; font-size:13px; border-bottom:1px solid #E4EBF6;">Detection Time</td>
                <td style="padding:10px 12px; color:#0F1F34; font-size:13px; font-weight:600; border-bottom:1px solid #E4EBF6;">{escape(timestamp_str)}</td>
            </tr>
        """

        if roi_data:
            details_rows += f"""
            <tr>
                <td style="padding:10px 12px; color:#73829D; font-size:13px;">Detection Zone</td>
                <td style="padding:10px 12px; color:#0F1F34; font-size:13px; font-weight:600;">{roi_name}</td>
            </tr>
            """

        actions_html = ''.join(
            f'<li style="margin:0 0 8px; color:#3C4D67;">{escape(action)}</li>'
            for action in theme['actions']
        )

        ai_block = ''
        if ai_description_safe:
            ai_block = f"""
            <tr>
                <td style="padding:0 28px 20px;">
                    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:separate; background:#EEF8FF; border:1px solid #CFE8FF; border-left:4px solid #2E82FF; border-radius:12px;">
                        <tr>
                            <td style="padding:14px 16px;">
                                <p style="margin:0 0 6px; font-size:12px; letter-spacing:0.08em; text-transform:uppercase; color:#2E82FF; font-weight:700;">AI Smart Analysis</p>
                                <p style="margin:0; color:#23354F; font-size:14px; line-height:1.45;">{ai_description_safe}</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
            """

        image_block = ''
        image_path = detection_data.get('image_path')
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-ID', '<detection_image>')
                msg.attach(img)
            image_block = """
            <tr>
                <td style="padding:0 28px 22px;">
                    <img src="cid:detection_image" alt="Detection Snapshot" width="624" style="display:block; width:100%; max-width:624px; border-radius:14px; border:1px solid #D9E5F4;" />
                </td>
            </tr>
            """

        playback_block = ''
        if detection_data.get('video_path'):
            playback_path = (
                f"/playback?video={quote_plus(os.path.basename(detection_data['video_path']))}"
                f"&camera={quote_plus(str(camera_id))}"
            )
            public_base_url = str(current_app.config.get('PUBLIC_BASE_URL', '')).rstrip('/')
            playback_url = f"{public_base_url}{playback_path}" if public_base_url else playback_path
            playback_block = f"""
            <tr>
                <td align="center" style="padding:0 28px 24px;">
                    <a href="{escape(playback_url)}" style="display:inline-block; padding:12px 20px; border-radius:12px; background:#1167D8; color:#FFFFFF; font-size:14px; font-weight:600; text-decoration:none;">Open Playback</a>
                </td>
            </tr>
            """

        body = f"""
        <html>
        <body style="margin:0; padding:0; background:#EEF2F7; font-family:'Segoe UI', Arial, sans-serif;">
            <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse; background:#EEF2F7;">
                <tr>
                    <td align="center" style="padding:22px 12px;">
                        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="max-width:680px; border-collapse:separate; background:#FFFFFF; border:1px solid #D9E5F4; border-radius:18px; overflow:hidden;">
                            <tr>
                                <td style="padding:0;">
                                    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse; background:{theme['accent']};">
                                        <tr>
                                            <td style="padding:16px 22px;">
                                                <p style="margin:0 0 6px; color:#DDEBFF; font-size:11px; font-weight:700; letter-spacing:0.09em; text-transform:uppercase;">SmartNVR | {escape(theme['badge'])}</p>
                                                <h2 style="margin:0; color:#FFFFFF; font-size:24px; line-height:1.2;">{escape(theme['title'])}</h2>
                                            </td>
                                        </tr>
                                    </table>
                                </td>
                            </tr>
                            <tr>
                                <td style="padding:18px 28px 12px;">
                                    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:separate; background:{theme['surface']}; border:1px solid #D9E5F4; border-radius:14px;">
                                        <tr>
                                            <td style="padding:14px 16px;">
                                                <p style="margin:0; color:#23354F; font-size:14px; line-height:1.5;">{escape(theme['summary'])}</p>
                                            </td>
                                        </tr>
                                    </table>
                                </td>
                            </tr>
                            <tr>
                                <td style="padding:0 28px 20px;">
                                    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:separate; background:#F8FBFF; border:1px solid #D9E5F4; border-radius:14px;">
                                        {details_rows}
                                    </table>
                                </td>
                            </tr>
                            {ai_block}
                            <tr>
                                <td style="padding:0 28px 18px;">
                                    <h3 style="margin:0 0 10px; color:#0F1F34; font-size:16px;">Recommended Actions</h3>
                                    <ul style="margin:0; padding-left:20px; font-size:14px; line-height:1.45;">
                                        {actions_html}
                                    </ul>
                                </td>
                            </tr>
                            {image_block}
                            {playback_block}
                            <tr>
                                <td style="padding:14px 22px; border-top:1px solid #E4EBF6; background:#F8FBFF;">
                                    <p style="margin:0; color:#73829D; font-size:12px; line-height:1.4;">Automated notification from SmartNVR Vision Control Center.</p>
                                </td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Connect to SMTP server and send email
        start_time = time.time()
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        end_time = time.time()
        
        if roi_data:
            logger.info(f"Email sent for {detection_data['class_name']} in ROI {roi_data['name']} on {camera_data['name']} (took {end_time-start_time:.2f}s)")
        else:
            logger.info(f"Email sent for {detection_data['class_name']} on {camera_data['name']} (took {end_time-start_time:.2f}s)")
        
        # Here we would mark the detection as notified, but since we're now using a dictionary
        # instead of the actual detection object, we can't modify it.
        # This would require updating the database in a properly bound session.
        # For now, we'll skip this step since the notification was still sent successfully.
        
        return True
        
    except Exception as e:
        logger.error(f"Error sending email notification: {str(e)}")
        # Print full stack trace for better debugging
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def send_test_email(smtp_server, smtp_port, smtp_username, smtp_password, recipients):
    """
    Send a test email to verify SMTP configuration
    
    Args:
        smtp_server (str): SMTP server address
        smtp_port (int): SMTP server port
        smtp_username (str): SMTP username
        smtp_password (str): SMTP password
        recipients (list): List of email addresses to send to
        
    Returns:
        dict: Result with success status and error message if applicable
    """
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = smtp_username
        msg['To'] = ', '.join(recipients)
        msg['Subject'] = "SmartNVR - Test Email"
        
        sent_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        body = f"""
        <html>
        <body style="margin:0; padding:0; background:#EEF2F7; font-family:'Segoe UI', Arial, sans-serif;">
            <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse; background:#EEF2F7;">
                <tr>
                    <td align="center" style="padding:22px 12px;">
                        <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="max-width:640px; border-collapse:separate; background:#FFFFFF; border:1px solid #D9E5F4; border-radius:18px; overflow:hidden;">
                            <tr>
                                <td style="padding:18px 22px; background:#1167D8;">
                                    <p style="margin:0 0 6px; color:#DDEBFF; font-size:11px; font-weight:700; letter-spacing:0.09em; text-transform:uppercase;">SmartNVR | Notification Test</p>
                                    <h2 style="margin:0; color:#FFFFFF; font-size:24px; line-height:1.2;">Email Configuration Verified</h2>
                                </td>
                            </tr>
                            <tr>
                                <td style="padding:18px 24px;">
                                    <p style="margin:0 0 14px; color:#23354F; font-size:14px; line-height:1.5;">
                                        This test confirms that your SmartNVR email notifications are configured correctly.
                                    </p>
                                    <table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="border-collapse:separate; background:#F8FBFF; border:1px solid #D9E5F4; border-radius:14px;">
                                        <tr>
                                            <td style="padding:10px 12px; color:#73829D; font-size:13px; border-bottom:1px solid #E4EBF6;">SMTP Server</td>
                                            <td style="padding:10px 12px; color:#0F1F34; font-size:13px; font-weight:600; border-bottom:1px solid #E4EBF6;">{escape(str(smtp_server))}</td>
                                        </tr>
                                        <tr>
                                            <td style="padding:10px 12px; color:#73829D; font-size:13px; border-bottom:1px solid #E4EBF6;">SMTP Port</td>
                                            <td style="padding:10px 12px; color:#0F1F34; font-size:13px; font-weight:600; border-bottom:1px solid #E4EBF6;">{escape(str(smtp_port))}</td>
                                        </tr>
                                        <tr>
                                            <td style="padding:10px 12px; color:#73829D; font-size:13px; border-bottom:1px solid #E4EBF6;">Username</td>
                                            <td style="padding:10px 12px; color:#0F1F34; font-size:13px; font-weight:600; border-bottom:1px solid #E4EBF6;">{escape(str(smtp_username))}</td>
                                        </tr>
                                        <tr>
                                            <td style="padding:10px 12px; color:#73829D; font-size:13px;">Sent At</td>
                                            <td style="padding:10px 12px; color:#0F1F34; font-size:13px; font-weight:600;">{escape(sent_at)}</td>
                                        </tr>
                                    </table>
                                </td>
                            </tr>
                            <tr>
                                <td style="padding:14px 22px; border-top:1px solid #E4EBF6; background:#F8FBFF;">
                                    <p style="margin:0; color:#73829D; font-size:12px; line-height:1.4;">Automated notification from SmartNVR Vision Control Center.</p>
                                </td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
        </body>
        </html>
        """
        
        msg.attach(MIMEText(body, 'html'))
        
        # Connect to SMTP server and send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
        
        logger.info(f"Test email sent to {', '.join(recipients)}")
        
        return {
            'success': True
        }
        
    except Exception as e:
        logger.error(f"Error sending test email: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

# Make sure email thread stops when app shuts down
def shutdown():
    """Shutdown the email worker thread"""
    global _email_thread_running
    _email_thread_running = False
    if _email_thread and _email_thread.is_alive():
        _email_thread.join(timeout=5.0)
        logger.info("Email worker thread shutdown complete")
