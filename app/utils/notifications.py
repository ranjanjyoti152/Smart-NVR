import smtplib
import os
import json
import threading
import time
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
_email_cooldown = 10  # Seconds between emails for the same ROI (1 minute)
_email_queue = []  # Queue for emails to be sent
_email_thread = None  # Thread for sending emails asynchronously
_email_thread_running = False
_email_thread_lock = threading.Lock()

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
                    
            return email_config
        except Exception as e:
            logger.error(f"Error loading email config: {str(e)}")
    return {}

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
    global _email_disabled_logged, _last_email_time
    
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
    
    # Apply rate limiting for this ROI
    roi_id = getattr(roi, 'id', 'unknown')
    now = datetime.now()
    last_email_time = _last_email_time.get(roi_id)
    
    if last_email_time and (now - last_email_time).total_seconds() < _email_cooldown:
        logger.info(f"Email notification for ROI {roi.name} (id={roi_id}) skipped due to cooldown period")
        return False
    
    # Update the last email time for this ROI
    _last_email_time[roi_id] = now
    
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
        
        # Email header style
        header_style = "style='padding: 10px; color: white; font-size: 18px; text-align: center;'"
        
        # Get custom styles and content based on ROI name
        header_bg = "#2C3E50"  # Default header background
        alert_box_style = "style='border: 1px solid #ddd; padding: 15px; margin: 15px 0; border-radius: 5px; background-color: #f9f9f9;'"
        custom_message = ""
        action_steps = ""
        
        if roi_data:
            if "intrusion" in roi_name_lower:
                header_bg = "#B90E0A"  # Red for intrusion
                alert_box_style = "style='border: 1px solid #B90E0A; padding: 15px; margin: 15px 0; border-radius: 5px; background-color: #FFEBEE;'"
                custom_message = f"""
                <div {alert_box_style}>
                    <h3 style='color: #B90E0A; margin-top: 0;'>⚠️ SECURITY BREACH DETECTED ⚠️</h3>
                    <p>A potential intruder has been detected in an area marked for intrusion monitoring.</p>
                    <p>Detected object: <strong>{detection_data['class_name']}</strong> with {detection_data['confidence']:.1%} confidence.</p>
                </div>
                """
                action_steps = """
                <h3>Recommended Actions:</h3>
                <ol>
                    <li>Verify the alert by checking the attached image and video</li>
                    <li>Contact security personnel immediately if intrusion is confirmed</li>
                    <li>Save video evidence for future reference</li>
                    <li>Check other camera feeds for additional intrusion points</li>
                </ol>
                """
                
            elif "fire" in roi_name_lower:
                header_bg = "#D32F2F"  # Bright red for fire
                alert_box_style = "style='border: 1px solid #D32F2F; padding: 15px; margin: 15px 0; border-radius: 5px; background-color: #FFEBEE;'"
                custom_message = f"""
                <div {alert_box_style}>
                    <h3 style='color: #D32F2F; margin-top: 0;'>🔥 FIRE EMERGENCY ALERT 🔥</h3>
                    <p>A potential fire has been detected in the monitoring area.</p>
                    <p>Detected object: <strong>{detection_data['class_name']}</strong> with {detection_data['confidence']:.1%} confidence.</p>
                </div>
                """
                action_steps = """
                <h3>Immediate Actions Required:</h3>
                <ol>
                    <li>Evacuate the area immediately</li>
                    <li>Contact emergency services (Fire Department) at once</li>
                    <li>Activate fire alarm systems if not already triggered</li>
                    <li>Do NOT attempt to fight substantial fires</li>
                    <li>Follow your organization's fire emergency procedures</li>
                </ol>
                """
                
            elif "smoke" in roi_name_lower:
                header_bg = "#FF9800"  # Orange for smoke warning
                alert_box_style = "style='border: 1px solid #FF9800; padding: 15px; margin: 15px 0; border-radius: 5px; background-color: #FFF3E0;'"
                custom_message = f"""
                <div {alert_box_style}>
                    <h3 style='color: #FF9800; margin-top: 0;'>🚨 SMOKE DETECTION WARNING 🚨</h3>
                    <p>Smoke has been detected in the monitoring area which could indicate a fire hazard.</p>
                    <p>Detected object: <strong>{detection_data['class_name']}</strong> with {detection_data['confidence']:.1%} confidence.</p>
                </div>
                """
                action_steps = """
                <h3>Recommended Actions:</h3>
                <ol>
                    <li>Investigate the source of smoke immediately</li>
                    <li>Prepare for possible evacuation</li>
                    <li>Alert fire safety personnel</li>
                    <li>Check for signs of fire</li>
                    <li>If fire is confirmed, follow fire emergency procedures</li>
                </ol>
                """
                
            elif "helmet" in roi_name_lower or "safety" in roi_name_lower:
                header_bg = "#FFC107"  # Yellow for safety violations
                alert_box_style = "style='border: 1px solid #FFC107; padding: 15px; margin: 15px 0; border-radius: 5px; background-color: #FFF8E1;'"
                custom_message = f"""
                <div {alert_box_style}>
                    <h3 style='color: #FFC107; margin-top: 0;'>👷 SAFETY COMPLIANCE ALERT 👷</h3>
                    <p>A safety-related detection has occurred in the monitoring area.</p>
                    <p>Detected object: <strong>{detection_data['class_name']}</strong> with {detection_data['confidence']:.1%} confidence.</p>
                </div>
                """
                action_steps = """
                <h3>Recommended Actions:</h3>
                <ol>
                    <li>Verify if proper safety equipment is being used</li>
                    <li>Address any safety violations</li>
                    <li>Document the incident for safety records</li>
                    <li>Consider additional safety training if violations are frequent</li>
                </ol>
                """
            
            elif "loitering" in roi_name_lower:
                header_bg = "#607D8B"  # Blue Grey for loitering
                alert_box_style = "style='border: 1px solid #607D8B; padding: 15px; margin: 15px 0; border-radius: 5px; background-color: #ECEFF1;'"
                custom_message = f"""
                <div {alert_box_style}>
                    <h3 style='color: #607D8B; margin-top: 0;'>⏱️ LOITERING ALERT ⏱️</h3>
                    <p>Potential loitering detected in the monitored area.</p>
                    <p>Detected object: <strong>{detection_data['class_name']}</strong> with {detection_data['confidence']:.1%} confidence.</p>
                </div>
                """
                action_steps = """
                <h3>Recommended Actions:</h3>
                <ol>
                    <li>Monitor the situation via live feed</li>
                    <li>Assess if the activity is suspicious or benign</li>
                    <li>If suspicious, contact security or relevant personnel</li>
                    <li>Note the time and duration for records</li>
                </ol>
                """

            elif "vehicle" in roi_name_lower:
                header_bg = "#03A9F4"  # Light Blue for vehicle
                alert_box_style = "style='border: 1px solid #03A9F4; padding: 15px; margin: 15px 0; border-radius: 5px; background-color: #E1F5FE;'"
                custom_message = f"""
                <div {alert_box_style}>
                    <h3 style='color: #03A9F4; margin-top: 0;'>🚗 VEHICLE ALERT 🚗</h3>
                    <p>A vehicle has been detected in the specified zone.</p>
                    <p>Detected object: <strong>{detection_data['class_name']}</strong> with {detection_data['confidence']:.1%} confidence.</p>
                </div>
                """
                action_steps = """
                <h3>Recommended Actions:</h3>
                <ol>
                    <li>Verify if the vehicle is authorized in this area</li>
                    <li>Check for parking violations if applicable</li>
                    <li>Monitor vehicle activity if necessary</li>
                    <li>Note license plate if possible and required</li>
                </ol>
                """

            elif "package" in roi_name_lower:
                header_bg = "#795548"  # Brown for package
                alert_box_style = "style='border: 1px solid #795548; padding: 15px; margin: 15px 0; border-radius: 5px; background-color: #EFEBE9;'"
                custom_message = f"""
                <div {alert_box_style}>
                    <h3 style='color: #795548; margin-top: 0;'>📦 PACKAGE DELIVERY DETECTED 📦</h3>
                    <p>A package-related activity has been detected.</p>
                    <p>Detected object: <strong>{detection_data['class_name']}</strong> with {detection_data['confidence']:.1%} confidence.</p>
                </div>
                """
                action_steps = """
                <h3>Recommended Actions:</h3>
                <ol>
                    <li>Check the camera feed to confirm delivery</li>
                    <li>Secure the package as soon as possible</li>
                    <li>Review footage if package goes missing</li>
                </ol>
                """

            elif "animal" in roi_name_lower:
                header_bg = "#4CAF50"  # Green for animal
                alert_box_style = "style='border: 1px solid #4CAF50; padding: 15px; margin: 15px 0; border-radius: 5px; background-color: #E8F5E9;'"
                custom_message = f"""
                <div {alert_box_style}>
                    <h3 style='color: #4CAF50; margin-top: 0;'>🐾 ANIMAL DETECTED 🐾</h3>
                    <p>An animal has been detected in the monitored area.</p>
                    <p>Detected object: <strong>{detection_data['class_name']}</strong> with {detection_data['confidence']:.1%} confidence.</p>
                </div>
                """
                action_steps = """
                <h3>Recommended Actions:</h3>
                <ol>
                    <li>Identify the type of animal if possible</li>
                    <li>Assess if the animal poses any risk or nuisance</li>
                    <li>Take appropriate action based on local wildlife guidelines or pet policies</li>
                    <li>Ensure pets are safe if wildlife is potentially dangerous</li>
                </ol>
                """
                
            elif "fall" in roi_name_lower:
                header_bg = "#9C27B0"  # Purple for fall detection
                alert_box_style = "style='border: 1px solid #9C27B0; padding: 15px; margin: 15px 0; border-radius: 5px; background-color: #F3E5F5;'"
                custom_message = f"""
                <div {alert_box_style}>
                    <h3 style='color: #9C27B0; margin-top: 0;'>❗ FALL DETECTED - ASSISTANCE MAY BE NEEDED ❗</h3>
                    <p>A potential fall has been detected.</p>
                    <p>Detected object: <strong>{detection_data['class_name']}</strong> (likely 'person') with {detection_data['confidence']:.1%} confidence.</p>
                </div>
                """
                action_steps = """
                <h3>Immediate Actions Recommended:</h3>
                <ol>
                    <li>Check the live camera feed immediately to verify the situation</li>
                    <li>Attempt to contact the person if possible and safe</li>
                    <li>Call emergency services (ambulance/paramedics) if a fall is confirmed and assistance is needed</li>
                    <li>Notify designated emergency contacts</li>
                    <li>Preserve the recording for review</li>
                </ol>
                """

            elif "parking" in roi_name_lower:
                header_bg = "#3F51B5"  # Indigo for parking violations
                alert_box_style = "style='border: 1px solid #3F51B5; padding: 15px; margin: 15px 0; border-radius: 5px; background-color: #E8EAF6;'"
                custom_message = f"""
                <div {alert_box_style}>
                    <h3 style='color: #3F51B5; margin-top: 0;'>🅿️ PARKING VIOLATION DETECTED 🅿️</h3>
                    <p>A vehicle appears to be parked in an incorrect zone.</p>
                    <p>Detected vehicle: <strong>{detection_data['class_name']}</strong> with {detection_data['confidence']:.1%} confidence.</p>
                    <p>Violation zone: <strong>{roi_data['name']}</strong></p>
                </div>
                """
                action_steps = """
                <h3>Recommended Actions:</h3>
                <ol>
                    <li>Verify that the vehicle is incorrectly parked by checking the camera feed</li>
                    <li>Determine if the vehicle belongs to a car, motorcycle, or other category</li>
                    <li>Document the violation including time, location, and vehicle details</li>
                    <li>Notify parking enforcement personnel if applicable</li>
                    <li>Monitor how long the vehicle remains in violation</li>
                </ol>
                """
                
            elif "restricted" in roi_name_lower:
                header_bg = "#E91E63"  # Pink for restricted zone violations
                alert_box_style = "style='border: 1px solid #E91E63; padding: 15px; margin: 15px 0; border-radius: 5px; background-color: #FCE4EC;'"
                custom_message = f"""
                <div {alert_box_style}>
                    <h3 style='color: #E91E63; margin-top: 0;'>⛔ RESTRICTED ZONE VIOLATION ⛔</h3>
                    <p>A heavy vehicle or unauthorized vehicle has entered a restricted zone.</p>
                    <p>Detected vehicle: <strong>{detection_data['class_name']}</strong> with {detection_data['confidence']:.1%} confidence.</p>
                    <p>Restricted zone: <strong>{roi_data['name']}</strong></p>
                </div>
                """
                action_steps = """
                <h3>Immediate Actions Required:</h3>
                <ol>
                    <li>Confirm the type of vehicle and whether it's restricted in this zone</li>
                    <li>Contact the driver or responsible party if possible</li>
                    <li>Alert security personnel to address the violation</li>
                    <li>Document the incident including time, location, and vehicle details</li>
                    <li>Check for any road signs or barriers that may need to be improved</li>
                </ol>
                """

        # Email body with HTML, using custom content if available
        body = f"""
        <html>
        <body style='font-family: Arial, sans-serif; max-width: 650px; margin: 0 auto;'>
            <div style='background-color: {header_bg}; {header_style[7:-1]}'>
                <h2>SmartNVR Detection Alert</h2>
            </div>
            
            {custom_message}
            
            <div style='padding: 15px;'>
                <p><strong>Camera:</strong> {camera_data['name']}</p>
                <p><strong>Object:</strong> {detection_data['class_name']}</p>
                <p><strong>Confidence:</strong> {detection_data['confidence']:.2%}</p>
                <p><strong>Time:</strong> {timestamp_str}</p>
        """
        
        # Add ROI information if available
        if roi_data:
            body += f"<p><strong>Detection Zone:</strong> {roi_data['name']}</p>"
        
        # Add action steps if available
        if action_steps:
            body += action_steps
        
        # Add image if available
        image_path = detection_data['image_path']
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-ID', '<detection_image>')
                msg.attach(img)
            body += '<p><img src="cid:detection_image" width="640" /></p>'
        
        # Add video link if available
        if detection_data['video_path']:
            video_url = f"/playback?video={os.path.basename(detection_data['video_path'])}&camera={camera_data['id']}"
            body += f'<p><a href="{video_url}" style="display: inline-block; padding: 8px 16px; background-color: #0078D7; color: white; text-decoration: none; border-radius: 4px;">View Recorded Video</a></p>'
        
        body += """
            </div>
            <div style='background-color: #f5f5f5; padding: 10px; font-size: 12px; text-align: center; margin-top: 20px;'>
                <p>This is an automated message from your SmartNVR system.</p>
            </div>
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
        
        # Email body with HTML
        body = f"""
        <html>
        <body>
            <h2>SmartNVR Email Test</h2>
            <p>This is a test email from your SmartNVR system.</p>
            <p>If you're receiving this message, your email notifications are configured correctly.</p>
            <p><strong>Configuration:</strong></p>
            <ul>
                <li>SMTP Server: {smtp_server}</li>
                <li>SMTP Port: {smtp_port}</li>
                <li>Username: {smtp_username}</li>
            </ul>
            <p>This email was sent at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>Thank you for using SmartNVR!</p>
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