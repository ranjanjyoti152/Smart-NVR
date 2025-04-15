from app import app, db
from sqlalchemy import text

with app.app_context():
    with db.engine.connect() as conn:
        conn.execute(text("ALTER TABLE camera ADD COLUMN status VARCHAR(20) DEFAULT 'offline'"))
        conn.commit()
    print("Added status column to camera table")