from app import app, db
from sqlalchemy import text

with app.app_context():
    with db.engine.connect() as conn:
        conn.execute(text("ALTER TABLE camera ADD COLUMN location VARCHAR(255)"))
        conn.commit()
    print("Added location column to camera table")
