import json
from sqlalchemy import create_engine, Column, Integer, String, func, select
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.types import JSON
import os # Import os to check if file exists

# 1. --- Database Setup ---
DATABASE_URL = "sqlite:///./zus_outlets.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

# 2. --- SQLAlchemy ORM Model ---
class Outlet(Base):
    __tablename__ = "outlets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True, unique=True)
    address = Column(String)
    city = Column(String, index=True)
    state = Column(String, index=True)
    operating_hours = Column(String)
    services = Column(JSON) # Store services as a JSON list, e.g., ["Dine-in", "24 Hours"]

def create_db_and_tables():
    """Creates the database and tables."""
    Base.metadata.create_all(bind=engine)

def is_database_empty() -> bool:
    """Checks if the 'outlets' table is empty."""
    db = SessionLocal()
    try:
        # Create a query that counts all rows in the Outlet table
        count_stmt = select(func.count()).select_from(Outlet)
        # Execute the query and get the scalar (single value) result
        outlet_count = db.scalar(count_stmt)
        return outlet_count == 0
    except Exception as e:
        # This can happen if the table doesn't exist yet
        print(f"Database check failed (this is OK if it's the first run): {e}")
        return True # Assume it's empty if we can't query it
    finally:
        db.close()

def populate_database():
    """
    Populates the database by reading from the 'outlets.jsonl' file.
    """
    db = SessionLocal()
    INPUT_FILE = "outlets.jsonl"
    
    if not os.path.exists(INPUT_FILE):
        print(f"Warning: '{INPUT_FILE}' not found. Database will be empty.")
        print("Run 'python scraper.py' to create it.")
        db.close()
        return

    print(f"Populating database from {INPUT_FILE}...")
    
    try:
        count = 0
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON line: {line}")
                    continue

                # Check if outlet already exists
                exists = db.query(Outlet).filter(Outlet.name == data["name"]).first()
                if not exists:
                    outlet = Outlet(
                        name=data["name"],
                        address=data["address"],
                        city=data["city"],
                        state=data["state"],
                        operating_hours=data["operating_hours"],
                        services=json.dumps(data["services"]) # Store list as JSON string
                    )
                    db.add(outlet)
                    count += 1
        
        db.commit()
        if count > 0:
            print(f"Successfully added {count} new outlets to the database.")
        else:
            print("Database is already up-to-date.")
    except Exception as e:
        print(f"Error populating database: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("Initializing database...")
    create_db_and_tables()
    print("Populating database from 'outlets.jsonl'...")
    populate_database()
    print("Database setup complete.")