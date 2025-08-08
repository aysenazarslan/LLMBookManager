from sqlalchemy import create_engine
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import sessionmaker

# MSSQL baðlantý bilgileri
DATABASE_SERVER = r"DESKTOP-FSML3LC\MSSQLSERVER01"
DATABASE_NAME = "LLM"
USERNAME = "sa"          # SQL Server kullanýcý adýn
PASSWORD = "145395"   # Þifren

# pyodbc connection string
DATABASE_URL = (
    f"mssql+pyodbc://{USERNAME}:{PASSWORD}@{DATABASE_SERVER}/{DATABASE_NAME}"
    "?driver=ODBC+Driver+17+for+SQL+Server"
)

# Engine oluþtur
engine = create_engine(DATABASE_URL)

# ORM Base
Base = automap_base()
Base.prepare(autoload_with=engine)

# Session factory
SessionLocal = sessionmaker(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()