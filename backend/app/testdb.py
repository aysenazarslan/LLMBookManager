# test_db.py
import os, urllib
from sqlalchemy import create_engine, text

SERVER = r"DESKTOP-FSML3LC\MSSQLSERVER01"
DB     = "LLM"

# --- SQL Authentication ---
USER = "llmuser"
PWD  = "1q"  
conn_str = (
    f"DRIVER={{ODBC Driver 17 for SQL Server}};"
    f"SERVER={SERVER};DATABASE={DB};UID={USER};PWD={PWD};"
    "Encrypt=yes;TrustServerCertificate=yes;"
)

# --- Windows Authentication (alternatif) ---
# conn_str = (
#     f"DRIVER={{ODBC Driver 17 for SQL Server}};"
#     f"SERVER={SERVER};DATABASE={DB};Trusted_Connection=yes;"
# )

odbc_url = "mssql+pyodbc:///?odbc_connect=" + urllib.parse.quote_plus(conn_str)

engine = create_engine(odbc_url, pool_pre_ping=True, echo=False, future=True)

try:
    with engine.connect() as conn:
        ver = conn.execute(text("SELECT @@VERSION")).scalar_one()
        ok  = conn.execute(text("SELECT 1")).scalar_one()
        if ok == 1:
            print("OK")
        else:
            print("NO")
    
        print("SQL Server surum:", ver.splitlines()[0])
except Exception as e:
    print("ERROR:", e)