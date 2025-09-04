# models.py
from sqlalchemy.ext.automap import automap_base
from .database import engine

Base = automap_base()
Base.prepare(autoload_with=engine)

# Artık tabloları sınıf gibi kullanabilirsiniz
Book = Base.classes.Books
Chunk = Base.classes.Chunks