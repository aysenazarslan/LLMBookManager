# models.py
from sqlalchemy.ext.automap import automap_base
from .database import engine

Base = automap_base()
Base.prepare(autoload_with=engine)

# Artýk tablolarý sýnýf gibi kullanabilirsiniz
Book = Base.classes.Books
Chunk = Base.classes.Chunks