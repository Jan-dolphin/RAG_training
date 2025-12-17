# Struktura projektu CV RAG v2.0

## 📁 Přehled adresářů

```
app_cvs/
├── 📂 src/                   # Zdrojové Python moduly
├── 📂 tests/                 # Unit testy
├── 📂 notebooks/             # Jupyter notebooky pro interaktivní práci
├── 📂 docs/                  # Dokumentace
├── 📂 data/                  # CV soubory (.docx)
├── 📂 chroma_db/            # Vector store databáze (generováno)
├── 📂 logs/                  # Trénovací logy (generováno)
├── 📂 venv/                  # Python virtual environment
├── 📄 app.py                # Chainlit aplikace (main entry point)
├── 📄 train.py              # CLI pro trénování
├── 📄 README.md             # Hlavní dokumentace
├── 📄 requirements.txt      # Python závislosti
└── 📄 .env                  # Konfigurační proměnné
```

---

## 📂 Detaily adresářů

### `src/` - Zdrojové moduly

| Soubor | Popis |
|--------|-------|
| `config.py` | Centralizovaná konfigurace (RAGConfig, AzureConfig) |
| `models.py` | Dataclass modely (Candidate, RetrievalResult, etc.) |
| `document_loader.py` | Načítání CV z DOCX souborů |
| `embeddings.py` | Azure OpenAI embeddings wrapper |
| `vector_store.py` | ChromaDB management (v2.0 zjednodušeno) |
| `parent_retriever.py` | ParentDocumentRetriever s LocalFileStore (v2.0) |
| `rag_chain.py` | RAG pipeline s LLM (LCEL) |
| `training.py` | Training pipeline orchestration |

### `notebooks/` - Jupyter notebooky 🆕

| Soubor | Popis |
|--------|-------|
| `training.ipynb` | Interaktivní krok-po-kroku trénování |
| `query.ipynb` | Interaktivní testování dotazů (simulace chatu) |

**Použití:**
```bash
# Training
jupyter notebook notebooks/training.ipynb

# Queries
jupyter notebook notebooks/query.ipynb
```

### `docs/` - Dokumentace 🆕

| Soubor | Popis |
|--------|-------|
| `CHANGES.md` | Detailní přehled změn v2.0 |
| `chainlit.md` | Uvítací zpráva pro Chainlit UI |
| `STRUCTURE.md` | Tento dokument - přehled struktury |

### `tests/` - Unit testy

| Soubor | Popis |
|--------|-------|
| `test_document_loader.py` | Testy pro načítání CV |
| `test_embeddings.py` | Testy pro embeddings |
| `test_vector_store.py` | Testy pro vector store |
| `test_parent_retriever.py` | Testy pro retriever |
| `test_rag_chain.py` | Testy pro RAG chain |
| `test_training.py` | Testy pro training pipeline |

**Spuštění:**
```bash
pytest tests/ -v
```

### `data/` - CV soubory

```
data/
└── OneDrive_2025-12-16/
    ├── Baláček_Daniel_CV_EN.docx
    ├── Bímová_Kamila_CV_EN.docx
    └── ... (další CV)
```

### `chroma_db/` - Vector store databáze (generováno při tréninku)

```
chroma_db/
├── chroma.sqlite3           # ChromaDB hlavní databáze
├── *.parquet               # ChromaDB data soubory
└── docstore/               # 🆕 LocalFileStore (v2.0)
    ├── uuid-1.txt          # Parent chunk 1
    ├── uuid-2.txt          # Parent chunk 2
    └── ...
```

**Důležité:**
- `chroma_db/` je generováno při `python train.py`
- `docstore/` obsahuje parent chunks (úplné CV context)
- Vše je persistentní - přežije restart aplikace 🆕

### `logs/` - Trénovací logy (generováno)

```
logs/
├── training_20251217_140000.log
├── training_20251217_150000.log
└── ...
```

**Formát:**
```
2025-12-17 14:00:01 - src.training - INFO - Loading documents...
2025-12-17 14:00:02 - src.document_loader - INFO - Loaded CV for Baláček Daniel
...
```

---

## 🚀 Entry pointy

### 1. Trénování

**CLI (automaticky):**
```bash
python train.py
```

**Jupyter (interaktivně):**
```bash
jupyter notebook notebooks/training.ipynb
```

### 2. Spuštění aplikace

**Chainlit UI:**
```bash
chainlit run app.py
```

### 3. Testování dotazů

**Jupyter (interaktivně):**
```bash
jupyter notebook notebooks/query.ipynb
```

**Python skript:**
```python
from src.config import AppConfig
from src.embeddings import EmbeddingsManager
from src.vector_store import VectorStoreManager
from src.parent_retriever import CVParentRetriever

config = AppConfig()
embeddings_mgr = EmbeddingsManager(config.azure)
vs_manager = VectorStoreManager(config.rag, embeddings_mgr.get_embeddings())
vectorstore = vs_manager.load_vectorstore()

retriever = CVParentRetriever(config.rag, vectorstore, config.azure)
retriever.load_from_existing_store()

results = retriever.retrieve("Python developer", top_k=5)
```

---

## 📊 Data flow

### Training flow:
```
data/*.docx
    ↓ (document_loader)
Candidate objects
    ↓ (convert_to_langchain_documents)
LangChain Documents
    ↓ (embeddings)
Embeddings
    ↓ (parent_retriever)
├─→ chroma_db/*.sqlite3       (child chunks)
└─→ chroma_db/docstore/*      (parent chunks) 🆕
```

### Query flow:
```
User query
    ↓ (embeddings)
Query embedding
    ↓ (similarity search in chroma_db)
Relevant child chunks
    ↓ (mapping via docstore) 🆕
Parent chunks (complete context)
    ↓ (rag_chain + LLM)
Final answer
```

---

## 🔧 Konfigurace

### `.env` soubor
```env
AZURE_OPENAI_ENDPOINT=https://...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002-dolphin-1
AZURE_OPENAI_API_VERSION=2023-05-15
```

### `src/config.py`
```python
@dataclass
class RAGConfig:
    parent_chunk_size: int = 2000
    child_chunk_size: int = 400
    top_k: int = 5
    persist_directory: str = "./chroma_db"
    data_directory: str = "./data/OneDrive_2025-12-16"

@dataclass
class AzureConfig:
    batch_size: int = 5      # chunks na batch
    batch_delay: float = 2.0 # delay mezi batches
```

---

## 🆕 Co je nového v v2.0

### Změny ve struktuře:

1. **`notebooks/`** - nový adresář pro Jupyter notebooky
   - `training.ipynb`
   - `query.ipynb`

2. **`docs/`** - nový adresář pro dokumentaci
   - `CHANGES.md`
   - `chainlit.md`
   - `STRUCTURE.md`

3. **`chroma_db/docstore/`** - nový podadresář
   - LocalFileStore pro parent chunks
   - Persistence mezi restarty

### Změny v kódu:

1. **`src/parent_retriever.py`**
   - LocalFileStore místo InMemoryStore
   - Odstranění loaded mode hacku
   - Vylepšený batch processing

2. **`src/vector_store.py`**
   - Zjednodušeno (-100 řádků)
   - Jen `create_or_load_vectorstore()`

3. **`src/training.py`**
   - Optimalizovaný flow
   - Odstranění redundance

---

## 📚 Další dokumentace

- **[README.md](../README.md)** - Hlavní dokumentace
- **[CHANGES.md](CHANGES.md)** - Detailní změny v2.0
- **[training.ipynb](../notebooks/training.ipynb)** - Interaktivní training
- **[query.ipynb](../notebooks/query.ipynb)** - Interaktivní queries

---

**Verze:** 2.0
**Datum:** 2025-12-17
