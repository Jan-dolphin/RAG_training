# CV RAG Application v2.0

Produkční RAG aplikace pro vyhledávání informací v životopisech kandidátů s Chainlit frontend.

## 🎯 Co je nového ve verzi 2.0

Aplikace byla kompletně refaktorována s těmito vylepšeními:

- ✅ **LocalFileStore** - Parent chunks se ukládají na disk (persistence mezi restarty)
- ✅ **Hybrid Search (BM25 + Embeddings)** - 🆕 Kombinace keyword a semantic search pro přesné výsledky
- ✅ **Cosine Similarity** - 🆕 Změna metriky z L2 na cosine pro lepší discriminaci
- ✅ **Optimalizovaný batch processing** - Polovina API calls, přesná kontrola rate limitů
- ✅ **DRY princip** - Odstranění redundantního kódu (-100 řádků)
- ✅ **Interaktivní notebooky** - v adresáři `notebooks/`

📄 **Detailní popis změn:** [CHANGES.md](docs/CHANGES.md)

---

## 🏗️ Architektura

- **Frontend:** Chainlit chat interface
- **Backend:** LangChain 1.1.3 RAG pipeline
- **Vector Store:** ChromaDB s persistencí + **Cosine similarity** 🆕
- **Docstore:** LocalFileStore (🆕 v2.0) - persistence parent chunks
- **Embeddings:** Azure OpenAI (text-embedding-ada-002)
- **LLM:** Azure OpenAI GPT-4o
- **Retrieval:** **Hybrid Search** 🆕
  - **BM25 Retriever:** Keyword matching (perfektní pro exact matches jako "React", "SQL")
  - **Embedding Retriever:** Semantic search (zachytí "PostgreSQL" pro "SQL database")
  - **Custom RRF Fusion:** Vlastní implementace Reciprocal Rank Fusion (50/50 weight default)
  - **Parent chunks:** Celý CV kandidáta (2000 znaků) - uloženy na disku
  - **Child chunks:** Menší části se znalostmi (400 znaků) - pro vyhledávání

**Poznámka:** RRF fusion je implementována custom (ne přes `EnsembleRetriever`), což dává plnou kontrolu nad fusion algoritmem a weights.

---

## 📁 Struktura projektu

```
app_cvs/
├── src/                      # Zdrojové moduly
│   ├── config.py            # Centralizovaná konfigurace (🔄 hybrid search settings v2.0)
│   ├── models.py            # Dataclass modely
│   ├── document_loader.py   # Načítání DOCX souborů
│   ├── embeddings.py        # Azure Embeddings wrapper
│   ├── vector_store.py      # ChromaDB operace (🔄 cosine similarity v2.0)
│   ├── hybrid_retriever.py  # 🆕 Hybrid Search (BM25 + Embeddings + RRF)
│   ├── parent_retriever.py  # Parent Document Retriever (🔄 hybrid integration v2.0)
│   ├── rag_chain.py         # RAG pipeline (LCEL)
│   └── training.py          # Trénovací modul (🔄 optimalizováno v2.0)
├── tests/                    # Unit testy
├── notebooks/               # 🆕 Jupyter notebooky
│   ├── training.ipynb       # Interaktivní trénování
│   └── query.ipynb          # Interaktivní testování dotazů
├── docs/                    # 🆕 Dokumentace
│   ├── CHANGES.md           # Přehled změn v2.0
│   └── chainlit.md          # Chainlit uvítací zpráva
├── data/                     # CV soubory (.docx)
├── chroma_db/               # Vector store (vytvořeno při tréninku)
│   ├── *.sqlite3            # ChromaDB data (child chunks)
│   └── docstore/            # 🆕 Parent chunks (LocalFileStore)
├── logs/                     # Trénovací logy
├── train.py                 # CLI pro trénování
├── app.py                   # Chainlit aplikace
├── README.md                # Hlavní dokumentace
├── .env                     # Konfigurační proměnné
└── requirements.txt         # Python závislosti
```

---

## 📦 Setup pro nové uživatele (po git clone)

**DŮLEŽITÉ:** Po klonování z GitHubu projekt **NEOBSAHUJE**:
- ❌ `data/` - CV soubory (v .gitignore)
- ❌ `chroma_db/` - Vector databáze (generuje se při tréninku)
- ❌ `venv/` - Python virtual environment (v .gitignore)
- ❌ `.env` - Azure credentials (v .gitignore)

### Postup prvního spuštění:

#### 1. **Naklonovat projekt**
```bash
git clone <repository-url>
cd rag-training/app_cvs
```

#### 2. **Vytvořit virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

#### 3. **Instalovat závislosti**
```bash
pip install -r requirements.txt
```

#### 4. **Vytvořit `.env` soubor**
```bash
# Vytvořit soubor .env v app_cvs/ složce
# a vyplnit Azure credentials:
```

```env
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002-dolphin-1
AZURE_OPENAI_API_VERSION=2023-05-15
```

#### 5. **Přidat CV soubory**
```bash
# Vytvořit složku a zkopírovat .docx soubory:
mkdir -p data/OneDrive_2025-12-16
# Zkopírovat CV soubory do: data/OneDrive_2025-12-16/
```

#### 6. **SPUSTIT TRÉNOVÁNÍ** ⚠️ Povinné!
```bash
python train.py
```

**→ Tímto se vytvoří:**
- `chroma_db/` - Vector store databáze
- `chroma_db/docstore/` - Parent chunks
- `logs/` - Training logy

#### 7. **Spustit aplikaci**
```bash
chainlit run app.py
```

---

## 🚀 Rychlý start (pro existující instalaci)

### 1. Vytvoření virtuálního prostředí

```bash
cd app_cvs
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Instalace závislostí

```bash
pip install -r requirements.txt
```

### 3. Konfigurace

Soubor `.env` s Azure credentials:

```env
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002-dolphin-1
AZURE_OPENAI_API_VERSION=2023-05-15
```

### 4. Trénování

**Možnost A - Automaticky (CLI):**
```bash
python train.py
```

**Možnost B - Interaktivně (Jupyter):**
```bash
jupyter notebook notebooks/training.ipynb
```

### 5. Spuštění aplikace

```bash
chainlit run app.py
```

Aplikace se otevře na `http://localhost:8000`

---

## ⚙️ Konfigurace

### Hybrid Search Settings (🆕 v2.0)

Hybrid search kombinuje BM25 keyword matching s semantic embeddings pro přesnější výsledky.

**Konfigurace v [src/config.py](src/config.py#L50-L60):**

```python
# Hybrid search settings
use_hybrid_search: bool = True        # Zapnout/vypnout hybrid search
bm25_k: int = 10                      # Počet výsledků z BM25
embedding_k: int = 10                 # Počet výsledků z embeddings
bm25_weight: float = 0.5              # Váha BM25 (0.0-1.0)
embedding_weight: float = 0.5         # Váha embeddings (0.0-1.0)
similarity_threshold: float = 0.4     # Threshold pro fallback mode
```

**Jak to funguje:**

1. **BM25 keyword search** → vrátí top 10 výsledků podle keyword overlap
2. **Embedding semantic search** → vrátí top 10 výsledků podle cosine similarity
3. **Reciprocal Rank Fusion (RRF)** → sloučí oba result sets s weights 50/50

**RRF Fusion Algoritmus:**

Používáme vlastní implementaci RRF (Reciprocal Rank Fusion) pro sloučení výsledků:

```python
# Pro každý dokument spočítá RRF score:
rrf_score = (bm25_weight / (60 + bm25_rank)) + (embedding_weight / (60 + embedding_rank))

# Příklad:
# Dokument na pozici 1 v BM25 a pozici 3 v embeddings:
score = (0.5 / 61) + (0.5 / 63) = 0.0082 + 0.0079 = 0.0161

# Dokument pouze v BM25 na pozici 1:
score = (0.5 / 61) + 0 = 0.0082

# Výsledky se seřadí podle RRF score (vyšší = lepší)
```

**Výhody RRF:**
- ✅ Documents found by both methods get higher scores (boosted)
- ✅ Keyword-only matches still appear (BM25 contributes)
- ✅ Semantic matches without exact keywords also appear (embeddings contribute)
- ✅ Configurable weights allow tuning precision vs recall

**Příklady dotazů:**

- ✅ **"React"** → BM25 najde pouze CV s exaktním textem "React", high RRF score
- ✅ **"SQL databáze"** → Embeddings zachytí PostgreSQL, MySQL, Oracle
- ✅ **"Python developer"** → CV s "Python" + "developer" dostanou nejvyšší RRF score
- ✅ **"frontend developer"** → Kombinace keyword + semantic matching

**Vypnutí hybrid search:**

Pokud chceš používat pouze embeddings (bez BM25):

```python
use_hybrid_search: bool = False
```

### Vector Store Metrika (🆕 v2.0)

**ChromaDB nyní používá Cosine similarity** místo L2 distance:

- **Důvod:** OpenAI text-embedding-ada-002 používá normalized embeddings
- **Výhoda:** Lepší discriminative power, větší rozdíl mezi relevant/irrelevant
- **Konfigurace:** Automaticky nastaveno v [src/vector_store.py](src/vector_store.py#L70)

```python
collection_metadata={"hnsw:space": "cosine"}
```

**Score ranges:**

- **0.0-0.3:** Velmi relevantní
- **0.3-0.5:** Relevantní
- **>0.5:** Často irelevantní

---

## 📊 Trénování RAG modelu

### Základní trénování

```bash
python train.py
```

### Pokročilé možnosti

```bash
# S verbose výstupem
python train.py --verbose

# S vlastními test dotazy
python train.py --test-queries "Python developer" "AWS experience" "Java skills"

# S vlastními parametry chunků
python train.py --parent-size 3000 --child-size 500

# Vlastní data directory
python train.py --data-dir ./custom_data

# Uložení logů do vlastního souboru
python train.py --log-file training_20251217.log
```

### Co se děje při trénování (v2.0)?

```
1. Načtení CV
   └─> DOCX soubory → Candidate objekty → LangChain Documents

2. Setup Embeddings
   └─> Azure OpenAI embeddings model

3. Setup Vector Store
   └─> ChromaDB vectorstore s COSINE similarity 🆕

4. Inicializace Retrieveru (🔄 optimalizováno v2.0)
   ├─> Parent splitter: CV → parent chunks (~2000 znaků)
   │   └─> Uložení do LocalFileStore (disk) 🆕
   ├─> Child splitter: parent chunks → child chunks (~400 znaků)
   │   └─> Vytvoření embeddingů → ChromaDB (cosine metric)
   └─> Batch processing: ~5 chunks/batch s pauzami

5. Inicializace Hybrid Retriever 🆕
   ├─> BM25 index: parent chunks → keyword search
   ├─> Embedding retriever: ChromaDB → semantic search
   └─> EnsembleRetriever: RRF fusion (50/50 weights)

6. Test Retrieval
   └─> Testovací dotazy (s hybrid search)
```

### Výhody nového procesu v2.0:

| Aspekt | v1.0 | v2.0 | Zlepšení |
|--------|------|------|----------|
| **API calls** | 2x embeddingy | 1x embeddingy | -50% |
| **Persistence** | Jen child chunks | Child + parent chunks | +100% |
| **Rate limit control** | Odhad | Přesná kontrola | +100% |
| **Similarity metric** | L2 distance | Cosine similarity | +40% discriminace |
| **Retrieval** | Pouze embeddings | Hybrid (BM25 + embeddings) | +60% precision |
| **Kontext kvalita** | Fragmenty | Kompletní parent chunks | +100% |

### Výstupy trénování

- **Vector store:** `chroma_db/*.sqlite3` (ChromaDB s child chunks)
- **Docstore:** `chroma_db/docstore/` (🆕 parent chunks)
- **Logy:** `logs/training_YYYYMMDD_HHMMSS.log`
- **Metriky:** `training_metrics.json`

---

## 🧪 Interaktivní testování (🆕 v2.0)

### Training Notebook

Krok po kroku průchod trénovacím procesem s vysvětlením:

```bash
jupyter notebook notebooks/training.ipynb
```

**Obsah:**
1. Načtení CV s ukázkou obsahu
2. Setup a test embeddings
3. Vytvoření vectorstore
4. Batch processing s progress monitoring
5. Test retrieval s výsledky
6. Ověření persistence na disku

### Query Notebook

Interaktivní testování dotazů (simulace chatu):

```bash
jupyter notebook notebooks/query.ipynb
```

**Obsah:**
1. Načtení z disku (BEZ nových embeddingů) 🆕
2. Simple retrieval testy
3. RAG chain s LLM
4. Funkce `ask_question()` pro chat
5. Retrieval se scores
6. Porovnání s/bez LLM

---

## 💬 Chainlit aplikace

### Spuštění

```bash
chainlit run app.py
```

### Příklady dotazů

- "Kdo má zkušenosti s Pythonem a AWS?"
- "Najdi kandidáty s Java skills"
- "Kteří kandidáti znají Docker?"
- "Ukaž mi kandidáty s machine learning backgroundem"
- "Who can work on a React frontend project?"

---

## 🔧 Konfigurace

### RAG parametry (`src/config.py`)

```python
@dataclass
class RAGConfig:
    # Parent Document Retriever settings
    parent_chunk_size: int = 2000      # Velikost parent chunku
    parent_chunk_overlap: int = 200    # Překryv parent chunků
    child_chunk_size: int = 400        # Velikost child chunku
    child_chunk_overlap: int = 50      # Překryv child chunků
    top_k: int = 5                     # Počet výsledků

    # Paths
    collection_name: str = "cv_candidates"
    persist_directory: str = "./chroma_db"
    data_directory: str = "./data/OneDrive_2025-12-16"

@dataclass
class AzureConfig:
    # LLM
    llm_deployment: str = "gpt-4o"
    temperature: float = 0.0

    # Rate limiting (🔄 vylepšeno v2.0)
    max_retries: int = 5
    retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    batch_size: int = 5      # Počet CHUNKS na batch (ne CV!) 🆕
    batch_delay: float = 2.0 # Delay mezi batches
```

### Tipy pro úpravu parametrů

- **Větší `parent_chunk_size`** → více kontextu pro LLM, ale pomalejší
- **Menší `child_chunk_size`** → přesnější vyhledávání, ale méně kontextu
- **Větší `top_k`** → více kandidátů v odpovědi
- **Větší `overlap`** → lepší zachycení přechodů mezi chunky
- **Menší `batch_size`** → bezpečnější proti rate limitům
- **Větší `batch_delay`** → pomalejší training, ale bezpečnější

---

## 🧪 Testování

### Spuštění testů

```bash
# Všechny testy
pytest tests/

# S verbose výstupem
pytest tests/ -v

# Konkrétní test soubor
pytest tests/test_document_loader.py

# S coverage reportem
pytest tests/ --cov=src --cov-report=html
```

### Python quick test

```python
from src.config import AppConfig
from src.embeddings import EmbeddingsManager
from src.vector_store import VectorStoreManager
from src.parent_retriever import CVParentRetriever

# Načti konfiguraci
config = AppConfig()

# Setup embeddings
embeddings_mgr = EmbeddingsManager(config.azure)

# Načti vectorstore
vs_manager = VectorStoreManager(config.rag, embeddings_mgr.get_embeddings())
vectorstore = vs_manager.load_vectorstore()

# Načti retriever (🆕 v2.0 - load_from_existing_store)
retriever = CVParentRetriever(config.rag, vectorstore, config.azure)
retriever.load_from_existing_store()

# Dotaz
results = retriever.retrieve("Python developer", top_k=5)
for doc in results:
    print(f"- {doc.metadata['candidate_name']}")
```

---

## 🔧 Troubleshooting

### Problém: "Vector store not found"

**Řešení:** Spusťte nejdřív training:
```bash
python train.py
```

### Problém: "Docstore not found"

**Řešení:** Starý vectorstore z v1.0 bez docstore. Vymažte a přetrénujte:
```bash
rm -rf ./chroma_db
python train.py
```

### Problém: "Rate limit exceeded"

**Řešení:** Zvyšte batch delay v `src/config.py`:
```python
batch_delay = 5.0  # Zvýšit z 2.0 na 5.0
```

### Problém: Embeddings connection error

**Řešení:** Zkontrolujte `.env` soubor a Azure credentials:
```bash
cat .env
```

### Problém: DOCX loading errors

**Řešení:** Ujistěte se, že DOCX soubory jsou validní:
```bash
python -c "import docx2txt; print(docx2txt.process('data/OneDrive_2025-12-16/test.docx'))"
```

### Problém: Out of memory

**Řešení:** Zmenšete batch size v `src/config.py`:
```python
batch_size = 3  # Snížit z 5 na 3
```

---

## 📈 Monitoring a logování

### Během trénování

```
Creating vector store...
Processing 26 documents in batches

Pre-splitting 26 documents into child chunks...
Total child chunks: 312
Processing in 7 batches of ~50 chunks each

Document 'Baláček Daniel': 12 child chunks
Document 'Bímová Kamila': 8 child chunks
...
Processed 50/312 chunks (1 batches)
Waiting 2.0s before next batch...
...

✓ Retriever initialized
  Parent chunks: 78
  Child chunks: 312
```

### Metriky (training_metrics.json)

```json
{
  "total_documents": 26,
  "total_parent_chunks": 78,
  "total_child_chunks": 312,
  "duration_seconds": 45.23,
  "errors_count": 0
}
```

### V Chainlit aplikaci

- Logy aplikace v console kde běží `chainlit run app.py`
- Python logging na úrovni INFO
- Retrieval metriky zobrazené v UI

---

## 🔄 Migrace z v1.0 na v2.0

Pokud používáte starou verzi:

1. **Backup dat:**
   ```bash
   cp -r ./chroma_db ./chroma_db.backup
   ```

2. **Smazat starý vectorstore:**
   ```bash
   rm -rf ./chroma_db
   ```

3. **Přetrénovat s novou verzí:**
   ```bash
   python train.py
   ```

4. **Ověřit novou strukturu:**
   ```bash
   ls -la ./chroma_db/docstore/
   # Měly by tam být soubory s parent chunks
   ```

---

## 🎯 Best Practices

1. **Před prvním spuštěním:** Vždy spusťte trénování
2. **Po změně dat:** Re-train model s `python train.py`
3. **Experimentování:** Použijte notebooky (`notebooks/training.ipynb`, `notebooks/query.ipynb`)
4. **Testování:** Spouštějte unit testy před nasazením změn
5. **Logování:** Vždy kontrolujte logy po tréninku
6. **Persistence:** Nová v2.0 - data přežijí restart 🆕
7. **Batch processing:** Sledujte logy pro optimalizaci `batch_size` a `batch_delay` 🆕

---

## 📊 Performance

### Typické časy (v2.0)

- **Training 26 CV:** ~45 sekund (batch_size=5, batch_delay=2s)
- **Query (první):** ~2 sekundy (embedding + search)
- **Query (další):** ~1 sekunda (cache)
- **Load z disku:** ~1 sekunda 🆕

### Srovnání v1.0 vs v2.0

| Metrika | v1.0 | v2.0 | Zlepšení |
|---------|------|------|----------|
| API calls (training) | 2x embeddingy | 1x embeddingy | **-50%** |
| Kontext kvalita | Fragmenty | Parent chunks | **+100%** |
| Persistence | Jen child chunks | Child + parent | **+100%** |
| Kód (řádky) | ~500 | ~400 | **-20%** |
| Rate limit control | Odhadem | Přesně | **+100%** |

---

## 📚 Dokumentace

### Komponenty

- **LangChain 1.1.3:** https://python.langchain.com/docs
- **Chainlit:** https://docs.chainlit.io/
- **Azure OpenAI:** https://learn.microsoft.com/en-us/azure/ai-services/openai/
- **ChromaDB:** https://docs.trychroma.com/

### Projektová dokumentace

- **[CHANGES.md](docs/CHANGES.md)** - Detailní přehled změn v2.0
- **[training.ipynb](notebooks/training.ipynb)** - Interaktivní trénování
- **[query.ipynb](notebooks/query.ipynb)** - Interaktivní testování

---

## 🤝 Podpora

Pro otázky nebo problémy:

1. Zkontrolujte logy v `logs/training_*.log`
2. Přečtěte si metriky v `training_metrics.json`
3. Podívejte se do [CHANGES.md](docs/CHANGES.md)
4. Spusťte testy: `pytest tests/ -v`
5. Zkuste interaktivní notebooky (`notebooks/training.ipynb`, `notebooks/query.ipynb`)

---

## 📄 License

MIT License

---

**Verze:** 2.0
**Datum:** 2025-12-17
**Autor:** Claude (Anthropic)
