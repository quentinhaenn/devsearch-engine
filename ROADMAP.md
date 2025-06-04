# Roadmap DevSearch Engine - Entretien BulQ AI

## 🎯 Planning 1.5 semaines (10 jours)

### Phase 1 : Foundation (Jours 1-2) ✅

- [x] Structure projet et architecture
- [x] Setup Docker + Elasticsearch
- [x] Tests de connexion ES

**Livrables** : Projet initialisé, Elasticsearch opérationnel

### Phase 2 : Moteur de recherche (Jours 3-4)

- [ ] **Elasticsearch Integration**
  - [x] Client Python (elasticsearch-py)
  - [x] Schema d'index avec dense_vector pour embeddings
  - [x] Indexation de documents (JSON)

- [ ] **Parsing de documents**
  - [x] Extraction métadonnées (titre, type, taille, dates)
  - [ ] Chunking intelligent pour gros documents
  - [x] Calcul scores de popularité (GitHub stars, vues)

**Livrables** : Recherche textuelle fonctionnelle avec métadonnées

### Phase 3 : IA et recherche sémantique

- [x] **Embeddings avec Sentence-BERT**
  - [x] Intégration sentence-transformers (`all-MiniLM-L6-v2`)
  - [x] Génération embeddings pour documents et requêtes
  - [x] Stockage dans Elasticsearch (dense_vector)

- [x] **Recherche hybride**
  - [x] Combinaison BM25 + similarité cosinus (RRF)

**Livrables** : Recherche sémantique opérationnelle

### Phase 4 : Modèles de reranking IA

- [ ] **Cross-encoder pour reranking**
  - [x] Intégration `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - [x] Pipeline : Top 100 → Reranking → Top 10
  - [x] Optimisation batch pour performance
  - [ ] Cache résultats de reranking

- [x] **Scoring composite intelligent**
  - [x] **Pertinence** : Score cross-encoder (poids 0.6)
  - [x] **Popularité** : GitHub stars, vues, citations (poids 0.3)
  - [x] **Fraîcheur** : Décroissance temporelle exponentielle (poids 0.1)
  - [x] Normalisation et combinaison des scores

- [x] **Système de synonymes contextuels**
  - [ ] Dictionnaires techniques par domaine (Dev, ML, Cloud)
  - [ ] Expansion automatique des requêtes
  - [ ] Support multi-langues (EN/FR)
  - [ ] Synonymes sémantiques via embeddings

**Livrables** : Reranking IA fonctionnel, synonymes intelligents

### Phase 5 : Optimisation et démo (Jours 9-10)

- [ ] **Interface CLI avancée**
  - [ ] Commandes complètes (`search`, `index`, `status`)
  - [ ] Options : `--semantic`, `--rerank`, `--recent`, `--boost-popular`
  - [ ] Output formaté (table, JSON) avec highlighting
  - [ ] Mode verbeux avec explication des scores

- [ ] **Dataset de démonstration riche**
  - [ ] Docs FastAPI, Django, Flask (popularité variable)
  - [ ] Guides ML/AI récents vs anciens
  - [ ] Documentation GitHub (avec stars réelles)
  - [ ] Articles Stack Overflow avec vues

- [ ] **Performance et monitoring**
  - [x] Métriques de latence (recherche, reranking, total)
  - [ ] Cache intelligent multi-niveaux
  - [ ] Logs structurés avec scoring détaillé
  - [ ] Tests de performance avec profiling

**Livrables** : Démo complète avec métriques en temps réel

## 🚀 Fonctionnalités pour la démo

### Scénarios de démonstration avancés

1. **Recherche basique** : `devsearch search "FastAPI authentication"`
2. **Sémantique + reranking** : `devsearch search "sécuriser une API" --semantic --rerank`
3. **Boost popularité** : `devsearch search "web framework" --boost-popular`
4. **Fraîcheur** : `devsearch search "Python 3.12" --recent`
5. **Synonymes contextuels** : "deploy" → "deployment", "containerization"
6. **Multilinguisme** : "authentification" ↔ "authentication"


### Elasticsearch Mapping pour configs et indexation

```python
INDEX_SETTINGS: Dict[str, Any] = {
        "analysis": {
            "filter": {
                "synonym_filter" :{
                    "type":"synonym",
                    "synonyms_path": "synonyms.txt",
                    "updateable": True  # Allows updating synonyms without reindexing
                }
            },
            "analyzer": {
                "standard_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "asciifolding", "synonym_filter", "stop"]
                }
            },
        },
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "refresh_interval": "1s",
    }

    # Mapping for the index
    INDEX_MAPPING: Dict[str, Any] = {
        "properties": {
            "id": {"type": "keyword"},
            "title": {
                "type": "text",
                "analyzer": "standard",
                "search_analyzer": "standard_analyzer",
                "fields": {
                    "keywords" : {"type": "keyword"}
                }
            },
            "content": {
                "type": "text",
                "analyzer": "standard",
                "search_analyzer": "standard_analyzer"
            },

            # vector embeddings for semantic search
            "embedding": {
                "type": "dense_vector",
                "dims": 384,  # Sentence-BERT base model dimensions. 
                "index": True,
                "similarity": "cosine" # Classic cosine similarity for NLP via embeddings. TODO: test if other similarity metrics work better.
            },
            #Metadata fields
            "file_type": {"type": "keyword"},
            "path": {"type": "keyword"},
            "language": {"type": "keyword"},
            "tags": {"type": "keyword"},
            "source": {"type": "keyword"},

            # Datetime fields
            "created_at": {"type": "date"},
            "modified_at": {"type": "date"},

            # Additional metadata for scoring
            "github_stars": {"type": "integer"},
            "view_count": {"type": "integer"},
            "popularity_score": {"type": "float"},
            "freshness_score": {"type": "float"},
        }
    }
```
