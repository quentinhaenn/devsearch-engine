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
  - [x] Combinaison BM25 + similarité cosinus
  - [x] Scoring pondéré configurable (60% textuel + 40% sémantique)
  - [x] Normalisation des scores
  - [x] Pipeline de recherche en 2 étapes (filtrage + précision)

**Livrables** : Recherche sémantique opérationnelle

### Phase 4 : Modèles de reranking IA

- [ ] **Cross-encoder pour reranking**
  - [ ] Intégration `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - [ ] Pipeline : Top 100 → Reranking → Top 10
  - [ ] Optimisation batch pour performance
  - [ ] Cache résultats de reranking

- [x] **Scoring composite intelligent**
  - [x] **Pertinence** : Score cross-encoder (poids 0.6)
  - [x] **Popularité** : GitHub stars, vues, citations (poids 0.3)
  - [x] **Fraîcheur** : Décroissance temporelle exponentielle (poids 0.1)
  - [x] Normalisation et combinaison des scores

- [ ] **Système de synonymes contextuels**
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
  - [ ] Métriques de latence (recherche, reranking, total)
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


### Elasticsearch optimisé

```json
{
  "mappings": {
    "properties": {
      "title": {"type": "text"},
      "content": {"type": "text"},
      "embedding": {"type": "dense_vector", "dims": 384, "similarity": "cosine"},
      "file_type": {"type": "keyword"},
      "created_at": {"type": "date"},
      "modified_at": {"type": "date"},
      "github_stars": {"type": "integer"},
      "view_count": {"type": "integer"},
      "popularity_score": {"type": "float"},
      "freshness_score": {"type": "float"}
    }
  }
}
```
