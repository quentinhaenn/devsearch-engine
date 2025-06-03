# 🔍 DevSearch Engine - Moteur de recherche technique avec IA

> Moteur de recherche sémantique pour documentation technique avec reranking intelligent par IA

## 🎯 **Objectif du projet**

Créer un moteur de recherche avancé pour la documentation technique intégrant :
- **Recherche hybride** : BM25 (textuelle) + embeddings sémantiques
- **Reranking IA** : Modèles de pertinence, popularité et fraîcheur
- **Synonymes intelligents** : Expansion contextuelle des requêtes
- **Interface CLI** : Simple et performante

## 🚀 **Fonctionnalités clés**

### Recherche multicritères
```bash
# Recherche basique
devsearch search "FastAPI authentication"

# Recherche sémantique + reranking
devsearch search "sécuriser une API" --semantic --rerank

# Avec filtres temporels
devsearch search "Docker deployment" --recent --boost-popular
```

### Intelligence artificielle intégrée
- **Sentence-BERT** : Compréhension sémantique multilingue
- **Modèle de reranking** : Cross-encoder pour affiner la pertinence
- **Scoring composite** : Pertinence + Popularité + Fraîcheur
- **Synonymes contextuels** : "deploy" ↔ "déploiement", "ML" ↔ "Machine Learning"


## 🛠️ **Stack technique**

```python
# Dependencies principales
elasticsearch>=8.0.0           # Moteur de recherche distribué
sentence-transformers>=2.2.0   # Embeddings sémantiques
transformers>=4.21.0          # Modèles de reranking
torch>=1.13.0                 # Backend ML
click>=8.1.0                  # CLI framework
rich>=13.0                    # Interface utilisateur élégante
numpy>=1.21.0                 # Calculs numériques
```

## 📊 **Architecture de reranking**

```
Requête utilisateur
       ↓
1. Recherche hybride (BM25 + Vector)     → Top 100 résultats
       ↓
2. Modèle de reranking (Cross-encoder)   → Score pertinence
       ↓
3. Facteurs contextuels                  → Score popularité + fraîcheur
       ↓
4. Scoring final pondéré                 → Top 10 optimisés
```

### Modèles de reranking
- **Pertinence** : `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Popularité** : Basé sur vues, étoiles GitHub, citations
- **Fraîcheur** : Décroissance temporelle avec boost récents

## 🎯 **Cas d'usage pour la démo**

### Scénarios techniques
1. **Recherche multilingue** : "authentication" vs "authentification"
2. **Synonymes contextuels** : "containerization" trouve "Docker", "Kubernetes"
3. **Reranking intelligent** : Documentation officielle prioritaire
4. **Recherche sémantique** : "comment sécuriser" trouve "security best practices"
5. **Fraîcheur** : Articles récents sur nouvelles versions