"""High-level CyteOnto orchestrator."""

import shutil
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd  # type: ignore
from pydantic_ai import Agent
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from tqdm.auto import tqdm  # type: ignore

from . import storage
from .config import Config
from .describe import describe_cells
from .embed import embed_texts
from .logger import logger
from .models import AgentUsage, CellDescription, EmbdConfig
from .ontology import OntologyMapping, OntologySimilarity
from .paths import PathConfig, _clean_identifier

config = Config()


class CyteOnto:
    """Compare two sets of cell type annotations against the Cell Ontology.

    Typical use::

        cyto = await CyteOnto.from_config(agent=agent, embedding=embd_cfg)
        results = await cyto.compare(
            run_id="sample_run",
            author_labels=[...],
            algorithms={"algo1": [...], "algo2": [...]},
        )
    """

    def __init__(
        self,
        agent: Agent,
        embedding: EmbdConfig,
        data_dir: str | Path | None = None,
        user_dir: str | Path | None = None,
        max_description_concurrency: int = 100,
        use_pubmed_tool: bool = True,
        reasoning: bool = False,
    ) -> None:
        self.agent = agent
        self.embedding = embedding
        if self.embedding.apiKey is None:
            self.embedding = embedding.model_copy(
                update={"apiKey": config.EMBEDDING_API_KEY}
            )

        if self.embedding.provider != "ollama" and not self.embedding.apiKey:
            raise ValueError(
                f"Embedding provider '{self.embedding.provider}' requires an API key. "
                "Pass it via EmbdConfig(apiKey=...) or set EMBEDDING_MODEL_API_KEY."
            )

        self.paths = PathConfig(data_dir, user_dir)
        self.text_model: str = str(agent.model.model_name)  # type: ignore
        self.max_description_concurrency = max_description_concurrency
        self.use_pubmed_tool = use_pubmed_tool
        self.reasoning = reasoning
        self.mapping = OntologyMapping(self.paths.ontology_csv)
        self.usage = AgentUsage(agentName="CyteOnto", modelName=self.text_model)

        self._similarity: OntologySimilarity | None = None
        self._ontology_embeddings: np.ndarray | None = None
        self._ontology_ids: list[str] | None = None

    # factory + setup

    @classmethod
    async def from_config(
        cls,
        agent: Agent,
        embedding: EmbdConfig,
        data_dir: str | Path | None = None,
        user_dir: str | Path | None = None,
        force_regenerate: bool = False,
        max_description_concurrency: int = 100,
        use_pubmed_tool: bool = True,
        reasoning: bool = False,
    ) -> "CyteOnto":
        """Build a ``CyteOnto`` and make sure the ontology embeddings are on disk.

        Generates descriptions and embeddings for every CL term if missing.
        """
        self = cls(
            agent=agent,
            embedding=embedding,
            data_dir=data_dir,
            user_dir=user_dir,
            max_description_concurrency=max_description_concurrency,
            use_pubmed_tool=use_pubmed_tool,
            reasoning=reasoning,
        )
        await self._prepare_ontology(force_regenerate=force_regenerate)
        return self

    async def _prepare_ontology(self, force_regenerate: bool) -> None:
        if not all(self.paths.core_files_present().values()):
            raise FileNotFoundError(
                f"Core ontology files missing under {self.paths.data_dir}/cell_ontology/"
            )

        emb_path = self.paths.ontology_embeddings(self.text_model, self.embedding.model)
        desc_path = self.paths.ontology_descriptions(self.text_model)

        if force_regenerate:
            for p in (emb_path, desc_path):
                if p.exists():
                    logger.info(f"force_regenerate: removing {p}")
                    p.unlink()

        if not self.mapping.load():
            raise RuntimeError("Failed to load ontology CSV mapping")

        ids, joined_labels = self.mapping.ids_and_joined_labels()

        existing: dict[str, CellDescription] = (
            storage.load_descriptions(desc_path) or {}
        )
        # Drop any pre-existing blank entries so they are retried below.
        existing = {oid: d for oid, d in existing.items() if not d.is_blank()}

        missing_indices = [i for i, oid in enumerate(ids) if oid not in existing]

        if not missing_indices and emb_path.exists():
            logger.info(f"Ontology embeddings already present at {emb_path}")
            return

        if missing_indices:
            missing_labels = [joined_labels[i] for i in missing_indices]
            missing_ids = [ids[i] for i in missing_indices]
            logger.info(
                f"Generating {len(missing_indices)} ontology descriptions "
                f"({len(ids) - len(missing_indices)} cached)"
            )
            desc_list, sub_usage = await describe_cells(
                base_agent=self.agent,
                labels=missing_labels,
                use_pubmed=self.use_pubmed_tool,
                max_concurrent=self.max_description_concurrency,
                reasoning=self.reasoning,
            )
            self.usage.merge(sub_usage)
            for oid, desc in zip(missing_ids, desc_list):
                if not desc.is_blank():
                    existing[oid] = desc
            storage.save_descriptions(desc_path, existing)

        # Embed every id. Use the joined label as a last-resort fallback when
        # a description is still blank so the array stays aligned with `ids`.
        # Such ids are not cached in the JSON and will be retried next run,
        # which will then overwrite the NPZ.
        texts: list[str] = []
        fallback_count = 0
        for oid, lbl in zip(ids, joined_labels):
            desc = existing.get(oid, CellDescription.blank(label=lbl))
            if desc is not None and not desc.is_blank():
                texts.append(desc.to_sentence())
            else:
                texts.append(lbl)
                fallback_count += 1
        if fallback_count:
            logger.warning(
                f"{fallback_count} ontology terms have no description after "
                "retries. Embedding them with the label text as a fallback; "
                "they will be retried on the next CyteOnto.from_config(...)."
            )

        embeddings = await embed_texts(texts, self.embedding)
        if embeddings is None:
            raise RuntimeError("Failed to generate ontology embeddings")

        storage.save_ontology_embeddings(
            emb_path,
            embeddings,
            ids,
            extra_metadata={
                "text_model": self.text_model,
                "embedding_model": self.embedding.model,
                "embedding_provider": self.embedding.provider,
            },
        )

    # lazy loaders

    def _load_ontology_embeddings(self) -> tuple[np.ndarray, list[str]]:
        if self._ontology_embeddings is None or self._ontology_ids is None:
            path = self.paths.ontology_embeddings(self.text_model, self.embedding.model)
            loaded = storage.load_ontology_embeddings(path)
            if loaded is None:
                raise FileNotFoundError(
                    f"Ontology embeddings not found at {path}. "
                    f"Run CyteOnto.from_config(...) first."
                )
            self._ontology_embeddings, self._ontology_ids, _ = loaded
        return self._ontology_embeddings, self._ontology_ids

    def _ensure_similarity(self) -> OntologySimilarity:
        if self._similarity is None:
            self._similarity = OntologySimilarity(
                owl_path=self.paths.ontology_owl,
                embeddings_path=self.paths.ontology_embeddings(
                    self.text_model, self.embedding.model
                ),
            )
        return self._similarity

    # user label embedding with cache

    async def _embed_user_labels(
        self,
        labels: list[str],
        run_id: str,
        kind: str,
        identifier: str,
        use_cache: bool,
    ) -> np.ndarray:
        emb_path = self.paths.user_embeddings(
            run_id, kind, identifier, self.text_model, self.embedding.model
        )
        desc_path = self.paths.user_descriptions(
            run_id, kind, identifier, self.text_model
        )

        raw_existing = storage.load_descriptions(desc_path) if use_cache else None
        # Drop any pre-existing blank entries so they are retried below.
        existing: dict[str, CellDescription] = {
            lbl: d for lbl, d in (raw_existing or {}).items() if not d.is_blank()
        }

        all_real = use_cache and all(lbl in existing for lbl in labels)
        if all_real:
            cached = storage.load_user_embeddings(emb_path)
            if cached is not None and cached[1] == labels:
                logger.info(
                    f"Reusing cached embeddings for '{identifier}' ({len(labels)} labels)"
                )
                return cached[0]

        missing = [lbl for lbl in labels if lbl not in existing]
        if missing:
            logger.info(
                f"Generating {len(missing)} new descriptions for '{identifier}' "
                f"(cached: {len(labels) - len(missing)})"
            )
            new_descs, sub_usage = await describe_cells(
                base_agent=self.agent,
                labels=missing,
                use_pubmed=self.use_pubmed_tool,
                max_concurrent=self.max_description_concurrency,
                reasoning=self.reasoning,
            )
            self.usage.merge(sub_usage)
            for lbl, desc in zip(missing, new_descs):
                if not desc.is_blank():
                    existing[lbl] = desc
            storage.save_descriptions(desc_path, existing)

        # Embed every label. Blanks are not cached, so the raw label text is
        # used as a fallback to keep the array aligned. Next compare(...) run
        # will retry description generation for those labels and overwrite.
        texts: list[str] = []
        fallback_count = 0
        for lbl in labels:
            desc = existing.get(lbl, CellDescription.blank(label=lbl))
            if desc is not None and not desc.is_blank():
                texts.append(desc.to_sentence())
            else:
                texts.append(lbl)
                fallback_count += 1
        if fallback_count:
            logger.warning(
                f"{fallback_count}/{len(labels)} labels for '{identifier}' "
                "have no description after retries. Embedding them with the "
                "label text as a fallback; they will be retried on the next run."
            )

        embeddings = await embed_texts(texts, self.embedding)
        if embeddings is None:
            raise RuntimeError(f"Failed to embed labels for '{identifier}'")

        storage.save_user_embeddings(
            emb_path,
            embeddings,
            labels,
            extra_metadata={
                "text_model": self.text_model,
                "embedding_model": self.embedding.model,
                "embedding_provider": self.embedding.provider,
                "reasoning": self.reasoning,
                "run_id": run_id,
                "kind": kind,
                "identifier": identifier,
            },
        )
        return embeddings

    # matching + scoring

    def _match(
        self, query_embeddings: np.ndarray, min_similarity: float = 0.1
    ) -> list[tuple[str | None, float]]:
        onto_emb, onto_ids = self._load_ontology_embeddings()
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        sims = cosine_similarity(query_embeddings, onto_emb)
        out: list[tuple[str | None, float]] = []
        for row in sims:
            idx = int(np.argmax(row))
            score = float(row[idx])
            if score < min_similarity:
                out.append((None, score))
            else:
                out.append((str(onto_ids[idx]), score))
        return out

    @staticmethod
    def _method_for(a_id: str | None, b_id: str | None, hierarchy_score: float) -> str:
        if a_id is None and b_id is None:
            return "no_matches"
        if a_id is None or b_id is None:
            return "partial_match"
        if a_id.startswith("CL:") and b_id.startswith("CL:"):
            return "cytescore"
        return "string_similarity"

    # public API

    async def compare(
        self,
        author_labels: list[str],
        algorithms: dict[str, list[str]] | Sequence[tuple[str, list[str]]],
        *,
        run_id: str | None = None,
        metric: str = "cosine_kernel",
        metric_params: dict[str, Any] | None = None,
        min_match_similarity: float = 0.1,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Compare author labels with one or more algorithm label sets.

        Args:
            author_labels: Reference labels. One entry per item.
            algorithms: Mapping ``{algo_name: labels}`` (or sequence of pairs).
                Each label list must be the same length as ``author_labels``.
                Algorithm names must be unique.
            run_id: Unique identifier for this comparison. Used to namespace
                user caches on disk and to tag each result row. Reuse the same
                ``run_id`` across calls to benefit from caching; pass it to
                ``clear_run`` to delete the artifacts. If ``None``, an
                auto-generated UUID is used and logged.
            metric: Similarity metric; see ``OntologySimilarity`` for the list.
            metric_params: Optional per-metric parameters (e.g. Gaussian hill width).
            min_match_similarity: Threshold below which a user label is considered
                unmatched against the ontology.
            use_cache: When ``True``, reuse on-disk embeddings and descriptions if
                the labels match.

        Returns:
            DataFrame with one row per (algorithm, pair_index) combination. The
            ``run_id`` column is populated with the id used for this call (the
            caller-provided value or the auto-generated UUID).
        """
        if run_id is None:
            run_id = f"run-{uuid.uuid4()}"
            logger.info(
                f"No run_id provided. Auto-generated run_id: '{run_id}'. "
                "Reuse it for caching and pass it to clear_run to purge."
            )

        if isinstance(algorithms, dict):
            algo_items = list(algorithms.items())
        else:
            algo_items = list(algorithms)
        if not algo_items:
            raise ValueError("At least one algorithm must be provided")

        seen: set[str] = set()
        for name, _ in algo_items:
            if name in seen:
                raise ValueError(f"Duplicate algorithm name: '{name}'")
            if name == "author":
                raise ValueError("Algorithm name 'author' is reserved")
            seen.add(name)

        logger.info(
            f"compare(run_id='{run_id}', author={len(author_labels)}, "
            f"algorithms={len(algo_items)}, metric='{metric}')"
        )

        author_emb = await self._embed_user_labels(
            labels=author_labels,
            run_id=run_id,
            kind="author",
            identifier="author",
            use_cache=use_cache,
        )
        author_matches = self._match(author_emb, min_similarity=min_match_similarity)
        similarity = self._ensure_similarity()

        rows: list[dict[str, Any]] = []
        for algo_name, algo_labels in tqdm(algo_items, desc="Comparing algorithms"):
            if len(algo_labels) != len(author_labels):
                raise ValueError(
                    f"Label length mismatch for '{algo_name}': "
                    f"{len(algo_labels)} vs {len(author_labels)} author labels"
                )

            algo_emb = await self._embed_user_labels(
                labels=algo_labels,
                run_id=run_id,
                kind="algorithm",
                identifier=algo_name,
                use_cache=use_cache,
            )
            algo_matches = self._match(algo_emb, min_similarity=min_match_similarity)

            for i, (a_lbl, g_lbl) in enumerate(zip(author_labels, algo_labels)):
                a_id, a_sim = author_matches[i]
                g_id, g_sim = algo_matches[i]
                hier = (
                    similarity.similarity(
                        a_id, g_id, metric=metric, metric_params=metric_params
                    )
                    if a_id and g_id
                    else 0.0
                )
                method = self._method_for(a_id, g_id, hier)
                rows.append(
                    {
                        "run_id": run_id,
                        "algorithm": algo_name,
                        "pair_index": i,
                        "author_label": a_lbl,
                        "algorithm_label": g_lbl,
                        "author_ontology_id": a_id,
                        "author_embedding_similarity": round(a_sim, 4),
                        "algorithm_ontology_id": g_id,
                        "algorithm_embedding_similarity": round(g_sim, 4),
                        "cytescore_similarity": round(hier, 4),
                        "similarity_method": method,
                    }
                )

        df = pd.DataFrame(rows, columns=config.RESULT_COLUMNS)
        logger.info(
            f"compare done: {len(df)} rows. methods={df['similarity_method'].value_counts().to_dict()}"
        )
        return df

    async def compare_anndata(
        self,
        author_labels: list[str],
        anndata_objects: Sequence[Any],
        target_columns: list[str],
        author_column: str,
        *,
        run_id: str | None = None,
        algorithm_names: list[str] | None = None,
        metric: str = "cosine_kernel",
        metric_params: dict[str, Any] | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Pull algorithm labels out of ``adata.obs`` and delegate to ``compare``.

        ``run_id`` has the same semantics as in ``compare``: pass one to reuse
        caches or ``None`` to let the call auto-generate and log a UUID.
        """
        names = algorithm_names or list(target_columns)
        if len(names) != len(target_columns):
            raise ValueError("algorithm_names length must match target_columns length")

        collected: list[tuple[str, list[str]]] = []
        for adata in anndata_objects:
            if author_column not in adata.obs:
                logger.warning(
                    f"Author column '{author_column}' missing from AnnData; skipping"
                )
                continue
            for col, name in zip(target_columns, names):
                if col not in adata.obs:
                    logger.warning(f"Column '{col}' missing from AnnData; skipping")
                    continue
                collected.append((name, adata.obs[col].astype(str).tolist()))

        return await self.compare(
            author_labels=author_labels,
            algorithms=collected,
            run_id=run_id,
            metric=metric,
            metric_params=metric_params,
            use_cache=use_cache,
        )

    # cache utilities

    def cache_stats(self, run_id: str | None = None) -> dict[str, Any]:
        """Summarise on-disk user embedding caches.

        Args:
            run_id: If given, restrict stats to that run. Otherwise aggregates
                across all runs under ``user_dir/embeddings``.
        """
        root = self.paths.user_dir / "embeddings"
        if run_id:
            root = root / _clean_identifier(run_id)
        total_files = 0
        total_bytes = 0
        stale_files = 0
        if root.exists():
            for p in root.rglob("*.npz"):
                total_files += 1
                total_bytes += p.stat().st_size
                if storage.load_user_embeddings(p) is None:
                    stale_files += 1
        return {
            "root": str(root),
            "total_files": total_files,
            "total_size_mb": round(total_bytes / (1024 * 1024), 2),
            "stale_files": stale_files,
        }

    def clear_run(
        self,
        run_id: str,
        kind: str | None = None,
        identifier: str | None = None,
    ) -> int:
        """Delete user embeddings and descriptions for a run.

        Args:
            run_id: The run identifier used in ``compare(...)``.
            kind: ``"author"`` or ``"algorithm"``. If omitted, all kinds under
                ``run_id`` are removed.
            identifier: When ``kind`` is given, restrict to a single identifier
                (e.g. one algorithm name). Ignored if ``kind`` is ``None``.

        Returns:
            Number of files removed.
        """
        if kind not in (None, "author", "algorithm"):
            raise ValueError("kind must be 'author', 'algorithm', or None")

        removed = 0

        if kind is None:
            for top in self.paths.run_embedding_dirs(run_id):
                if top.exists():
                    removed += self._count_files(top)
                    shutil.rmtree(top)
                    logger.info(f"Removed {top}")
            return removed

        if identifier is None:
            for top in self.paths.run_kind_dirs(run_id, kind):
                if top.exists():
                    removed += self._count_files(top)
                    shutil.rmtree(top)
                    logger.info(f"Removed {top}")
            return removed

        emb_path = self.paths.user_embeddings(
            run_id, kind, identifier, self.text_model, self.embedding.model
        )
        desc_path = self.paths.user_descriptions(
            run_id, kind, identifier, self.text_model
        )
        for p in (emb_path, desc_path):
            if p.exists():
                p.unlink()
                removed += 1
                logger.info(f"Removed {p}")
        return removed

    def purge_stale(self, run_id: str | None = None) -> int:
        """Delete user NPZ files that are missing the inline labels (stale format)."""
        root = self.paths.user_dir / "embeddings"
        if run_id:
            root = root / _clean_identifier(run_id)
        removed = 0
        if not root.exists():
            return 0
        for p in root.rglob("*.npz"):
            if storage.load_user_embeddings(p) is None:
                try:
                    p.unlink()
                    removed += 1
                    logger.info(f"Removed stale cache file: {p}")
                except Exception as e:
                    logger.error(f"Could not remove {p}: {e}")
        return removed

    @staticmethod
    def _count_files(directory: Path) -> int:
        return sum(1 for _ in directory.rglob("*") if _.is_file())


__all__ = ["CyteOnto"]
