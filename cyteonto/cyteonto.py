"""High-level CyteOnto orchestrator."""

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd  # type: ignore
from pydantic_ai import Agent
from scipy.optimize import linear_sum_assignment  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from tqdm.auto import tqdm  # type: ignore

from . import storage
from .config import Config
from .describe import decompose_labels, describe_cells
from .embed import embed_texts
from .logger import logger
from .models import (
    AgentUsage,
    CellDescription,
    EmbdConfig,
    LlmConfig,
    ModelArtifactKey,
    ModelPairUsage,
)
from .ontology import OntologyMapping, OntologySimilarity
from .paths import PathConfig, _clean_identifier, artifact_key_segment

config = Config()


def _api_key_for_provider(provider: str, fallback: str = "") -> str | None:
    env_var = config.PROVIDER_API_KEY_ENV.get(provider)
    if env_var:
        key = os.getenv(env_var, "")
        if key:
            return key
    return fallback or None


def _is_empty(label: str) -> bool:
    return not label or not label.strip()


def _unique_parts(labels: list[str], parts_map: dict[str, list[str]]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for lbl in labels:
        if _is_empty(lbl):
            continue
        for part in parts_map.get(lbl, [lbl]):
            if part not in seen:
                seen.add(part)
                out.append(part)
    return out


def _hungarian_match_mean(
    scores: np.ndarray,
) -> tuple[float, list[tuple[int, int]]]:
    """Max-weight bipartite match mean, with coverage penalty when m != n."""
    if scores.size == 0:
        return 0.0, []
    m, n = scores.shape
    k = min(m, n)
    if k == 0:
        return 0.0, []

    row_ind, col_ind = linear_sum_assignment(1.0 - scores)
    assignments = list(zip(row_ind.tolist(), col_ind.tolist()))
    match_mean = float(scores[row_ind, col_ind].sum()) / k
    if m == n:
        return match_mean, assignments
    coverage = k / max(m, n)
    return match_mean * coverage, assignments


class CyteOnto:
    """Compare two sets of cell type annotations against the Cell Ontology.

    Typical use::

        cyto = await CyteOnto.from_config(agent=agent, embedding=embd_cfg, llm=llm_cfg)
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
        llm: LlmConfig,
        data_dir: str | Path | None = None,
        user_dir: str | Path | None = None,
        max_description_concurrency: int = 100,
        use_pubmed_tool: bool = True,
        reasoning: bool = False,
        fallback_agent: Agent | None = None,
        fallback_embedding: EmbdConfig | None = None,
        fallback_llm: LlmConfig | None = None,
    ) -> None:
        """Set up paths and cache keys without loading the ontology or calling a model.

        Args:
            agent: pydantic-ai agent whose model describes cell types and splits
                compound labels.
            embedding: Embedding provider and model. When ``embedding.apiKey`` is
                ``None``, the key is read from the provider's environment variable,
                such as ``NEBIUS_API_KEY``, or else from ``EMBEDDING_MODEL_API_KEY``.
            llm: Provider and model of ``agent``, which name the description caches.
                ``llm.model`` must equal ``agent.model.model_name``.
            data_dir: Directory with the Cell Ontology files and the ontology caches.
                Defaults to the ``data`` directory inside the installed package.
            user_dir: Directory for the caches of comparison runs. Defaults to
                ``<data_dir>/user_files``.
            max_description_concurrency: Maximum number of LLM requests that run at
                the same time.
            use_pubmed_tool: Let the agent search PubMed abstracts while it describes
                a label.
            reasoning: Leave the model's reasoning mode as the provider sets it. When
                ``False``, the requests ask the provider to turn reasoning off.
            fallback_agent: Agent that describes the labels whose description is
                still blank after the primary agent's retries.
            fallback_embedding: Embedding configuration to use when the primary
                embedding provider fails, unless it equals ``embedding``. The
                instance first loads the ontology embeddings of this model, or makes
                them from the ontology descriptions, and then keeps embedding with
                it. If that step fails, the call raises and the instance keeps the
                primary model.
            fallback_llm: Provider and model of ``fallback_agent``. It is reported in
                ``model_pair_usage``; descriptions from the fallback agent are cached
                under ``llm``.

        Raises:
            ValueError: If no embedding API key is found for a provider other than
                ``ollama``, or if ``llm.model`` differs from the agent's model name.
        """
        self.agent = agent
        self.embedding = embedding
        self.llm = llm
        self.fallback_agent = fallback_agent
        self.fallback_embedding = fallback_embedding
        self.fallback_llm = fallback_llm
        self._llm_primary_used = False

        if self.embedding.apiKey is None:
            resolved = _api_key_for_provider(
                self.embedding.provider, config.EMBEDDING_API_KEY
            )
            self.embedding = embedding.model_copy(update={"apiKey": resolved})

        if (
            self.fallback_embedding is not None
            and self.fallback_embedding.apiKey is None
        ):
            fb_key = _api_key_for_provider(
                self.fallback_embedding.provider, config.EMBEDDING_API_KEY
            )
            self.fallback_embedding = self.fallback_embedding.model_copy(
                update={"apiKey": fb_key}
            )

        if self.embedding.provider != "ollama" and not self.embedding.apiKey:
            raise ValueError(
                f"Embedding provider '{self.embedding.provider}' requires an API key. "
                "Pass it via EmbdConfig(apiKey=...) or set EMBEDDING_MODEL_API_KEY."
            )

        agent_model = str(agent.model.model_name)  # type: ignore
        if llm.model != agent_model:
            raise ValueError(
                f"LlmConfig.model ({llm.model!r}) does not match "
                f"agent.model.model_name ({agent_model!r}). "
                "Align them so cache keys match the agent in use."
            )

        self.llm_key: ModelArtifactKey = llm.to_artifact_key()
        self.embd_key: ModelArtifactKey = self.embedding.to_artifact_key()
        self.paths = PathConfig(data_dir, user_dir)
        self.max_description_concurrency = max_description_concurrency
        self.use_pubmed_tool = use_pubmed_tool
        self.reasoning = reasoning
        self.mapping = OntologyMapping(self.paths.ontology_mapping_csv)
        self.usage = AgentUsage(agentName="CyteOnto", modelName=llm.model)
        self.model_pair_usage = ModelPairUsage(llm=llm, embedding=self.embedding)

        self._similarity: OntologySimilarity | None = None
        self._ontology_embeddings: np.ndarray | None = None
        self._ontology_ids: list[str] | None = None
        # The embedding models whose ontology files force_regenerate deletes.
        self._embd_keys = [self.embd_key]
        if self.fallback_embedding is not None:
            fallback_key = self.fallback_embedding.to_artifact_key()
            if fallback_key != self.embd_key:
                self._embd_keys.append(fallback_key)
        self._embedding_switched = False
        self._embedding_primary_used = False
        # The parts of each label decomposed in the current compare call, which
        # every side of the call reuses, so that a label is split the same way on
        # each. A label whose decomposition failed keeps its cached split, or else
        # maps to itself; it is then not cached, and the next call tries again.
        self._call_decompositions: dict[str, list[str]] = {}

    def _record_llm_tier(self, used_fallback: bool) -> None:
        if not used_fallback:
            self._llm_primary_used = True
            return
        if self._llm_primary_used:
            self.model_pair_usage.llmTier = "mixed"
            if self.fallback_llm:
                self.model_pair_usage.llm = self.fallback_llm
        else:
            self.model_pair_usage.llmTier = "fallback"
            if self.fallback_llm:
                self.model_pair_usage.llm = self.fallback_llm

    async def _describe_labels(
        self, labels: list[str]
    ) -> tuple[list[CellDescription], AgentUsage]:
        if not labels:
            return [], AgentUsage(agentName="CellDescriptionAgent")
        descs, usage = await describe_cells(
            base_agent=self.agent,
            labels=labels,
            use_pubmed=self.use_pubmed_tool,
            max_concurrent=self.max_description_concurrency,
            reasoning=self.reasoning,
        )
        self._record_llm_tier(used_fallback=False)
        if self.fallback_agent:
            blank_indices = [i for i, d in enumerate(descs) if d.is_blank()]
            if blank_indices:
                retry_labels = [labels[i] for i in blank_indices]
                retry_descs, retry_usage = await describe_cells(
                    base_agent=self.fallback_agent,
                    labels=retry_labels,
                    use_pubmed=self.use_pubmed_tool,
                    max_concurrent=self.max_description_concurrency,
                    reasoning=self.reasoning,
                )
                usage.merge(retry_usage)
                for idx, desc in zip(blank_indices, retry_descs):
                    if not desc.is_blank():
                        descs[idx] = desc
                self._record_llm_tier(used_fallback=True)
        return descs, usage

    def _record_embedding_tier(self) -> None:
        """Record that labels were matched with embeddings of the current model."""
        if not self._embedding_switched:
            self._embedding_primary_used = True
            return
        self.model_pair_usage.embeddingTier = (
            "mixed" if self._embedding_primary_used else "fallback"
        )
        self.model_pair_usage.embedding = self.embedding

    async def _embed_with_failover(self, texts: list[str]) -> np.ndarray | None:
        emb = await embed_texts(texts, self.embedding)
        fallback = self.fallback_embedding
        # A fallback equal to the current configuration would fail the same way.
        if emb is not None or fallback is None or fallback == self.embedding:
            return emb
        logger.warning(
            f"Primary embedding failed (provider={self.embedding.provider}); "
            f"trying fallback ({fallback.provider})"
        )
        emb = await embed_texts(texts, fallback)
        if emb is None:
            return None
        # The labels are matched against ontology embeddings of the same model.
        # Get them before switching, so that a failure keeps the primary.
        ontology = await self._load_or_embed_ontology(fallback)
        self._switch_embedding(fallback, ontology)
        return emb

    def _switch_embedding(
        self, embedding: EmbdConfig, ontology: tuple[np.ndarray, list[str]]
    ) -> None:
        """Embed with ``embedding`` from now on, and match against ``ontology``.

        The labels, their cache files and the ontology embeddings they are matched
        to then share a model.
        """
        self.embedding = embedding
        self.embd_key = embedding.to_artifact_key()
        self._embedding_switched = True
        self._ontology_embeddings, self._ontology_ids = ontology
        self._similarity = None

    async def _load_or_embed_ontology(
        self,
        embedding: EmbdConfig,
        descriptions: dict[str, CellDescription] | None = None,
    ) -> tuple[np.ndarray, list[str]]:
        """Load the ontology embeddings of ``embedding``, or create and save them.

        After an embedding failover, labels are matched against the ontology
        embeddings of the fallback model, which ``setup`` may not have downloaded.
        They are made from ``descriptions``, or else from the ontology descriptions
        on disk.
        """
        embd_key = embedding.to_artifact_key()
        loaded = storage.load_ontology_embeddings(
            self.paths.ontology_embeddings(self.llm_key, embd_key)
        )
        desc_path = self.paths.ontology_descriptions(self.llm_key)
        if loaded is not None and storage.made_after_descriptions(loaded[2], desc_path):
            return loaded[0], loaded[1]
        if descriptions is None:
            descriptions = storage.load_descriptions(
                self.paths.ontology_descriptions(self.llm_key)
            )
            if descriptions is None:
                raise FileNotFoundError(
                    f"No ontology descriptions for {self.llm.model} to embed with "
                    f"{embedding.model}. Run CyteOnto.from_config(...) first."
                )
        ids, texts = self._ontology_texts(descriptions)
        logger.warning(
            f"Embedding the {len(ids)} ontology descriptions with {embedding.model} "
            f"({embedding.provider}) for the first time"
        )
        embeddings = await embed_texts(texts, embedding)
        if embeddings is None:
            raise RuntimeError(
                f"Failed to embed the ontology descriptions with {embedding.model}"
            )
        self._save_ontology_embeddings(embd_key, embeddings, ids)
        return embeddings, ids

    def _save_ontology_embeddings(
        self, embd_key: ModelArtifactKey, embeddings: np.ndarray, ids: list[str]
    ) -> None:
        path = self.paths.ontology_embeddings(self.llm_key, embd_key)
        saved = storage.save_ontology_embeddings(
            path,
            embeddings,
            ids,
            self.llm_key,
            embd_key,
            extra_metadata={"reasoning": self.reasoning},
        )
        if not saved:
            raise RuntimeError(f"Failed to save the ontology embeddings to {path}")

    def _ontology_texts(
        self, descriptions: dict[str, CellDescription]
    ) -> tuple[list[str], list[str]]:
        """Return the ontology IDs and the text to embed for each of them."""
        if not self.mapping.load():
            raise RuntimeError("Failed to load ontology CSV mapping")
        ids, joined_labels = self.mapping.ids_and_joined_labels()

        # Embed every id. Use the joined label as a last-resort fallback when
        # a description is still blank so the array stays aligned with `ids`.
        # Such ids are not cached in the JSON; from_config(...) describes them
        # again and embeds the terms of its model pair again.
        texts: list[str] = []
        fallback_count = 0
        for oid, lbl in zip(ids, joined_labels):
            desc = descriptions.get(oid)
            if desc is not None and not desc.is_blank():
                texts.append(desc.to_sentence())
            else:
                texts.append(lbl)
                fallback_count += 1
        if fallback_count:
            logger.warning(
                f"{fallback_count} ontology terms have no description. Embedding "
                "them with the label text as a fallback; the next "
                "CyteOnto.from_config(...) describes them again."
            )
        return ids, texts

    async def _embed_ontology(self, descriptions: dict[str, CellDescription]) -> None:
        """Embed every ontology term and save the embeddings of the model pair.

        If the primary model fails and one request to the fallback model succeeds,
        the instance switches to the ontology embeddings of the fallback model, which
        are made first if they are missing.
        """
        ids, texts = self._ontology_texts(descriptions)
        embeddings = await embed_texts(texts, self.embedding)
        if embeddings is not None:
            self._save_ontology_embeddings(self.embd_key, embeddings, ids)
            return
        fallback = self.fallback_embedding
        if fallback is None or fallback == self.embedding:
            raise RuntimeError("Failed to generate ontology embeddings")
        logger.warning(
            f"Primary embedding failed (provider={self.embedding.provider}); "
            f"using the ontology embeddings of the fallback ({fallback.provider})"
        )
        # Those may be loaded without a request, so check that the fallback works
        # before switching to it for good.
        if await embed_texts(texts[:1], fallback) is None:
            raise RuntimeError("Failed to generate ontology embeddings")
        ontology = await self._load_or_embed_ontology(fallback, descriptions)
        self._switch_embedding(fallback, ontology)

    # factory + setup

    @classmethod
    async def from_config(
        cls,
        agent: Agent,
        embedding: EmbdConfig,
        llm: LlmConfig,
        data_dir: str | Path | None = None,
        user_dir: str | Path | None = None,
        force_regenerate: bool = False,
        max_description_concurrency: int = 100,
        use_pubmed_tool: bool = True,
        reasoning: bool = False,
        fallback_agent: Agent | None = None,
        fallback_embedding: EmbdConfig | None = None,
        fallback_llm: LlmConfig | None = None,
    ) -> "CyteOnto":
        """Create a ``CyteOnto`` and make sure the ontology embeddings are on disk.

        Takes the arguments of ``CyteOnto`` and ``force_regenerate``. Loads the
        descriptions of the Cell Ontology terms for ``llm`` from ``data_dir`` and
        checks that the term embeddings for ``llm`` and ``embedding`` exist. Terms
        without a description are described with ``agent``, and ``fallback_agent``
        if given, and then every term is embedded again. When every term has a
        description and the embeddings file was made after the descriptions last
        changed, no model is called. If
        ``embedding`` fails to embed the terms, the instance checks that
        ``fallback_embedding`` works and switches to it and its ontology embeddings,
        which are made if they are missing.

        Args:
            force_regenerate: Delete the ontology descriptions of ``llm``, which all
                model pairs with this LLM share, and their embeddings with
                ``embedding`` and ``fallback_embedding`` first. The descriptions and
                the embeddings with ``embedding`` are then made again for every term,
                and those with ``fallback_embedding`` at the next failover.

        Raises:
            ValueError: For the same reasons as the constructor.
            FileNotFoundError: If ``cell_to_cell_ontology.csv``,
                ``cell_to_cell_ontology_enriched.csv`` or ``cl.owl`` is missing from
                ``<data_dir>/cell_ontology``.
            RuntimeError: If the ontology CSV cannot be read, or the term
                embeddings cannot be made or saved.
        """
        self = cls(
            agent=agent,
            embedding=embedding,
            llm=llm,
            data_dir=data_dir,
            user_dir=user_dir,
            max_description_concurrency=max_description_concurrency,
            use_pubmed_tool=use_pubmed_tool,
            reasoning=reasoning,
            fallback_agent=fallback_agent,
            fallback_embedding=fallback_embedding,
            fallback_llm=fallback_llm,
        )
        await self._prepare_ontology(force_regenerate=force_regenerate)
        return self

    async def _prepare_ontology(self, force_regenerate: bool) -> None:
        if not all(self.paths.core_files_present().values()):
            raise FileNotFoundError(
                f"Core ontology files missing under {self.paths.data_dir}/cell_ontology/"
            )

        emb_path = self.paths.ontology_embeddings(self.llm_key, self.embd_key)
        desc_path = self.paths.ontology_descriptions(self.llm_key)
        emb_paths = [
            self.paths.ontology_embeddings(self.llm_key, key) for key in self._embd_keys
        ]

        if force_regenerate:
            for p in (*emb_paths, desc_path):
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

        metadata = storage.load_embeddings_metadata(emb_path)
        current = metadata is not None and storage.made_after_descriptions(
            metadata, desc_path
        )
        if not missing_indices and current:
            logger.info(f"Ontology embeddings already present at {emb_path}")
            return
        if metadata is not None and not current:
            logger.info(f"Ontology embeddings at {emb_path} predate the descriptions")

        if missing_indices:
            missing_labels = [joined_labels[i] for i in missing_indices]
            missing_ids = [ids[i] for i in missing_indices]
            logger.info(
                f"Generating {len(missing_indices)} ontology descriptions "
                f"({len(ids) - len(missing_indices)} cached)"
            )
            desc_list, sub_usage = await self._describe_labels(missing_labels)
            self.usage.merge(sub_usage)
            described = {
                oid: desc
                for oid, desc in zip(missing_ids, desc_list)
                if not desc.is_blank()
            }
            if described:
                # Embedding files made before this are now stale for every model.
                # They stay on disk until new ones replace them.
                existing.update(described)
                storage.save_descriptions(desc_path, existing, self.llm_key)

        await self._embed_ontology(existing)

    # lazy loaders

    def _load_ontology_embeddings(self) -> tuple[np.ndarray, list[str]]:
        if self._ontology_embeddings is None or self._ontology_ids is None:
            path = self.paths.ontology_embeddings(self.llm_key, self.embd_key)
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
                    self.llm_key, self.embd_key
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
            run_id, kind, identifier, self.llm_key, self.embd_key
        )
        desc_path = self.paths.user_descriptions(run_id, kind, identifier, self.llm_key)

        # Read the file even when use_cache is False, so that rewriting it keeps
        # the descriptions that this call does not replace. Drop any pre-existing
        # blank entries so they are retried below.
        stored: dict[str, CellDescription] = {
            lbl: d
            for lbl, d in (storage.load_descriptions(desc_path) or {}).items()
            if not d.is_blank()
        }
        existing = dict(stored) if use_cache else {}

        all_real = use_cache and all(
            _is_empty(lbl) or lbl in existing for lbl in labels
        )
        if all_real:
            cached = storage.load_user_embeddings(emb_path)
            if (
                cached is not None
                and cached[1] == labels
                and storage.made_after_descriptions(cached[2], desc_path)
            ):
                logger.info(
                    f"Reusing cached embeddings for '{identifier}' ({len(labels)} labels)"
                )
                if labels:
                    self._record_embedding_tier()
                return cached[0]

        unique_labels = list(dict.fromkeys(labels))
        missing = [
            lbl for lbl in unique_labels if not _is_empty(lbl) and lbl not in existing
        ]
        if missing:
            logger.info(
                f"Generating {len(missing)} new descriptions for '{identifier}' "
                f"({len(labels)} labels -> {len(unique_labels)} unique, "
                f"{len(unique_labels) - len(missing)} cached)"
            )
            new_descs, sub_usage = await self._describe_labels(missing)
            self.usage.merge(sub_usage)
            described = {
                lbl: desc
                for lbl, desc in zip(missing, new_descs)
                if not desc.is_blank()
            }
            # A label that gets no new description keeps its cached one, which is
            # embedded below, so that the file matches the embeddings.
            existing = {**stored, **described}
            if described:
                # Embedding files made before this are now stale for every model.
                storage.save_descriptions(desc_path, existing, self.llm_key)

        # Build the text to embed for every label position. Empty labels are
        # skipped entirely (no description, no embedding) and get a zero
        # vector below. Blanks are not cached, so the raw label text is used
        # as a fallback to keep the array aligned. Next compare(...) run will
        # retry description generation for those labels and overwrite.
        texts: list[str | None] = []
        fallback_count = 0
        for lbl in labels:
            if _is_empty(lbl):
                texts.append(None)
                continue
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

        # Only embed each unique non-empty text once, then fan results back
        # out so the final array stays aligned with `labels` order/length.
        text_to_idx: dict[str, int] = {}
        unique_texts: list[str] = []
        for t in texts:
            if t is None:
                continue
            if t not in text_to_idx:
                text_to_idx[t] = len(unique_texts)
                unique_texts.append(t)
        if len(unique_texts) < len([t for t in texts if t is not None]):
            logger.info(
                f"Embedding {len(unique_texts)} unique texts for '{identifier}' "
                f"({len(texts)} total label positions)"
            )

        if unique_texts:
            embedded = await self._embed_with_failover(unique_texts)
            if embedded is None:
                raise RuntimeError(f"Failed to embed labels for '{identifier}'")
            self._record_embedding_tier()
            unique_embeddings = embedded
        else:
            onto_dim = self._load_ontology_embeddings()[0].shape[1]
            unique_embeddings = np.zeros((0, onto_dim), dtype=np.float32)
        dim = unique_embeddings.shape[1]

        emb_path = self.paths.user_embeddings(
            run_id, kind, identifier, self.llm_key, self.embd_key
        )
        embeddings = np.zeros((len(labels), dim), dtype=np.float32)
        for j, t in enumerate(texts):
            if t is not None:
                embeddings[j] = unique_embeddings[text_to_idx[t]]

        storage.save_user_embeddings(
            emb_path,
            embeddings,
            labels,
            self.llm_key,
            self.embd_key,
            extra_metadata={
                "reasoning": self.reasoning,
                "run_id": run_id,
                "kind": kind,
                "identifier": identifier,
            },
        )
        return embeddings

    # matching + scoring

    def _match(
        self,
        query_embeddings: np.ndarray,
        min_similarity: float = 0.1,
        # labels: list[str] | None = None
    ) -> list[tuple[str | None, float]]:
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        # A side whose labels are all empty has no embeddings to match.
        if len(query_embeddings) == 0:
            return []
        onto_emb, onto_ids = self._load_ontology_embeddings()
        sims = cosine_similarity(query_embeddings, onto_emb)
        out: list[tuple[str | None, float]] = []
        for i, row in enumerate(sims):
            idx = int(np.argmax(row))
            # nearest_5 = [
            #     (str(onto_ids[int(j)]), float(row[int(j)])) for j in np.argsort(row)[::-1][:5]
            # ]
            score = float(row[idx])
            if score < min_similarity:
                out.append((None, score))
            else:
                # if labels is not None:
                #     logger.info(f"ontology match for {labels[i]!r}: {onto_ids[idx]!r} {score:.4f}")
                #     logger.info(f"nearest 5: {nearest_5}")
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

    def _ontology_name_for_id(self, ontology_id: str) -> str:
        labels = self.mapping.labels_for_id(ontology_id)
        if labels:
            return labels[0]
        cls = self._ensure_similarity()._find_class(ontology_id)
        if cls is not None:
            owl_labels = getattr(cls, "label", None)
            if owl_labels:
                if isinstance(owl_labels, list):
                    return str(owl_labels[0])
                return str(owl_labels)
        return ""

    def _ontology_names_for_ids(self, ids_joined: str) -> str:
        if not ids_joined:
            return ""
        names: list[str] = []
        for oid in dict.fromkeys(ids_joined.split(";")):
            oid = oid.strip()
            if not oid:
                continue
            name = self._ontology_name_for_id(oid)
            if name:
                names.append(name)
        return ";".join(names)

    async def _resolve_label_parts(
        self,
        labels: list[str],
        run_id: str,
        use_cache: bool,
    ) -> dict[str, list[str]]:
        """Map each label to one or more cell-type parts (LLM + on-disk cache).

        Non-compound labels map to themselves, e.g. ``"T cell" -> ["T cell"]``.
        Compound labels map to sanitized parts, e.g.
        ``"AT2 cell-plasma cell doublet" -> ["AT2 cell", "plasma cell"]``.
        """
        unique = list(dict.fromkeys(lbl for lbl in labels if not _is_empty(lbl)))
        if not unique:
            return {}

        cache_path = (
            self.paths.user_dir
            / "decompositions"
            / _clean_identifier(run_id)
            / f"decompositions_{artifact_key_segment(self.llm_key)}.json"
        )
        # Read the file even when use_cache is False, so that rewriting it keeps
        # the labels of the run that this call does not decompose.
        stored: dict[str, list[str]] = {}
        if cache_path.exists():
            try:
                raw = json.loads(cache_path.read_text(encoding="utf-8"))
                if raw.get("schemaVersion") == config.SCHEMA_VERSION:
                    stored = raw.get("decompositions", {})
            except Exception as e:
                logger.warning(f"Failed to load decompositions from {cache_path}: {e}")
        cached = dict(stored) if use_cache else {}
        cached.update(self._call_decompositions)

        missing = [lbl for lbl in unique if lbl not in cached]
        if missing:
            logger.info(f"Decomposing {len(missing)} labels ({len(cached)} cached)")
            decomps, sub_usage = await decompose_labels(
                base_agent=self.agent,
                labels=missing,
                max_concurrent=self.max_description_concurrency,
                reasoning=self.reasoning,
            )
            self.usage.merge(sub_usage)
            # A decomposition without parts failed. Its label keeps its cached split,
            # or else stays whole for this call and is retried by the next one.
            for dec in decomps:
                self._call_decompositions[dec.initialLabel] = (
                    dec.parts or stored.get(dec.initialLabel) or [dec.initialLabel]
                )
            cached.update(self._call_decompositions)
            decomposed = {dec.initialLabel: dec.parts for dec in decomps if dec.parts}
            if decomposed:
                stored.update(decomposed)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                envelope = {
                    "schemaVersion": config.SCHEMA_VERSION,
                    "artifactKey": self.llm_key.model_dump(mode="json"),
                    "updatedAt": datetime.now(timezone.utc).isoformat(),
                    "decompositions": stored,
                }
                cache_path.write_text(
                    json.dumps(envelope, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

        return {lbl: cached.get(lbl, [lbl]) for lbl in unique}

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
        compound_scoring: Literal["max", "hungarian_mean"] = "max",
    ) -> pd.DataFrame:
        """Compare author labels with the labels of one or more algorithms.

        Labels are lowercased, compound labels are split into cell types, and each
        label or part is described, embedded and matched to its closest Cell
        Ontology term. Each author label is then scored against the algorithm label
        at the same position.

        Args:
            author_labels: Reference labels, one per item, such as a cell or a
                cluster.
            algorithms: Mapping ``{algo_name: labels}`` (or sequence of pairs).
                Each label list must be the same length as ``author_labels``.
                Algorithm names must be unique, and ``"author"`` is reserved.
            run_id: Unique identifier for this comparison. Used to namespace
                user caches on disk and to tag each result row. Reuse the same
                ``run_id`` across calls to benefit from caching; pass it to
                ``clear_run`` to delete the descriptions and embeddings. If ``None``,
                an id of the form ``run-<uuid4>`` is generated and logged.
            metric: Similarity metric for each pair of matched terms:
                ``cosine_kernel``, ``cosine_direct``, ``path``, ``set:jaccard``,
                ``set:cosine``, ``set:weighted_jaccard``,
                ``weighted:num_ancestors``, ``weighted:specificity``,
                ``weighted:embedding_cosine`` or ``simple``. Other names fall back
                to ``simple``.
            metric_params: Parameters of ``cosine_kernel``: the ``center``,
                ``width`` and ``amplitude`` of its Gaussian. Other metrics ignore
                them.
            min_match_similarity: Lowest cosine similarity between a label and its
                closest Cell Ontology term at which the label counts as matched.
            use_cache: When ``True``, reuse the label decompositions, descriptions
                and embeddings saved for ``run_id``. When ``False``, generate them
                again.
            compound_scoring: How to reduce the m×n part score matrix for compound
                pairs. ``"max"`` takes the maximum entry; ``"hungarian_mean"`` uses
                Hungarian assignment mean with a coverage penalty when m != n.

        Returns:
            DataFrame with one row per (algorithm, pair_index) combination. The
            ``run_id`` column is populated with the id used for this call (the
            caller-provided value or the auto-generated UUID).

        Raises:
            ValueError: If ``compound_scoring`` is invalid, ``algorithms`` is empty,
                an algorithm name is repeated or is ``"author"``, or a label list
                differs in length from ``author_labels``.
            RuntimeError: If labels cannot be embedded, or, at an embedding
                failover, the ontology embeddings of the fallback model cannot be
                made or saved; the instance then keeps the primary model.
            FileNotFoundError: If the ontology embeddings of the model pair are
                missing, or, at an embedding failover, both those of the fallback
                model and the ontology descriptions to make them from. Run
                ``from_config`` first.
        """
        if compound_scoring not in ("max", "hungarian_mean"):
            raise ValueError(
                f"compound_scoring must be 'max' or 'hungarian_mean', "
                f"got {compound_scoring!r}"
            )

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

        author_labels = [
            lbl.lower() if not _is_empty(lbl) else lbl for lbl in author_labels
        ]
        algo_items = [
            (
                name,
                [lbl.lower() if not _is_empty(lbl) else lbl for lbl in labels],
            )
            for name, labels in algo_items
        ]

        logger.info(
            f"compare(run_id='{run_id}', author={len(author_labels)}, "
            f"algorithms={len(algo_items)}, metric='{metric}')"
        )

        # Compound labels (e.g. "AT2 cell–plasma cell doublet") embed poorly as one
        # string and often score near zero. Decompose each label into cell-type parts
        # via LLM, then embed and match each part separately.
        self._call_decompositions = {}
        primary_used = self._embedding_primary_used
        # Every side is embedded before any is matched. Its parts and embeddings, with
        # the model that made them, are keyed by "author" or the algorithm name.
        parts: dict[str, tuple[dict[str, list[str]], list[str]]] = {}
        embedded: dict[str, tuple[np.ndarray, ModelArtifactKey]] = {}

        async def embed_side(identifier: str, cached: bool) -> None:
            emb = await self._embed_user_labels(
                labels=parts[identifier][1],
                run_id=run_id,
                kind="author" if identifier == "author" else "algorithm",
                identifier=identifier,
                use_cache=cached,
            )
            embedded[identifier] = (emb, self.embd_key)

        sides = [("author", author_labels)]
        for algo_name, algo_labels in algo_items:
            if len(algo_labels) != len(author_labels):
                raise ValueError(
                    f"Label length mismatch for '{algo_name}': "
                    f"{len(algo_labels)} vs {len(author_labels)} author labels"
                )
            sides.append((algo_name, algo_labels))
        for identifier, labels in tqdm(sides, desc="Embedding labels"):
            parts_map = await self._resolve_label_parts(labels, run_id, use_cache)
            parts[identifier] = (parts_map, _unique_parts(labels, parts_map))
            await embed_side(identifier, cached=use_cache)

        # An embedding failover partway through switches the instance to the
        # fallback model and its ontology embeddings. The sides embedded before it
        # are embedded again with that model, so that every label of the call is
        # matched and scored with one model. Their descriptions were saved above.
        stale = [side for side, (_, key) in embedded.items() if key != self.embd_key]
        if stale:
            # The primary model's embeddings of this call are no longer matched.
            self._embedding_primary_used = primary_used
        for identifier in stale:
            await embed_side(identifier, cached=True)

        author_parts_map, author_parts_list = parts["author"]
        author_matches = self._match(
            embedded["author"][0],
            min_similarity=min_match_similarity,
            # labels=author_parts_list
        )
        author_lookup = dict(zip(author_parts_list, author_matches))
        author_empty = [_is_empty(lbl) for lbl in author_labels]
        similarity = self._ensure_similarity()

        rows: list[dict[str, Any]] = []
        for algo_name, algo_labels in algo_items:
            algo_parts_map, algo_parts_list = parts[algo_name]
            algo_matches = self._match(
                embedded[algo_name][0],
                min_similarity=min_match_similarity,
                # labels=algo_parts_list
            )
            algo_lookup = dict(zip(algo_parts_list, algo_matches))
            algo_empty = [_is_empty(lbl) for lbl in algo_labels]

            for i, (a_lbl, g_lbl) in enumerate(zip(author_labels, algo_labels)):
                if author_empty[i] and algo_empty[i]:
                    rows.append(
                        {
                            "run_id": run_id,
                            "algorithm": algo_name,
                            "pair_index": i,
                            "author_label": a_lbl,
                            "algorithm_label": g_lbl,
                            "author_ontology_id": "",
                            "author_ontology_name": "",
                            "author_embedding_similarity": 0.0,
                            "algorithm_ontology_id": "",
                            "algorithm_ontology_name": "",
                            "algorithm_embedding_similarity": 0.0,
                            "cytescore_similarity": 0.0,
                            "similarity_method": "empty",
                        }
                    )
                    continue

                if author_empty[i] or algo_empty[i]:
                    if author_empty[i]:
                        g_parts = algo_parts_map[g_lbl]
                        g_part_sims = [algo_lookup[gp][1] for gp in g_parts]
                        g_ids = [algo_lookup[gp][0] or "" for gp in g_parts]
                        g_id_out = ";".join(g_ids)
                        g_emb_out: str | float = (
                            ";".join(str(round(s, 4)) for s in g_part_sims)
                            if len(g_parts) > 1
                            else round(g_part_sims[0], 4)
                        )
                        rows.append(
                            {
                                "run_id": run_id,
                                "algorithm": algo_name,
                                "pair_index": i,
                                "author_label": a_lbl,
                                "algorithm_label": g_lbl,
                                "author_ontology_id": "",
                                "author_ontology_name": "",
                                "author_embedding_similarity": 0.0,
                                "algorithm_ontology_id": g_id_out,
                                "algorithm_ontology_name": ";".join(
                                    self._ontology_name_for_id(oid) if oid else ""
                                    for oid in g_ids
                                ),
                                "algorithm_embedding_similarity": g_emb_out,
                                "cytescore_similarity": 0.0,
                                "similarity_method": "empty",
                            }
                        )
                    else:
                        a_parts = author_parts_map[a_lbl]
                        a_part_sims = [author_lookup[ap][1] for ap in a_parts]
                        a_ids = [author_lookup[ap][0] or "" for ap in a_parts]
                        a_id_out = ";".join(a_ids)
                        a_emb_out: str | float = (
                            ";".join(str(round(s, 4)) for s in a_part_sims)
                            if len(a_parts) > 1
                            else round(a_part_sims[0], 4)
                        )
                        rows.append(
                            {
                                "run_id": run_id,
                                "algorithm": algo_name,
                                "pair_index": i,
                                "author_label": a_lbl,
                                "algorithm_label": g_lbl,
                                "author_ontology_id": a_id_out,
                                "author_ontology_name": ";".join(
                                    self._ontology_name_for_id(oid) if oid else ""
                                    for oid in a_ids
                                ),
                                "author_embedding_similarity": a_emb_out,
                                "algorithm_ontology_id": "",
                                "algorithm_ontology_name": "",
                                "algorithm_embedding_similarity": 0.0,
                                "cytescore_similarity": 0.0,
                                "similarity_method": "empty",
                            }
                        )
                    continue

                a_parts = author_parts_map[a_lbl]
                g_parts = algo_parts_map[g_lbl]
                m, n = len(a_parts), len(g_parts)

                a_part_sims = [author_lookup[ap][1] for ap in a_parts]
                g_part_sims = [algo_lookup[gp][1] for gp in g_parts]
                is_compound = m > 1 or n > 1

                a_ids = [author_lookup[ap][0] or "" for ap in a_parts]
                g_ids = [algo_lookup[gp][0] or "" for gp in g_parts]
                a_id_out = ";".join(a_ids)
                g_id_out = ";".join(g_ids)
                a_name_out = ";".join(
                    self._ontology_name_for_id(oid) if oid else "" for oid in a_ids
                )
                g_name_out = ";".join(
                    self._ontology_name_for_id(oid) if oid else "" for oid in g_ids
                )
                if len(a_parts) > 1:
                    a_emb_out = ";".join(str(round(s, 4)) for s in a_part_sims)
                else:
                    a_emb_out = round(a_part_sims[0], 4)
                if len(g_parts) > 1:
                    g_emb_out = ";".join(str(round(s, 4)) for s in g_part_sims)
                else:
                    g_emb_out = round(g_part_sims[0], 4)

                if not is_compound:
                    a_id, _ = author_lookup[a_parts[0]]
                    g_id, _ = algo_lookup[g_parts[0]]
                    hier = (
                        similarity.similarity(
                            a_id, g_id, metric=metric, metric_params=metric_params
                        )
                        if a_id and g_id
                        else 0.0
                    )
                    method = self._method_for(a_id, g_id, hier)
                else:
                    # Build an m x n part score matrix S, then reduce by compound_scoring:
                    # - max: highest entry in S
                    # - hungarian_mean: one-to-one assignment mean; when m != n,
                    #   multiply by coverage = min(m,n) / max(m,n)
                    score_matrix = np.zeros((m, n), dtype=np.float64)
                    for ai, ap in enumerate(a_parts):
                        a_id, _ = author_lookup[ap]
                        for aj, gp in enumerate(g_parts):
                            g_id, _ = algo_lookup[gp]
                            score_matrix[ai, aj] = (
                                similarity.similarity(
                                    a_id,
                                    g_id,
                                    metric=metric,
                                    metric_params=metric_params,
                                )
                                if a_id and g_id
                                else 0.0
                            )

                    if compound_scoring == "max":
                        hier = float(score_matrix.max()) if score_matrix.size else 0.0
                    else:
                        hier, _ = _hungarian_match_mean(score_matrix)
                    method = "cytescore_compound"

                rows.append(
                    {
                        "run_id": run_id,
                        "algorithm": algo_name,
                        "pair_index": i,
                        "author_label": a_lbl,
                        "algorithm_label": g_lbl,
                        "author_ontology_id": a_id_out,
                        "author_ontology_name": a_name_out,
                        "author_embedding_similarity": a_emb_out,
                        "algorithm_ontology_id": g_id_out,
                        "algorithm_ontology_name": g_name_out,
                        "algorithm_embedding_similarity": g_emb_out,
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
        min_match_similarity: float = 0.1,
        use_cache: bool = True,
        compound_scoring: Literal["max", "hungarian_mean"] = "max",
    ) -> pd.DataFrame:
        """Pull algorithm labels out of ``adata.obs`` and delegate to ``compare``.

        Each column in ``target_columns`` becomes one algorithm. Its values are
        converted to strings, so missing values become labels such as ``"nan"``.

        Args:
            author_labels: Reference labels, one per row of ``adata.obs``.
            anndata_objects: AnnData objects to read the columns from. An object
                without ``author_column`` in ``obs`` is skipped with a warning.
            target_columns: Columns of ``adata.obs`` with the algorithm labels. A
                missing column is skipped with a warning.
            author_column: Column that each object must have in ``obs``. Only its
                presence is checked; the author labels come from
                ``author_labels``.
            run_id: As in ``compare``: pass one to reuse caches, or ``None`` to
                let the call generate and log one.
            algorithm_names: Names of the algorithms in the results, one per
                target column. Defaults to the column names.
            metric: As in ``compare``.
            metric_params: As in ``compare``.
            min_match_similarity: As in ``compare``.
            use_cache: As in ``compare``.
            compound_scoring: As in ``compare``.

        Returns:
            The DataFrame that ``compare`` returns.

        Raises:
            ValueError: If ``algorithm_names`` and ``target_columns`` differ in
                length, or for the reasons that ``compare`` raises it. Each object
                adds the same algorithm names, so several objects with the target
                columns raise it as duplicate names.
        """
        names = algorithm_names or list(target_columns)
        if len(names) != len(target_columns):
            raise ValueError("algorithm_names length must match target_columns length")

        author_labels = [
            lbl.lower() if not _is_empty(lbl) else lbl for lbl in author_labels
        ]

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
                algo_labels = adata.obs[col].astype(str).tolist()
                collected.append(
                    (
                        name,
                        [
                            lbl.lower() if not _is_empty(lbl) else lbl
                            for lbl in algo_labels
                        ],
                    )
                )

        return await self.compare(
            author_labels=author_labels,
            algorithms=collected,
            run_id=run_id,
            metric=metric,
            metric_params=metric_params,
            min_match_similarity=min_match_similarity,
            use_cache=use_cache,
            compound_scoring=compound_scoring,
        )

    # cache utilities

    def cache_stats(self, run_id: str | None = None) -> dict[str, Any]:
        """Summarise on-disk user embedding caches.

        Args:
            run_id: If given, restrict stats to that run. Otherwise aggregates
                across all runs under ``user_dir/embeddings``.

        Returns:
            A dict with ``root`` (the directory searched), ``total_files`` (the
            number of NPZ files), ``total_size_mb`` and ``stale_files`` (NPZ files
            that cannot be loaded, such as files without inline labels).
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

        The label decompositions of the run are kept.

        Args:
            run_id: The run identifier used in ``compare(...)``.
            kind: ``"author"`` or ``"algorithm"``. If omitted, all kinds under
                ``run_id`` are removed.
            identifier: When ``kind`` is given, restrict to a single identifier
                (e.g. one algorithm name) and to the files of the current model
                pair. Ignored if ``kind`` is ``None``.

        Returns:
            Number of files removed.

        Raises:
            ValueError: If ``kind`` is not ``"author"``, ``"algorithm"`` or
                ``None``.
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
            run_id, kind, identifier, self.llm_key, self.embd_key
        )
        desc_path = self.paths.user_descriptions(run_id, kind, identifier, self.llm_key)
        for p in (emb_path, desc_path):
            if p.exists():
                p.unlink()
                removed += 1
                logger.info(f"Removed {p}")
        return removed

    def purge_stale(self, run_id: str | None = None) -> int:
        """Delete user NPZ files that cannot be loaded (such as the stale format).

        Files in the stale format lack the inline labels.

        Args:
            run_id: If given, only delete files of that run.

        Returns:
            Number of files removed.
        """
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
