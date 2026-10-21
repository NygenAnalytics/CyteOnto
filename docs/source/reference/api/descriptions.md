---
myst:
  html_meta:
    description: Reference for CellDescription, the structured description that the LLM writes for each label and each Cell Ontology term, and the text that CyteOnto embeds.
---

# Cell type descriptions

CyteOnto does not embed labels directly. The LLM first writes a `CellDescription` of
each label, and of each Cell Ontology term, and `to_sentence()` turns it into the text
that is embedded:

```text
<initialLabel> is <descriptiveName>. Function: <function> Disease relevance: <diseaseRelevance> Developmental stage: <developmentalStage>
```

The lengths below are what the prompt asks the model for; they are not validated. When
a label cannot be described after all retries, it gets a blank description, which is
not cached, and its embedding is computed from the label text instead.
[How CyteOnto works](../getting-started/how-cyteonto-works) describes the steps.

```{eval-rst}
.. autoclass:: cyteonto.CellDescription
   :class-doc-from: class
   :members: blank, is_blank, to_sentence, get_blank

   .. py:attribute:: initialLabel
      :type: str

      Required. The label, as the model copied it. The prompt asks for a verbatim
      copy, including any synonyms separated by semicolons, but CyteOnto does not
      check it.

   .. py:attribute:: descriptiveName
      :type: str

      Required. One noun phrase of at most 12 words that names the cell type.

   .. py:attribute:: function
      :type: str

      Required. One to three sentences, at most 60 words, on the primary role of
      the cell type.

   .. py:attribute:: diseaseRelevance
      :type: str

      Required. One to three sentences, at most 60 words, on the conditions that
      involve the cell type, or ``Not established``.

   .. py:attribute:: developmentalStage
      :type: str

      Required. One sentence of at most 30 words on the lineage or developmental
      window, ``Terminally differentiated`` or ``Not established``.
```
