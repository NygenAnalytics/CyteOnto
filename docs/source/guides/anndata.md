---
description: Compare cell type labels stored in the obs columns of an AnnData object with CyteOnto, per cell with compare_anndata or per combination of labels with compare, and avoid the pitfalls of missing values.
---

(anndata)=
# Labels in AnnData

When the author labels and the predictions of your annotation methods are columns of
`adata.obs`, compare them per cell with `compare_anndata()`, or per combination of
labels with `compare()`.

## Compare per cell

`compare_anndata()` reads each column in `target_columns` from `adata.obs` and compares
it, as one algorithm, with `author_labels`:

```python
df = await cyto.compare_anndata(
    author_labels=adata.obs["cell_type"].astype(str).tolist(),
    anndata_objects=[adata],
    target_columns=["predicted_a", "predicted_b"],
    author_column="cell_type",
    algorithm_names=["method_a", "method_b"],
    run_id="pbmc_per_cell",
)
```

- `author_labels` must have one label per row of `adata.obs`, in the same order. The
  method does not read them from `author_column`: it only checks that the column exists,
  and skips the object with a warning if it does not.
- A target column that is missing is skipped with a warning. If no column is left,
  `compare()` raises `ValueError: At least one algorithm must be provided`.
- `algorithm_names` names the algorithms in the results; by default they are the column
  names.
- The other arguments are those of `compare()`.

The results have one row per cell and algorithm. The models are called per distinct
label, not per cell: once for each distinct label of the author labels and of each
algorithm. So a large dataset costs as much as its number of distinct labels.

Pass one AnnData object per call. Each object adds the same algorithm names, so with
two objects that both have the target columns, `compare()` raises a `ValueError` for
the repeated names.

## Compare each combination of labels once

Per-cell results repeat the same pair of labels many times. To score each combination of
an author label and a prediction once, count the cells of each combination and pass the
combinations to `compare()`:

```python
# Count the cells of each combination of author label and prediction.
pairs = (
    adata.obs[["cell_type", "predicted_a"]]
    .dropna()
    .astype(str)
    .value_counts()
    .reset_index(name="n_cells")
)

# Score each combination once.
df = await cyto.compare(
    author_labels=pairs["cell_type"].tolist(),
    algorithms={"method_a": pairs["predicted_a"].tolist()},
    run_id="pbmc_pairs",
)

# Weight each combination's score by its number of cells.
score = (df["cytescore_similarity"] * pairs["n_cells"]).sum() / pairs["n_cells"].sum()
```

With one algorithm, the rows of `df` are in the order of `pairs`, so the two align by
position. The weighted score is the mean score over all cells with both labels.

## Missing values

`compare_anndata()` converts each target column to strings, so a missing value becomes a
label such as `nan`, `none` or `<na>` once lowercased, which is described and matched
like any other label.
Convert the author labels the same way, and drop or fill the rows with missing labels
first, as `dropna()` does above. Empty labels are not matched: a pair with an empty label
scores 0 and its `similarity_method` is `empty`.
