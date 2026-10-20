---
description: The environment variables that CyteOnto reads, its logging, and the fixed retry rules, time limits, usage limits and concurrency defaults of its model requests.
---

(configuration)=
# Environment and limits

## Environment variables

| Variable | Read | Purpose | Default |
|---|---|---|---|
| `NEBIUS_API_KEY`, `DEEPINFRA_API_KEY`, `FIREWORKS_API_KEY`, `GOOGLE_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `TOGETHER_API_KEY` | When `CyteOnto` is created | The API key of the embedding provider, when `EmbdConfig.apiKey` is `None` | Unset |
| `EMBEDDING_MODEL_API_KEY` | When `cyteonto` is imported | The embedding API key when the provider's own variable is unset | Empty |
| `NCBI_API_KEY` | When `cyteonto` is imported | Sent with PubMed searches, to raise NCBI's rate limit | Empty |
| `CYTEONTO_LOG_LEVEL` | When `cyteonto` is imported | The log level, such as `DEBUG`, `INFO` or `WARNING` | `DEBUG` |
| `CYTEONTO_LOG_FILE` | When `cyteonto` is imported | A file to write the log to as well, rotated at 10 MB | Unset |

Set the variables that are read on import before the first `import cyteonto`. When it is
imported, `cyteonto` loads a `.env` file with python-dotenv: in a notebook or another
interactive session, the nearest one from the working directory up, and in a script, the
nearest one from the installed package's directory up. Variables that are already set
keep their values. In a script, call `load_dotenv()` from the `dotenv` package yourself
to load the `.env` file next to the script.

CyteOnto does not read the API key of the LLM: you give it to the pydantic-ai provider of
your agent, which may read its own variable, such as `OPENAI_API_KEY`, when you do not.

## Logging

CyteOnto logs with [loguru](https://loguru.readthedocs.io/) to standard output, as
`LEVEL: message`, and to `CYTEONTO_LOG_FILE` with timestamps. When imported, it removes
the handlers of loguru's global logger and adds its own, so configure your own loguru
handlers after importing `cyteonto`.

At the `INFO` level, the log shows the progress of descriptions and embeddings, the
generated run IDs and the files that are read and written. `DEBUG`, the default, adds
details such as missing embeddings. `WARNING` shows only failed attempts, labels that
stayed undescribed and other problems.

## Retries and limits

These rules are set in the code, mostly in `cyteonto.config.Config`. Only the
concurrency limits are options of CyteOnto.

| Rule | Value |
|---|---|
| Attempts per LLM request, to describe or to split a label | 5, with waits of 1, 2, 4 and 8 seconds |
| Time limit per LLM attempt | 120 seconds |
| Second pass over the labels whose description failed | Once, after 5 seconds, with 5 attempts per label |
| Usage limits per description or split | 50 requests and 60,000 input tokens |
| PubMed searches per description | At most 3, as the prompt asks |
| Attempts per embedding request | 3, with waits of 1 and 2 seconds |
| Time limit per embedding request | 5 minutes, the default of aiohttp |
| Concurrent LLM requests | `max_description_concurrency`, 100 by default |
| Concurrent embedding requests | `EmbdConfig.maxConcurrent`, 10 by default |

LLM requests are retried after errors of the model's API, such as HTTP and connection
errors, unexpected model behavior, output that fails validation and attempts that run out
of time. A request that exceeds the usage limits is not retried. When all attempts fail:

- **A description** is tried again in the second pass, and then by the fallback agent,
  in two passes as well, if there is one. A label that is still undescribed is embedded
  from its text and is not cached, so the next call tries again. With `use_cache=False`,
  a label with a cached description keeps it instead.
- **A split** leaves the label whole for this call, and the next call with the run
  tries again. With `use_cache=False`, a label with a cached split keeps it instead.
- **An embedding** fails the batch. CyteOnto then switches to the fallback embedding
  configuration, if there is one that differs from `embedding`, as
  [Fall back to other models](../guides/models-and-providers.md#models-fallback)
  describes, and otherwise raises a `RuntimeError`.

[Troubleshooting](../support/troubleshooting.md) lists the errors and what to do about
them.
