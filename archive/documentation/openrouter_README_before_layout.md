# OpenRouter grammar-context experiments

Models:

- Gemini 2.5 Flash Lite: `google/gemini-2.5-flash-lite`
- GPT-5.6 Luna: `openai/gpt-5.6-luna`

The suite reuses the GrammarMT dataset split, 21 support examples, generated grammar materials,
normal context prompts, and ModelGloss prompts. API generation runs on CPU. XCOMET scoring is a
separate GPU job.

## Secret setup

Create `/dss/dsshome1/07/ge92kun2/.config/grammamt/openrouter.env` with this single line:

```bash
export OPENROUTER_API_KEY='your-key-here'
```

Protect it with:

```bash
chmod 600 /dss/dsshome1/07/ge92kun2/.config/grammamt/openrouter.env
```

The key file is outside the project and is never written to result files or Slurm logs.

## Smoke test

This sends nine requests: two Gitksan test sentences with text and image contexts in normal and
ModelGloss modes, plus one Natugu request containing all nine selected impactful pages. The Natugu
request checks the multimodal page payload before a full run.

```bash
bash "Run Scripts/submit/submit_gemini25flashlite_smoke.sh"
```

Inspect the Slurm output and `metrics/openrouter/gemini25flashlite/smoke/` before starting full runs.

For Luna, first run:

```bash
bash "Run Scripts/submit/submit_gpt56luna_smoke.sh"
```

Luna outputs use the `gpt56luna` label. The Luna scripts omit `temperature`, which that model does
not advertise as a supported parameter, while preserving `seed=42` and disabled reasoning.

## Full suite

```bash
bash "Run Scripts/submit/submit_gemini25flashlite_full.sh"
```

For the Luna suite:

```bash
bash "Run Scripts/submit/submit_gpt56luna_full.sh"
```

This submits five CPU jobs and one dependent GPU XCOMET job. The runner appends each completed
example immediately and safely resumes partial outputs. Existing completed examples are not billed
again.

The Tsez Gemini 2.5 Flash Lite and GPT-5.6 Luna jobs use the first 99 test sentences. The source
dataset contains 445 test sentences; the reduced API subset matches Natugu, the next-largest test
set.
