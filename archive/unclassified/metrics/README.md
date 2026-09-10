# Metrics Layout

Metrics are grouped to match the run-script structure.

- `qwen3/`: Qwen 3 metrics from grammar-context experiments.
- `qwen35/`: Qwen 3.5 metrics from grammar-context experiments.
- `legacy/`: older baseline metrics such as Llama, Apertus, and plain baseline Qwen.

Gitksan metrics are split by grammar PDF:

```text
qwen3/gitksan/pdf1_brown/
qwen3/gitksan/pdf2_rigsby/
qwen35/gitksan/pdf1_brown/
qwen35/gitksan/pdf2_rigsby/
```

Lezgi and Natugu metrics live under:

```text
qwen3/lezgi/
qwen3/natugu/
qwen35/lezgi/
qwen35/natugu/
```
