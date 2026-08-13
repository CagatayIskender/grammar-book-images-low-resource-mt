#!/usr/bin/env python3
import json
import subprocess

from mtob_grammar.chunking import verify_chunks
from mtob_grammar.config import ROOT, models, output_path, sources
from mtob_grammar.data import parse_igt
from mtob_grammar.experiment import CONDITIONS
from mtob_grammar.io import read_jsonl
from mtob_grammar.prompts import appendix_c_prompt, long_context, passage_context


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> None:
    errors: list[str] = []
    for source_id, config in sources().items():
        chunks_path = output_path("chunks", source_id, "chunks_512_overlap256.jsonl")
        index_path = output_path("indexes", source_id, "all_mpnet_base_v2.npy")
        gl_path = output_path("gl_contexts", source_id, "grammar_long.txt")
        require(chunks_path.is_file(), f"missing chunks for {source_id}", errors)
        require(index_path.is_file(), f"missing embedding index for {source_id}", errors)
        require(gl_path.is_file(), f"missing Gl context for {source_id}", errors)
        if not (chunks_path.is_file() and index_path.is_file() and gl_path.is_file()):
            continue
        chunks = read_jsonl(chunks_path)
        errors.extend(f"{source_id}: {item}" for item in verify_chunks(chunks))
        examples = parse_igt((ROOT / config["test_file"]).resolve(), limit=int(config["test_n"]))
        require(bool(examples), f"no test examples for {source_id}", errors)
        if not examples:
            continue
        source = examples[0].source
        retrieval_by_condition = {}
        for name in ("ge", "gs"):
            retrieval_path = output_path("manifests", source_id, f"retrieval_{name}.jsonl")
            require(retrieval_path.is_file(), f"missing {name} retrieval manifest for {source_id}", errors)
            if retrieval_path.is_file():
                retrieval = read_jsonl(retrieval_path)
                require(len(retrieval) == int(config["test_n"]), f"wrong query count in {source_id}/{name}", errors)
                require(
                    all(len(record["passages"]) == 2 for record in retrieval),
                    f"{source_id}/{name} did not retrieve exactly two passages",
                    errors,
                )
                retrieval_by_condition[name] = {int(record["index"]): record for record in retrieval}
        gl_text = gl_path.read_text(encoding="utf-8")
        for example in examples:
            contexts = {
                "ge": passage_context(
                    config["language"],
                    [item["text"] for item in retrieval_by_condition["ge"][example.index]["passages"]],
                ),
                "gs": passage_context(
                    config["language"],
                    [item["text"] for item in retrieval_by_condition["gs"][example.index]["passages"]],
                ),
                "gl": long_context(config["language"], gl_text),
            }
            for condition, context in contexts.items():
                prompt = appendix_c_prompt(config["language"], config["location"], example.source, context)
                require(
                    prompt.count(example.source) == 2,
                    f"{source_id}/{condition}/{example.index}: source does not occur exactly twice",
                    errors,
                )

    require(len(models()) * len(sources()) * len(CONDITIONS) == 60, "matrix is not 60 experiments", errors)
    job_files = sorted(output_path("jobs").glob("*/*/run_*.sh"))
    submit_files = sorted(output_path("submit").glob("submit_*.sh"))
    require(len(job_files) == 60, f"expected 60 job scripts, found {len(job_files)}", errors)
    require(len(submit_files) == 20, f"expected 20 submit scripts, found {len(submit_files)}", errors)
    for path in job_files + submit_files:
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        require(result.returncode == 0, f"bash -n failed for {path}: {result.stderr}", errors)
    backend_text = (ROOT / "mtob_grammar" / "backends.py").read_text(encoding="utf-8")
    require('"enable_thinking": False' in backend_text, "Qwen thinking control missing", errors)
    require('"reasoning": {"effort": "none", "exclude": True}' in backend_text, "OpenRouter reasoning control missing", errors)
    require('"messages": [{"role": "user", "content": prompt}]' in backend_text, "single-user-message payload missing", errors)
    for model in ("qwen3", "qwen35"):
        thinking_path = output_path("manifests", "thinking", f"{model}.json")
        require(thinking_path.is_file(), f"missing actual chat-template check for {model}", errors)
        if thinking_path.is_file():
            thinking = json.loads(thinking_path.read_text(encoding="utf-8"))
            require(thinking["enable_thinking"] is False, f"thinking not disabled for {model}", errors)
            require(thinking["think_content_empty"] is True, f"thinking content present for {model}", errors)
    if errors:
        print("\n".join(f"[ERROR] {error}" for error in errors))
        raise SystemExit(1)
    print("[OK] All extraction, retrieval, prompt, thinking, isolation, and shell validations passed")


if __name__ == "__main__":
    main()
