# Full run: 21 support examples, 37 test examples
# python run_grammamt_ModelGloss.py --language Gitksan --model_id swiss-ai/Apertus-8B-Instruct-2509 --support_n 21 --test_n 37 --use_float32 --out_metrics metrics/metrics_gitksan_ModelGloss_Apertus.json --out_jsonl results/results_gitksan_ModelGloss_Apertus.jsonl

# Quick test: 1 support, 1 test
python run_grammamt_ModelGloss.py --language Gitksan --model_id swiss-ai/Apertus-8B-Instruct-2509 --support_n 1 --test_n 1 --max_new_tokens 64 --use_float32 --out_metrics metrics/metrics_gitksan_ModelGloss_test.json --out_jsonl results/results_gitksan_ModelGloss_test.jsonl
