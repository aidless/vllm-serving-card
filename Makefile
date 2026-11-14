# --- acceptance_backfill (EVAL_ACCEPTANCE_STANDARD.md P4) ---
# 证据来源: bench-lint.yml
# 命令逐字取自该来源，未增删任何参数。
.PHONY: eval

eval:
	python3 parse_results.py --selftest
