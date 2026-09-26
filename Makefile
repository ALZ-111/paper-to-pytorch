# Each paper folder is its own little project: they all have a model.py and a
# test_model.py, and each test imports the one next to it. A single pytest process
# cannot do that (same module basenames, one sys.path), so the suites run one folder
# at a time. This is what CI runs too.

PAPERS := attention-is-all-you-need auto-encoding-variational-bayes neural-collaborative-filtering
PYTEST := python -m pytest -q

.PHONY: test test-utils $(PAPERS) figures clean-pyc

test: test-utils $(PAPERS)

test-utils:
	@echo "== shared utils"
	@$(PYTEST) utils

$(PAPERS):
	@echo "== $@"
	@cd papers/$@ && $(PYTEST)

figures:
	@cd papers/attention-is-all-you-need && python visualize.py --no-model
	@cd papers/auto-encoding-variational-bayes && python visualize.py
	@cd papers/neural-collaborative-filtering && python visualize.py

clean-pyc:
	@find . -name __pycache__ -type d -prune -exec rm -rf {} + -o -name '*.pyc' -delete
