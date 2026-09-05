PYTHON ?= python3
export PYTHONPATH := src:.

WORKERS ?= 7

.PHONY: demo test calibrate power cuped peeking srm bandit criteo figures clean

demo: test power cuped peeking srm bandit figures
	@echo "study complete, see results/ and figures/"

test:
	$(PYTHON) -m pytest tests/ -q

calibrate:
	@test -n "$(EXPCOST_OLIST_DIR)" || (echo "set EXPCOST_OLIST_DIR to the Olist CSV directory" && exit 1)
	$(PYTHON) scripts/calibrate.py

power:
	$(PYTHON) scripts/run_power.py --workers $(WORKERS)

cuped:
	$(PYTHON) scripts/run_cuped_curve.py --workers $(WORKERS)

peeking:
	$(PYTHON) scripts/run_peeking.py --workers $(WORKERS)

srm:
	$(PYTHON) scripts/run_srm.py

bandit:
	$(PYTHON) scripts/run_bandit.py

criteo:
	$(PYTHON) scripts/run_criteo.py

figures:
	$(PYTHON) scripts/make_figures.py

clean:
	rm -rf __pycache__ .pytest_cache src/expcost/__pycache__ tests/__pycache__
