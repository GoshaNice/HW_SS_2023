CODE = hw_asr

switch_to_macos:
	rm poetry.lock
	cat utils/pyproject_macos.txt > pyproject.toml

switch_to_linux:
	rm poetry.lock
	cat utils/pyproject_linux.txt > pyproject.toml

install:
	python3.10 -m pip install poetry
	poetry install

test:
	poetry run python -m unittest discover hw_asr/tests

lint:
	pflake8 $(CODE)

format:
	#format code
	black $(CODE)
