CODE = src

switch_to_macos:
	rm poetry.lock
	cat utils/pyproject_macos.txt > pyproject.toml

switch_to_linux:
	rm poetry.lock
	cat utils/pyproject_linux.txt > pyproject.toml

install:
	python3.10 -m pip install poetry
	poetry install

lint:
	poetry run pflake8 $(CODE)

format:
	#format code
	poetry run black $(CODE)

download_checkpoint:
	wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1yhJX9IyXZ1L1SbFbhW0gwpipghsJwI9w" -O default_test_model/model_best.pth && rm -rf /tmp/cookies.txt

test_model:
	poetry run python test.py -r default_test_model/model_best.pth -o output_test_clean.json -b 1

train:
	poetry run python train.py -c src/ss_config.json