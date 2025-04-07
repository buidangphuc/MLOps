isort . --skip env/ && pyink . --exclude env/ && ruff check . --fix --exclude env/ && ruff format . --exclude env/
