
format() {
  dir_path=$1
  echo
  echo "******* start formating ${dir_path}... *******"
  ruff check --config backend/pyproject.toml --fix "${dir_path}"
  ruff format --config backend/pyproject.toml "${dir_path}"
}

format backend/src
