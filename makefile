export_env:
	export $(grep -v '^#' .env | xargs)
