# Use a specific slim distribution tag and apply OS package updates to reduce known vulnerabilities
FROM python:3.11-slim-bullseye
WORKDIR /app
COPY requirements.txt .
# Ensure OS packages are up-to-date before installing Python deps
RUN apt-get update \
	&& apt-get upgrade -y \
	&& apt-get install -y --no-install-recommends ca-certificates \
	&& rm -rf /var/lib/apt/lists/* \
	&& pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "src/main.py"]