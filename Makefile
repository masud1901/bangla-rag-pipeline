# Makefile for Multilingual RAG System

# Use bash as the shell
SHELL := /bin/bash

# Default PDF path, can be overridden
PDF_NAME ?= your_pdf_name.pdf
# Host-side location of the PDF (on the developer machine)
PDF_PATH_HOST = data/$(PDF_NAME)
# Where that same PDF appears *inside* the rag-api container because
# docker-compose mounts ./data -> /app/data (see docker-compose.yml)
PDF_PATH_CONTAINER = /app/data/$(PDF_NAME)

.PHONY: help up down start stop build logs ingest clean ingest-clear gui api

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  up/start      Build and start all services in detached mode."
	@echo "  down/stop     Stop and remove all services."
	@echo "  build         Force a rebuild of all Docker images."
	@echo "  logs          Tail the logs of all running services."
	@echo "  ingest        Run the data ingestion pipeline. Usage: make ingest PDF_NAME=my_book.pdf"
	@echo "  ingest-clear  Run the data ingestion pipeline with --clear-index. Usage: make ingest-clear PDF_NAME=my_book.pdf"
	@echo "  gui           Start only the Streamlit GUI."
	@echo "  api           Start only the FastAPI backend."
	@echo "  clean         Remove temporary Python files and build artifacts."

up: start
start:
	@echo "Starting up all services..."
	docker compose up --build -d

down: stop
stop:
	@echo "Stopping and removing all services..."
	docker compose down

build:
	@echo "Forcing a rebuild of all Docker images..."
	docker compose build --no-cache

logs:
	@echo "Tailing logs... (Press Ctrl+C to exit)"
	docker compose logs -f

ingest:
	@echo "Running data ingestion for $(PDF_PATH_HOST)..."
	@if [ ! -f "$(PDF_PATH_HOST)" ]; then \
		echo "Error: PDF file not found at $(PDF_PATH_HOST)"; \
		exit 1; \
	fi
	docker compose exec rag-api python -m src.pipeline.ingest $(PDF_PATH_CONTAINER)

# Same as 'ingest' but clears the existing vectors first.
ingest-clear:
	@echo "Running data ingestion with --clear-index for $(PDF_PATH_HOST)..."
	@if [ ! -f "$(PDF_PATH_HOST)" ]; then \
		echo "Error: PDF file not found at $(PDF_PATH_HOST)"; \
		exit 1; \
	fi
	docker compose exec rag-api python -m src.pipeline.ingest $(PDF_PATH_CONTAINER) --clear-index

# Bring up only the GUI (useful when API is already running elsewhere)
gui:
	@echo "Starting only the Streamlit GUI..."
	docker compose up --build -d rag-gui

# Bring up only the API service
api:
	@echo "Starting only the FastAPI backend..."
	docker compose up --build -d rag-api

clean:
	@echo "Cleaning up temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "Cleanup complete."

# Set a default goal
.DEFAULT_GOAL := help 