run:
	poetry run uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

ui:
	poetry run streamlit run ui.py
