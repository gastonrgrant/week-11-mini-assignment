.PHONY: up down trigger agent logs

up:
	docker compose up --build -d

down:
	docker compose down

trigger:
	docker compose exec airflow-webserver airflow dags unpause cbb_home_win_pipeline
	docker compose exec airflow-webserver airflow dags trigger cbb_home_win_pipeline

logs:
	docker compose logs -f --tail=100 airflow-scheduler airflow-webserver

agent:
	docker compose up --build -d agent
