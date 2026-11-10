# Week 11: Did the home team win?

Men's college basketball pipeline. Airflow joins two Barttorvik tables and fits one fixed model. A separate agent loop then tries feature sets until the critic stops, and scores the winner once on a holdout.

The shape matches the class demo: parallel checks, a merge, a profile, a baseline, a published table, then a looping agent graph with a sealed holdout. The question, the tables, and the retry rule are different. The class loop researched credit-card fraud and had one revise path. This critic can send a weak model back to the feature planner, or an overfit model straight back to the trainer.

## Basketball terms

- **Barttorvik** is a public college-basketball statistics site. One table is who played and who won. The other is how good each team was at the end of a season.
- **Team strength (Barthag)** is that site's single rating for a team: the chance it would beat an average college team on a neutral court, from 0 to 1. Houston at 0.98 was excellent. A team at 0.20 was not.
- **Strength gap** is the home team's rating minus the opponent's. Positive means the home team was better last season. The chart is home win rate by that gap.
- **Pace (tempo)** is how fast a team plays, in possessions per 40 minutes. It does not say who is better.
- **Adjusted offense and adjusted defense** are points scored, or allowed, per 100 possessions, after accounting for the opponent. Higher offense is better. Lower defense is better, because it means fewer points allowed.

## Question

Did the home team win? Inputs are last season's ratings only. Same-game points and box-score lines are not features.

Home teams won 64.8% of these games, so accuracy is compared with an "always pick home" baseline.

## Data

Both files are public and need no key. Barttorvik rejects a bare script user agent, so the download sends a browser user agent.

- Game logs for the seasons ending 2024 and 2025: `https://barttorvik.com/getgamestats.php?year=YEAR&json=1`
- End-of-season team ratings for 2023 and 2024: `https://barttorvik.com/YEAR_team_results.csv`

Cleaning keeps the home team's row only, so each game appears once. The 2024 games are joined to 2023 ratings, and the 2025 games to 2024 ratings, on Torvik's team name. Games with a team missing from the prior season are dropped.

The published table has 9,936 home games, 362 teams, from 2023-11-06 through 2025-03-26, and no nulls. It is loaded into Postgres as `cbb_home_games` and copied to `data/published/merged_latest.csv`.

## Pipeline

Two downloads start at the same time: the game logs, and the team ratings. Each one is cleaned on its own. Nothing else starts until both are clean.

Then, in order: join the two tables, save the result in Postgres, summarize it, train the simple model, and publish the CSV the agent loop reads.

Airflow runs this every day at 6:00am Eastern. It does not go back and rerun old days. You can also start it immediately with the trigger command below.

The fixed model uses last season's team strength for both teams, and the gap between them. On the holdout, the last 15% of games by date (1,491 games):

| | Accuracy | F1 | ROC-AUC |
| --- | --- | --- | --- |
| Always pick home | 0.608 | | |
| Airflow baseline (team strength) | 0.631 | 0.734 | 0.643 |

Home teams win more often when they were the better team last year. The bars are the strength gap, home rating minus away rating.

![Home win rate by last season's strength gap](artifacts/airflow/win_rate_by_barthag_gap.png)

## Agent loop

Six steps run in order: check the table, pick features, train, score the validation slice, then a critic decides what happens next. If the score is weak, the critic sends the run back to pick a richer feature set. If the training score is much better than the validation score, it sends the run back to train again with stronger regularization. Otherwise it stops and scores the holdout once.

The loop accepts a setup when the validation ROC-AUC reaches 0.70, and it stops after 4 tries even if that never happens. The first try uses pace on purpose, because pace should not decide the winner. A weak score adds team strength. If that is still weak, the next try adds offense, defense, and pace. If the training score runs more than 0.08 ahead of the validation score, the model is retrained more cautiously instead of adding inputs. The holdout is scored once, after the loop stops.

This run:

| Attempt | Features | Validation ROC-AUC | Critic |
| --- | --- | --- | --- |
| 1 | Pace only | 0.495 | too weak, add team strength |
| 2 | Team strength | 0.665 | still under 0.70, add offense, defense, and pace |
| 3 | Strength, offense, defense, and pace | 0.671 | nothing useful left to add |

Holdout for that last setup: accuracy 0.640, F1 0.735, ROC-AUC 0.650. That is a small step past the fixed team-strength model (0.643) and past always picking the home team (0.608 accuracy). Last season's ratings help, and they are not close to a 0.70 ROC-AUC. How the teams are playing this season was left out on purpose, so the model cannot see the game it is trying to predict.

## Comparison

Three systems score the same holdout, the last 15% of games by date.

The fixed Airflow pipeline always uses team strength. It has no choices to make. Holdout ROC-AUC 0.643.

The simple agent is the class-style baseline: check the table, pick team strength, train once, stop. It has no critic. It scores 0.643, the same model as Airflow.

The adaptive critic is the system above. It throws out pace, then team strength alone, then keeps strength plus offense, defense, and pace. Holdout ROC-AUC 0.650. The extra routing is a small gain. It does not change the conclusion: last season's ratings are a limited signal.

`python -m eval.suite` runs 15 cases. Each one names its input, the expected behavior, the score, and the pass line. Some score a single routing decision, some score the holdout, and some score whether the agents handed work to the right next step. All 15 passed on this table. The page at port 8000 shows the three scores and that pass list under "Three systems, same games."

## Run it

Docker Desktop has to be running. From this folder:

```bash
cp .env.example .env
# Set AIRFLOW_UID to the output of: id -u
docker compose up --build -d
```

Airflow: [http://localhost:8088](http://localhost:8088) (`admin` / `admin`).

```bash
docker compose exec airflow-webserver airflow dags unpause cbb_home_win_pipeline
docker compose exec airflow-webserver airflow dags trigger cbb_home_win_pipeline
```

Agent graph: [http://localhost:8000](http://localhost:8000). Press **Run loop** after the DAG has published `data/published/merged_latest.csv`. The same loop, the simple baseline, and the evaluation suite from a terminal:

```bash
docker compose run --rm agent python -m agents.run
docker compose run --rm agent python -m eval.suite
```

`docker compose down` stops the stack. A dev container is defined in `.devcontainer/devcontainer.json` and uses the `agent` service.

## Screenshots

![Airflow run](docs/images/airflow-grid.png)

![Airflow graph](docs/images/airflow-graph.png)

![Agent graph](docs/images/agent-loop.png)

![Holdout evaluation](docs/images/agent-evaluation.png)
