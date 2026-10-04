echo .ipynb\_checkpoints/ > .gitignore

git init

git add .

git commit -m "AI SQL Analytics Agent"

git branch -M main

gh repo create ai-sql-analytics-agent --public --source=. --remote=origin --push

