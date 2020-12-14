# rust-text-analysis

Rustによるlindera、neologd、fasttext、XGBoostを用いたテキスト分類のお試し.

# Usage

```
docker build -t rusttext .
docker run -it -v ${PWD}/data:/app/data rusttext /app/run.sh
```
