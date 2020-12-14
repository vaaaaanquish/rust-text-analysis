# rust-text-analysis

Rustによるlindera、neologd、fasttext、XGBoostを用いたテキスト分類のお試し.

# Usage

```
cd ./text_analysis
docker build -t rusttext .
docker run -it -v ${PWD}/data:/app/data rusttext /app/run.sh
```
