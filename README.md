# rust-text-analysis

Rustによるlindera、neologd、fasttext、XGBoostを用いたテキスト分類のお試し.

ref: Rustによるlindera、neologd、fasttext、XGBoostを用いたテキスト分類 - stimulator https://vaaaaaanquish.hatenablog.com/entry/2020/12/14/192246

# Usage

```
cd ./text_analysis
docker build -t rusttext .
docker run -it -v ${PWD}/data:/app/data rusttext /app/run.sh
```
