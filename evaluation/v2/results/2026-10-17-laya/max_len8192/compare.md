# C6 比較

| C1 | C2 | C1 経路正答 | C2 検出（stop=C2） | 誤遮断（sent→block/hold） | 誤送信・判断層（local_only/forbidden→A/B+allow） |
|---|---|---|---|---|---|
| rules | rules | 34/39 (87.2%) | 8/10 (80.0%) | 1/10 (10.0%) | 2/22 (9.1%) |
| http:laya-multilingual | rules | 25/39 (64.1%) | 7/10 (70.0%) | 1/10 (10.0%) | 2/22 (9.1%) |
| rules | http:laya-multilingual | 34/39 (87.2%) | 10/10 (100.0%) | 10/10 (100.0%) | 0/22 (0.0%) |
| http:laya-multilingual | http:laya-multilingual | 25/39 (64.1%) | 9/10 (90.0%) | 8/10 (80.0%) | 0/22 (0.0%) |
