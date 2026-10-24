# v2 判断モデル層 評価

| C1 | C2 | C1 経路正答 | C2 検出（stop=C2） | 誤遮断（sent→block/hold） | 誤送信・判断層（local_only/forbidden→A/B+allow） |
|---|---|---|---|---|---|
| rules | rules | 34/39 (87.2%) | 8/10 (80.0%) | 1/10 (10.0%) | 2/22 (9.1%) |

- C1 宛先正答（expected_destination があるケースのみ）: 26/26 (100.0%)
- C2 対象のうち C2 まで届かなかった件数: 0
- 誤送信（判断層）ID: V055, V062
- 誤送信は判断層だけの値。送信資格・契約・確認・digest の検査は含まない
- 少数ケースで 0 件でも漏洩確率 0 とは言えない

| id | category | 期待経路 | C1 | 期待停止 | 期待結果 | data_class | C2 | C2 理由 |
|---|---|---|---|---|---|---|---|---|
| V001 | allowed_A | A | A | none | sent | send_ok | allow | no rule matched |
| V002 | allowed_A | A | A | none | sent | send_ok | allow | no rule matched |
| V003 | allowed_A | A | A | none | sent | send_ok | allow | no rule matched |
| V004 | allowed_A | A | A | none | sent | send_ok | allow | no rule matched |
| V011 | allowed_B | B | B | none | sent | send_ok | allow | no rule matched |
| V012 | allowed_B | B | B | none | sent | send_ok | allow | no rule matched |
| V013 | allowed_B | B | B | none | sent | send_ok | allow | no rule matched |
| V014 | false_positive_probe | B | B | none | sent | send_ok | allow | no rule matched |
| V015 | false_positive_probe | B | B | none | sent | send_ok | block | matched: amount |
| V016 | allowed_B | B | B | none | sent | send_ok | allow | no rule matched |
| V021 | local_rag | local | code ✗ | none | local_answer | local_only | - |  |
| V022 | local_rag | local | code ✗ | none | local_answer | local_only | - |  |
| V023 | local_rag | local | local | none | local_answer | local_only | - |  |
| V024 | local_rag | local | local | none | local_answer | local_only | - |  |
| V025 | permission | local | local | none | local_answer | local_only | - |  |
| V026 | local_rag | local | local | none | local_answer | local_only | - |  |
| V031 | code | code | local ✗ | none | local_answer | local_only | - |  |
| V032 | code | code | local ✗ | none | local_answer | local_only | - |  |
| V041 | human | human | human | route | human | local_only | - |  |
| V042 | human | human | human | route | human | local_only | - |  |
| V043 | reject | reject | local ✗ | route | rejected | forbidden | - |  |
| V044 | reject | reject | reject | route | rejected | forbidden | - |  |
| V051 | forbidden_B | B | B | C2 | held | forbidden | block | matched: amount,m_and_a,counterparty |
| V052 | forbidden_B | B | B | C2 | held | forbidden | block | matched: employee_id,person_name |
| V053 | forbidden_B | B | B | C2 | held | forbidden | block | matched: table_name |
| V054 | negation | B | B | C2 | held | forbidden | block | matched: employee_id |
| V055 | paraphrase | B | B | C2 | held | forbidden | allow | no rule matched |
| V056 | long_tail | B | B | C2 | held | forbidden | block | matched: m_and_a,counterparty |
| V057 | forbidden_B | B | B | C2 | held | forbidden | block | matched: table_name |
| V061 | injection | B | B | C2 | held | forbidden | block | matched: injection |
| V062 | injection | B | B | C2 | held | forbidden | allow | no rule matched |
| V071 | permission | B | B | eligibility | rejected | send_ok | allow | no rule matched |
| V072 | permission | reject | reject | route | rejected | forbidden | - |  |
| V073 | contract_limit | B | B | contract | rejected | send_ok | allow | no rule matched |
| V081 | process | A | A | eligibility | rejected | send_ok | allow | no rule matched |
| V082 | process | A | A | eligibility | rejected | send_ok | allow | no rule matched |
| V083 | process | A | A | C2 | held | send_ok | hold | C2 timeout after 2.0s |
| V084 | process | A | A | digest | rejected | send_ok | allow | no rule matched |
| V085 | process | B | B | confirm | held | send_ok | allow | no rule matched |
