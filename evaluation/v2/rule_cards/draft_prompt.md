<!-- 規程カードの下書きのプロンプト（Issue #16・docs/v2-rule-card-procedure.md の流れ 1）。
封を開ける前に固定する。変えたら指紋が変わり、固定し直すまで封用の入口は流れない。
モデルは Fable 5.1（claude-fable-5-1・本人の決定 11/4）。温度は設定できないので、1 規程につき 1 回だけ流し、結果を選び直さない。
{{...}} は流す人が機械で埋める。規程の文と、問から切り離した結論の語の組だけを渡す。問の文・事実・正解は渡さない。 -->
<documents>
<document>
<source>{{SOURCE}}</source>
<document_content>
{{REGULATION}}
</document_content>
</document>
</documents>

<conclusion_pairs>
{{OPTION_PAIRS}}
</conclusion_pairs>

あなたは社内規程の文を、コードで判定できる「規程カード」に写す担当です。上の規程の文だけを根拠にします。文に無い事を足しません。一般常識や別の規程で補いません。

カードは、従業員が質問に書いた事実（数・日付・あり/なし）から、規程どおりの結論をコードが決めるために使います。値を拾うのは小型のモデルで、計算はコードがします。

## 手順

1. 結論を決められる定めを探し、その文をそのまま `<quotes>` に抜き出す（言い換えない）。ただし書き・例外・前提の文も一緒に抜き出す。
2. 定めごとにカードを 1 枚作り、`<cards>` に JSON の配列で書く。結論の 2 語は `<conclusion_pairs>` の組のどれかと完全に同じにする（`then` は条件を満たす時の語、`else` は満たさない時の語）。合う組が無い定めのカードは作らない。
3. 自分で確かめる: 境目のちょうどの値を式に入れて、文の「以上・超・以下・未満・以内・まで」と合うか。分子と分母が逆でないか。例外とただし書きを全部入れたか。単位はそろっているか。
4. 文の定めでも下の形で表せない物は、作らずに `<unable>` に `[{"rule": 定めの短い名前, "reason": 理由}]` で書く。無理に近い形で作らない。無ければ `[]`。

## カードの形（rule-card@1）

- `format`: "rule-card@1"。`id`: 英小文字とハイフン。`status`: "draft"。`drafted_by`: "claude-fable-5-1"。`confirmed_by`: null
- `doc`: `source`（文書の名前）・`section`（条や節）・`quote`（定めの文そのまま）
- `target`: 何の判定か。`match`: `conclusions`（結論の 2 語）・`keywords`（質問に必ず出る、その定めに固有の語。1〜3 語。ほかの定めと紛れない語）
- `slots`: 質問から拾う値。`{"名前": {"type": "number|date|bool", "unit": "数の単位（number だけ。円・日・か月・% など質問に書かれる形）", "describe": "何の値か"}}`
- `derive`（任意）: 書かれていない時に別の値から出す値。`{"名前": 式}`。例: 出勤日数 = 全労働日数 − 欠勤日数
- `constraints`（任意）: 値が満たすはずの条件。外れたら保留になる。例: 出勤日数は全労働日数以下
- `conditions`: 結論を決める条件の配列。`{"name": 条件の文, "left": 式, "op": ">|>=|<|<=|==", "right": 式}`
- `combine`: "all"（全部満たす）か "any"（どれか満たす）
- `exceptions`（任意）: `{"name", "left", "op", "right", "conclusion": 結論の語, "if_unknown": "hold|ignore"}`。成り立てば conditions より優先する。質問に書かれていない時に結論が変わりうるなら "hold"、ふつうは書かれないただし書きなら "ignore"
- `conclusion`: `{"then": 語, "else": 語}`

式は 枠の名前・数・"あり"/"なし"・`{"add": [..]}`・`{"sub": [a, b]}`・`{"mul": [..]}`・`{"div": [a, b]}`・`{"add_days": [日付, 日数]}`・`{"month_day": [年, 月, 日]}`・`{"date": "YYYY-MM-DD"}`。

割合は質問に書かれる形に合わせる: 「8割以上」で分子と分母を拾うなら `{"div": [...]}` と 0.8、質問に「80%」と書かれる値なら unit "%" と 80。金額は円にそろえ、「5万円」は 50000 と書く。時刻の型は無いので、時間は分などの数で扱えない時は `<unable>` に書く。

## 見本（架空の規程）

規程の文:「勤続6ヶ月を経過し、全労働日の8割以上出勤した者に年休を付与する。」 語の組: 付与される / 付与されない

<example>
<quotes>勤続6ヶ月を経過し、全労働日の8割以上出勤した者に年休を付与する。</quotes>
<cards>[{"format":"rule-card@1","id":"sample-nenkyu","status":"draft","drafted_by":"claude-fable-5-1","confirmed_by":null,"doc":{"source":"就業規程","section":"第1条","quote":"勤続6ヶ月を経過し、全労働日の8割以上出勤した者に年休を付与する。"},"target":"年休の付与","match":{"conclusions":["付与される","付与されない"],"keywords":["年休"]},"slots":{"勤続月数":{"type":"number","unit":"か月","describe":"入社から経過した期間"},"出勤日数":{"type":"number","unit":"日","describe":"実際に出勤した日数"},"全労働日数":{"type":"number","unit":"日","describe":"その期間の全労働日の日数"},"欠勤日数":{"type":"number","unit":"日","describe":"欠勤した日数"}},"derive":{"出勤日数":{"sub":["全労働日数","欠勤日数"]}},"constraints":[{"name":"出勤日数は全労働日数以下","left":"出勤日数","op":"<=","right":"全労働日数"},{"name":"全労働日数は 0 より大きい","left":"全労働日数","op":">","right":0}],"conditions":[{"name":"勤続6ヶ月を経過した","left":"勤続月数","op":">=","right":6},{"name":"全労働日の8割以上出勤した","left":{"div":["出勤日数","全労働日数"]},"op":">=","right":0.8}],"combine":"all","conclusion":{"then":"付与される","else":"付与されない"}}]</cards>
<unable>[]</unable>
</example>
理由の一文: 「8割以上」は分子と分母の役割を名前で分け、「以上」なので >= にした。

規程の文:「備品の申請は購入日から14日以内に行う。」 語の組: 受け付けられる / 受け付けられない

<example>
<quotes>備品の申請は購入日から14日以内に行う。</quotes>
<cards>[{"format":"rule-card@1","id":"sample-bihin","status":"draft","drafted_by":"claude-fable-5-1","confirmed_by":null,"doc":{"source":"備品規程","section":"第2条","quote":"備品の申請は購入日から14日以内に行う。"},"target":"備品の申請の期限","match":{"conclusions":["受け付けられる","受け付けられない"],"keywords":["備品"]},"slots":{"購入日":{"type":"date","describe":"備品を購入した日"},"申請日":{"type":"date","describe":"申請した日"}},"conditions":[{"name":"購入日から14日以内に申請した","left":"申請日","op":"<=","right":{"add_days":["購入日",14]}}],"combine":"all","conclusion":{"then":"受け付けられる","else":"受け付けられない"}}]</cards>
<unable>[]</unable>
</example>
理由の一文: 日数の計算はコードに任せ、値は日付のまま拾う。

規程の文:「1件5万円を超える購入は部長の承認を要する。ただし緊急時の購入は除く。」 語の組: 必要 / 不要

<example>
<quotes>1件5万円を超える購入は部長の承認を要する。ただし緊急時の購入は除く。</quotes>
<cards>[{"format":"rule-card@1","id":"sample-kounyu","status":"draft","drafted_by":"claude-fable-5-1","confirmed_by":null,"doc":{"source":"購入規程","section":"第3条","quote":"1件5万円を超える購入は部長の承認を要する。ただし緊急時の購入は除く。"},"target":"購入の部長承認","match":{"conclusions":["必要","不要"],"keywords":["部長の承認"]},"slots":{"金額":{"type":"number","unit":"円","describe":"1件の購入の金額"},"緊急時":{"type":"bool","describe":"緊急時の購入かどうか"}},"conditions":[{"name":"1件5万円を超える","left":"金額","op":">","right":50000}],"combine":"all","exceptions":[{"name":"緊急時の購入は除く","left":"緊急時","op":"==","right":"あり","conclusion":"不要","if_unknown":"ignore"}],"conclusion":{"then":"必要","else":"不要"}}]</cards>
<unable>[]</unable>
</example>
理由の一文: 「超える」は > で、ただし書きは exceptions に分けた。

## 答え方

前置きを書かず、`<quotes>`、`<cards>`、`<unable>` の 3 つだけをこの順で書きます。`<cards>` と `<unable>` の中は JSON だけにします。
