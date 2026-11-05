"""正解の枠値をコードに入れた時の上限。"""
import json,os,sys,collections
from decimal import Decimal
from datetime import date
os.environ['RULE_CARDS_V2']='true'
sys.path[:0]=['backend','.']
from evaluation.gpu_local import numeric_replay as nr
from app.services import rule_cards as rc
base=os.path.expanduser('~/.cache/lr-gpu')
cards=nr.load_cards(base+'/v2c/cards')
gold=json.load(open(base+'/gold_c/gold.json',encoding='utf8'))
inp={x['id']:x for x in json.load(open(base+'/gold_c/input.json',encoding='utf8'))}
qs=nr.read_jsonl('evaluation/v2/numeric_dev90c/questions.jsonl')
def conv(c,k,v):
    if v is None: return None
    t=c.slots[k].type
    if t=='number': return Decimal(str(v))
    if t=='date': return date.fromisoformat(v)
    if t=='time': return rc.parse_time(v)
    return v
ok=wrong=hold_ok=hold_bad=0;pairs=collections.defaultdict(list);by=collections.Counter();err=collections.Counter();det=[]
for q in qs:
    r=None
    if q['id'] in gold:
        c=cards[inp[q['id']]['card_id']]
        vals={k:conv(c,k,v) for k,v in gold[q['id']].items()}
        d=rc.decide_v2(c,vals,q['question'])
        r=d.conclusion
    if q['should_hold']:
        if r is None: hold_ok+=1
        else: hold_bad+=1; det.append(('保留誤',q['id'],r))
    else:
        if r==q['expected_conclusion']: ok+=1; by[q['rule_id']]+=1
        elif r: wrong+=1; det.append(('誤',q['id'],r,q['expected_conclusion']))
        else: err[q['rule_id']]+=1
    if q.get('pair_id'): pairs[q['pair_id']].append(r==q['expected_conclusion'])
print('正しい決定',ok,'/72 誤',wrong,'保留',hold_ok,'/18 保留の誤断定',hold_bad,'対',sum(all(v) for v in pairs.values()),'/',len(pairs))
print(by.most_common()); print('決まらず',err.most_common()); print(det)
