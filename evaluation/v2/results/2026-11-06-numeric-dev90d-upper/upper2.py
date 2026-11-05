"""正解の枠値をコードに入れた時の上限。引数: カードのタグ 正解のフォルダ 問の jsonl [v で詳細]"""
import json,os,sys,collections
from decimal import Decimal
from datetime import date
os.environ['RULE_CARDS_V2']='true'
sys.path[:0]=['backend','.']
from evaluation.gpu_local import numeric_replay as nr
from app.services import rule_cards as rc
tag,gd,qsf=sys.argv[1:4]; verbose=len(sys.argv)>4
base=os.path.expanduser('~/.cache/lr-gpu/')
cards=nr.load_cards(base+tag+'/cards')
gold=json.load(open(base+gd+'/gold.json',encoding='utf8'))
inp={x['id']:x for x in json.load(open(base+gd+'/input.json',encoding='utf8'))}
def conv(c,k,v):
    if v is None: return None
    t=c.slots[k].type
    return Decimal(str(v)) if t=='number' else date.fromisoformat(v) if t=='date' else rc.parse_time(v) if t=='time' else v
ok=wrong=ho=hb=0;P=collections.defaultdict(list);by=collections.Counter();un=collections.Counter();det=[]
for q in nr.read_jsonl(qsf):
    r=None;d=None
    if q['id'] in gold:
        c=cards[inp[q['id']]['card_id']]
        d=rc.decide_v2(c,{k:conv(c,k,v) for k,v in gold[q['id']].items()},q['question'])
        r=d.conclusion
    if q['should_hold']:
        ho+=r is None
        if r: hb+=1; det.append(('保留誤',q['id'],r))
    else:
        if r==q['expected_conclusion']: ok+=1; by[q['rule_id'][:5]]+=1
        elif r: wrong+=1; det.append(('誤',q['id'],q['group'],r,q['question'].split('\n')[0][:90]))
        else:
            un[q['rule_id'][:5]]+=1
            if verbose: print(q['id'],q['group'][:12],q['expected_conclusion'],'|',((d.hold_reason if d else '') or '')[:70],'|',q['question'].split('\n')[0][:95])
    if q.get('pair_id'): P[q['pair_id']].append(r==q['expected_conclusion'])
print('上限: 正しい決定',ok,'/72 誤',wrong,'保留',ho,'/18 保留の誤断定',hb,'対',sum(all(v) for v in P.values()),'/',len(P))
print(by.most_common(),'決まらず',un.most_common()); print(det)
