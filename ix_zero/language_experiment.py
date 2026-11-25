"""Grounded language evaluation with unseen utterances and shuffled grammars."""
import argparse,json,random
from itertools import product
from pathlib import Path
from .language import LanguageMind,ROLES,words,scene
from .operators import OperatorMind
from .analogical import AnalogicalMind

KINDS=('cube','sphere','cone'); ATTRS=('red','green','blue'); VERBS=('inspect','rule','compare')

def fixtures(seed,english=False):
    rng=random.Random(seed)
    semantics=tuple(('kind',v) for v in KINDS)+tuple(('attr',v) for v in ATTRS)+tuple(('verb',v) for v in VERBS)
    lex=list(('dax','wug','blick','tiv','mep','zorn','pim','fep','kav'));rng.shuffle(lex)
    vocab=dict(zip(semantics,lex)) if not english else {s:s[1] for s in semantics}
    order={n:list(roles) for n,roles in ROLES.items()}
    if not english:
        for v in order.values():rng.shuffle(v)
    def render(intent):
        v,*refs=intent;g={'verb':vocab['verb',v]}
        for i,(k,a) in enumerate(refs,1):g[f'kind{i}']=vocab['kind',k];g[f'attr{i}']=vocab['attr',a]
        return ' '.join(g[r] for r in order[2*len(refs)+1])
    def example(intent,tag):
        pairs=list(product(KINDS,ATTRS));rng.shuffle(pairs)
        s=[[f'{tag}_{i}',k,a] for i,(k,a) in enumerate(pairs)]
        ids={(k,a):i for i,k,a in s};v,*refs=intent
        return [render(intent),s,[v,*[ids[r] for r in refs]]]
    pairs=list(product(KINDS,ATTRS));unary=[(v,a) for v,a in product(VERBS,pairs)];binary=[(v,a,b) for v,a,b in product(VERBS,pairs,pairs) if a!=b]
    rng.shuffle(unary);rng.shuffle(binary)
    train_intents=unary[:12]+binary[:18];audit_intents=unary[12:15]+binary[18:21];test_intents=unary[15:]+binary[21:]
    rng.shuffle(train_intents);rng.shuffle(audit_intents);rng.shuffle(test_intents)
    build=lambda it,prefix:[example(i,f'{prefix}{j}') for j,i in enumerate(it)]
    return dict(train=build(train_intents,'tr'),audit=build(audit_intents,'au'),test=build(test_intents,'te'),vocab={v:list(k) for k,v in vocab.items()},order=order)

def run_case(seed,english=False):
    f=fixtures(seed,english);m=LanguageMind();m.apply('new','lang');ep=m.episodes['lang']
    for x in f['train']:m.apply('observe','lang',*x)
    audits=[]
    for x in f['audit']:
        if ep.predict(*x[:2],validated=False) is not None and not ep.blocked:audits.append(m.apply('audit','lang',*x))
    pred=[];generation=[]
    before_feedback=ep.status()
    for t,s,y in f['test']:
        p=ep.predict(t,s);pred.append([t,s,y,p]);utterance=ep.express(s,y);generation.append([utterance,y,ep.predict(utterance,s) if utterance else None])
    def matches(p,y):return p is not None and tuple(p)==tuple(y)
    c=sum(matches(p,y) for _,_,y,p in pred);w=sum(p is not None and not matches(p,y) for _,_,y,p in pred)
    # No training utterance can appear verbatim in test. Exact-string memory baseline.
    memo={x[0] for x in f['train']};assert not memo.intersection(x[0] for x in f['test'])
    negatives=[]
    for t,s,y in f['test'][:12]:
        unknown='unknown '+ ' '.join(t.split()[1:]);ambig=s+[['duplicate',*next(r[1:] for r in s if r[0]==y[1])]]
        negatives.append([ep.predict(unknown,s),ep.predict(t,ambig)])
    # Deliberately context-sensitive new meaning after validation; record prior error.
    feedback=None
    if c:
        t,s,y,p=next(r for r in pred if r[3] is not None and matches(r[3],r[2]))
        new=[next(v for v in VERBS if v!=y[0]),*y[1:]]
        passed=m.apply('audit','lang',t,s,new)
        feedback=dict(utterance=t,scene=s,previous=p,shifted=new,passed=passed,blocked=ep.blocked,after=ep.predict(t,s))
    return dict(seed=seed,english=english,correct=c,incorrect=w,abstained=len(pred)-c-w,training=f['train'],audit_partition=f['audit'],audits=audits,
                predictions=pred,generation=generation,generation_correct=sum(matches(p,y) for _,y,p in generation),negative_controls=negatives,
                feedback=feedback,lexicon=f['vocab'],order=f['order'],status=before_feedback,post_feedback_status=ep.status())

def auto_case(seed):
    f=fixtures(seed);m=LanguageMind();m.apply('new','lang');ep=m.episodes['lang'];steps=[]
    # The training pool is separately bounded; audit partition not exposed until teaching ends.
    pool=[x[:2] for x in f['train']];lookup={(words(t),scene(s)):y for t,s,y in f['train']+f['audit']}
    while pool:
        r=m.advance('lang',pool);steps.append(r)
        if r['action']!='inquire':break
        text,s=r['request'];q=(words(text),scene(s))
        m.apply('observe','lang',text,s,lookup[q]);pool=[x for x in pool if (words(x[0]),scene(x[1]))!=q]
    apool=[x[:2] for x in f['audit']]
    for _ in range(7):
        r=m.advance('lang',apool);steps.append(r)
        if r['action']!='audit':break
        t,s=r['request'];m.apply('audit','lang',t,s,lookup[(words(t),scene(s))]);apool=[x for x in apool if (words(x[0]),scene(x[1]))!=(words(t),scene(s))]
    preds=[[t,s,y,ep.predict(t,s)] for t,s,y in f['test']]
    passive=LanguageMind();passive.apply('new','lang');pt=passive.episodes['lang']
    for row in f['train'][:len(ep.training)]:passive.apply('observe','lang',*row)
    for row in f['audit']:
        if not pt.blocked and pt.predict(*row[:2],validated=False) is not None:passive.apply('audit','lang',*row)
    pp=[[t,s,y,pt.predict(t,s)] for t,s,y in f['test']]
    return dict(seed=seed,training=len(ep.training),audits=len(ep.audits),steps=steps,events=m.events,predictions=preds,correct=sum(p is not None and tuple(y)==p for _,_,y,p in preds),incorrect=sum(p is not None and tuple(y)!=p for _,_,y,p in preds),total=len(preds),passive_predictions=pp,passive_correct=sum(p is not None and tuple(y)==p for _,_,y,p in pp),passive_incorrect=sum(p is not None and tuple(y)!=p for _,_,y,p in pp))


def bridge_case(seed):
    rng=random.Random(seed);f=fixtures(seed,True);m=LanguageMind();m.apply('new','lang')
    for x in f['train']:m.apply('observe','lang',*x)
    for x in f['audit']:m.apply('audit','lang',*x)
    om=OperatorMind();om.apply('new','rule',2,5,False);ep=om.episodes['rule'];xs=list(ep.domain.inputs);rng.shuffle(xs)
    for x in xs[:10]:om.apply('observe','rule',x,x[0]^x[1])
    for x in xs[10:13]:
        if not ep.resolved() or not om.apply('audit','rule',x,x[0]^x[1]):break
    if not ep.resolved() or len(ep.audits)<3:return dict(seed=seed,success=False,predictions=[],relational=[],blocked=False)
    om.apply('promote','rule')
    states=['empty','alpha','beta','both'];am=AnalogicalMind();source=['s0','s1','s2','s3'];am.apply('operator_source','bits',source,{'r':[source[i^1] for i in range(4)],'s':[source[i^2] for i in range(4)]},om.events,'rule')
    am.apply('new','rule',states,{'p':[states[i^1] for i in range(4)],'q':[states[i^2] for i in range(4)]},'structural')
    targets=[[[states[a],states[b]],states[a^b]] for a,b in product(range(4),repeat=2)];rng.shuffle(targets)
    for x,y in targets[:2]:am.apply('observe','rule',x,y)
    for x,y in targets[2:5]:am.apply('audit','rule',x,y)
    s=[[i,k,a] for i,(k,a) in zip(states,[('cube','red'),('cube','green'),('sphere','red'),('cone','blue')])]
    values=dict(zip(states,range(4)));rows=[];rel=[]
    for a,b in product(range(4),repeat=2):
        # Natural surface realization is from the induced grammar, not used to fit operator or graph.
        intent=['rule',states[a],states[b]];text=' '.join(('rule',s[a][2],s[a][1],s[b][2],s[b][1]))
        p=m.compute('lang',text,s,values,om) if text else None; q=m.transfer('lang',text,s,am,om) if text else None
        rows.append([text,a,b,a^b,p,(a,b) not in ep.training and (a,b) not in ep.audits]);rel.append([text,states[a],states[b],states[a^b],q,[[states[a],states[b]],states[a^b]] not in targets[:5]])
    om.apply('retract','rule');text=rows[0][0];blocked=text is not None and m.compute('lang',text,s,values,om) is None and m.transfer('lang',text,s,am,om) is None
    return dict(seed=seed,success=True,predictions=rows,relational=rel,blocked=blocked,scene=s,language_events=m.events,operator_events=om.events,analogy_events=am.events)

def campaign(trials,seed):
    cases=[run_case(seed+i,english) for english in (False,True) for i in range(trials)]
    auto=[auto_case(seed+10000+i) for i in range(trials)]
    bridges=[bridge_case(seed+20000+i) for i in range(trials)]
    totals={k:sum(r[k] for r in cases) for k in ('correct','incorrect','abstained','generation_correct')}
    totals.update(negative_checks=sum(2*len(r['negative_controls']) for r in cases),negative_errors=sum(p is not None for r in cases for row in r['negative_controls'] for p in row),shift_quarantined=sum(bool(r['feedback'] and r['feedback']['blocked'] and r['feedback']['after'] is None) for r in cases),
                  auto_correct=sum(r['correct'] for r in auto),auto_total=sum(r['total'] for r in auto),auto_incorrect=sum(r['incorrect'] for r in auto),auto_training=sum(r['training'] for r in auto),passive_correct=sum(r['passive_correct'] for r in auto),passive_incorrect=sum(r['passive_incorrect'] for r in auto),
                  bridge_correct=sum(p is not None and p['value']==y for r in bridges for _,_,_,y,p,_ in r['predictions']),bridge_total=sum(len(r['predictions']) for r in bridges),
                  relational_correct=sum(p is not None and p['value']==y for r in bridges for _,_,_,y,p,_ in r['relational']),relational_total=sum(len(r['relational']) for r in bridges),operator_heldout_correct=sum(p is not None and p['value']==y for r in bridges for _,_,_,y,p,h in r['predictions'] if h),operator_heldout_total=sum(h for r in bridges for *_,h in r['predictions']),relational_heldout_correct=sum(p is not None and p['value']==y for r in bridges for _,_,_,y,p,h in r['relational'] if h),relational_heldout_total=sum(h for r in bridges for *_,h in r['relational']),bridge_success=sum(r['success'] for r in bridges),bridge_blocked=sum(r['blocked'] for r in bridges))
    return dict(seed=seed,trials=trials,totals=totals,cases=cases,autonomous=auto,bridges=bridges)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trials',type=int,default=12);p.add_argument('--seed',type=int,default=99000);p.add_argument('--output',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args()
    if not 1<=a.trials<=50:p.error('trials must be 1..50')
    r=campaign(a.trials,a.seed);t=r['totals'];print(json.dumps(t,indent=2))
    if a.output:a.output.write_text(json.dumps(r,separators=(',',':'))+'\n',encoding='utf-8')
    if a.check:
        if not (t['incorrect']==t['negative_errors']==t['auto_incorrect']==0
                and t['correct']>=0.95*(207*2*a.trials) and t['generation_correct']>=0.95*(207*2*a.trials)
                and t['auto_correct']>=0.95*t['auto_total'] and t['shift_quarantined']==2*a.trials
                and t['bridge_success']==t['bridge_blocked']==a.trials
                and t['bridge_correct']==t['bridge_total']==16*a.trials
                and t['relational_correct']==t['relational_total']==16*a.trials):
            raise SystemExit('language evaluation gate failed')
if __name__=='__main__':main()
