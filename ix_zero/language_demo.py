"""Interactive example of learned, grounded sentence interpretation."""
import argparse
import json
import random
from .language import LanguageMind
from .language_experiment import fixtures
from .operators import OperatorMind


def build():
    # Synthetic teacher data. The language learner sees complete utterances,
    # scene descriptions and semantic outcomes, not the evaluator's word map.
    f=fixtures(99000,english=True);m=LanguageMind();m.apply('new','demo')
    for row in f['train']:m.apply('observe','demo',*row)
    for row in f['audit']:
        if not m.apply('audit','demo',*row):raise RuntimeError('language validation failed')
    om=OperatorMind();om.apply('new','rule',2,5,False);ep=om.episodes['rule']
    xs=list(ep.domain.inputs);random.Random(119000).shuffle(xs)
    for x in xs[:10]:om.apply('observe','rule',x,x[0]^x[1])
    for x in xs[10:13]:
        if not om.apply('audit','rule',x,x[0]^x[1]):raise RuntimeError('operator validation failed')
    om.apply('promote','rule')
    rows=[['A','cube','red'],['B','cube','green'],['C','sphere','red'],['D','cone','blue']]
    values=dict(A=0,B=1,C=2,D=3)
    return m,om,rows,values


def reply(text,m,om,rows,values):
    intent=m.episodes['demo'].predict(text,rows)
    if intent is None:return {'abstained':True}
    result={'intent':intent}
    if len(intent)==3 and intent[0] in om.promoted:
        result['computed']=m.compute('demo',text,rows,values,om)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check',action='store_true');a=p.parse_args()
    m,om,rows,values=build()
    print('Grounded language example | synthetic scene | no external model')
    print('Scene:',json.dumps(rows));print('Word values:',json.dumps(values))
    print('Learned vocabulary:',json.dumps(m.episodes['demo'].status()['lexicon'],sort_keys=True))
    print('Examples: inspect red cube / compare green cube blue cone / rule green cube red sphere')
    if a.check:
        assert reply('inspect red cube',m,om,rows,values)['intent']==('inspect','A')
        assert reply('rule green cube red sphere',m,om,rows,values)['computed']['value']==3
        assert reply('inspect orange cube',m,om,rows,values)['abstained']
        for text in ('inspect red cube','rule green cube red sphere','inspect orange cube'):
            print(text, json.dumps(reply(text,m,om,rows,values)))
        return
    while True:
        try:
            text=input('You > ').strip()
            if text.lower() in ('exit','quit'):break
            print('IX-Zero > '+json.dumps(reply(text,m,om,rows,values)))
        except (EOFError,KeyboardInterrupt):break
        except ValueError as e:print('IX-Zero > Unsupported construction:',e)

if __name__=='__main__':main()
