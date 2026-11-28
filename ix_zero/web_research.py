"""Read-only public HTTPS research. The learner selects queries and sources.
Network content is untrusted data, never executable instructions. No accounts,
credentials, private network access, or external AI inference are used.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import os
from html.parser import HTMLParser
import hashlib
import http.client
import ipaddress
import re
import socket
import ssl
from urllib.parse import urlencode, urlsplit, urljoin, parse_qs

MAX_BYTES=1_000_000

@dataclass(frozen=True)
class Page:
    url: str
    body: bytes
    content_type: str='text/html'


def validate_url(url):
    if not isinstance(url,str) or len(url)>4096 or any(ord(c)<32 for c in url):raise ValueError('invalid URL')
    p=urlsplit(url)
    if p.scheme!='https' or not p.hostname or p.username is not None or p.password is not None or p.port not in (None,443):raise ValueError('public HTTPS only')
    return p


class PublicHTTPS:
    """Resolve and pin a globally routable address, preserving TLS hostname checks.
    Every redirect is independently resolved and checked. No proxy or cookies.
    """
    def get(self,url):
        for _ in range(5):
            p=validate_url(url)
            addresses=socket.getaddrinfo(p.hostname,443,type=socket.SOCK_STREAM)
            if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):raise ValueError('non-public address')
            ip=addresses[0][4][0]
            conn=http.client.HTTPSConnection(p.hostname,443,timeout=8,context=ssl.create_default_context())
            # Use resolved address rather than a second hostname lookup.
            conn._create_connection=lambda addr,timeout=None,source_address=None:socket.create_connection((ip,443),timeout,source_address)
            try:
                path=p.path or '/';path+=('?' + p.query) if p.query else ''
                conn.request('GET',path,headers={'User-Agent':'IX-Zero-Research/5.0 (read-only)','Accept':'text/html,text/plain,application/json','Accept-Encoding':'identity'})
                r=conn.getresponse()
                if r.status in (301,302,303,307,308):
                    location=r.getheader('Location')
                    if not location:raise OSError('redirect without location')
                    url=urljoin(url,location);continue
                if r.status!=200:raise OSError(f'HTTP {r.status}')
                ct=r.getheader('Content-Type','').split(';')[0].lower()
                if ct not in ('text/html','text/plain','application/json','application/xhtml+xml'):raise ValueError('unsupported content type: '+ct)
                if r.getheader('Content-Encoding','identity').lower()!='identity':raise ValueError('unsupported encoding')
                data=r.read(MAX_BYTES+1)
                if len(data)>MAX_BYTES:raise ValueError('response exceeds size budget')
                return Page(url,data,ct)
            finally:conn.close()
        raise OSError('redirect budget exhausted')


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True);self.text=[];self.links=[];self.skip=0;self.anchor=None;self.parts=[]
    def handle_starttag(self,tag,attrs):
        if tag in ('script','style','noscript'):self.skip+=1
        if self.skip:return
        if tag=='a':self.anchor=dict(attrs).get('href');self.parts=[]
        if tag in ('p','div','li','br','h1','h2','h3','tr'):self.text.append('\n')
    def handle_endtag(self,tag):
        if tag in ('script','style','noscript') and self.skip:self.skip-=1;return
        if self.skip:return
        if tag=='a' and self.anchor is not None:
            self.links.append((self.anchor,' '.join(self.parts)));self.anchor=None
        if tag in ('p','div','li','h1','h2','h3','tr'):self.text.append('\n')
    def handle_data(self,data):
        if not self.skip:
            self.text.append(data)
            if self.anchor is not None:self.parts.append(data)


def parse(page):
    content=page.body.decode('utf-8',errors='replace');p=TextParser()
    if page.content_type in ('text/html','application/xhtml+xml'):p.feed(content)
    else:p.text=[content]
    return p


def tokens(s):return set(re.findall(r'[a-z0-9_]+',s.lower()))


def extract_links(page):
    seen=set();out=[]
    for href,title in parse(page).links:
        try:u=urljoin(page.url,href);p=urlsplit(u)
        except (ValueError,TypeError):continue
        if p.hostname in ('duckduckgo.com','html.duckduckgo.com'):
            u=parse_qs(p.query).get('uddg',[''])[0]
        try:p=validate_url(u)
        except (ValueError,TypeError):continue
        if p.hostname in ('duckduckgo.com','html.duckduckgo.com') or not title.strip() or u in seen:continue
        seen.add(u);out.append((u,title.strip()[:300]))
    return out


def excerpts(page,subject,property_name):
    wanted=tokens(subject+' '+property_name);text=''.join(parse(page).text)
    chunks=[' '.join(s.split()) for s in re.split(r'\n+|(?<=[.!?])\s+',text)]
    return sorted((c for c in chunks if 5<=len(c)<=600 and tokens(subject)&tokens(c) and tokens(property_name)&tokens(c)),key=lambda c:-len(tokens(c)&wanted))[:3]


class LocalDocs:
    """Optional, explicitly designated read-only research corpus. No implicit
    home-directory scanning, execution, network transfer, or symlink traversal.
    """
    def __init__(self,root):
        self.root=Path(root).resolve(strict=True)
        if not self.root.is_dir():raise ValueError('workspace must be a directory')

    def find(self,subject,property_name,limit):
        matches=[];visited=0
        for directory,dirs,files in os.walk(self.root,followlinks=False):
            dirs[:]=sorted(d for d in dirs if not d.startswith('.') and not (Path(directory)/d).is_symlink())
            for name in sorted(files):
                if visited>=128:return matches[:limit],visited
                p=Path(directory)/name
                if p.suffix.lower() not in ('.md','.txt','.rst','.json') or p.is_symlink() or not p.is_file():continue
                visited+=1
                try:
                    actual=p.resolve(strict=True)
                    if not actual.is_relative_to(self.root):continue
                    with actual.open('rb') as f:data=f.read(MAX_BYTES+1)
                    if len(data)>MAX_BYTES:continue
                    label=actual.relative_to(self.root).as_posix();page=Page('local:'+label,data,'text/plain');found=excerpts(page,subject,property_name)
                    if found:matches.append({'url':page.url,'title':label,'sha256':hashlib.sha256(data).hexdigest(),'excerpts':found,'kind':'local_document'})
                except (OSError,ValueError):continue
        return matches[:limit],visited


class Research:
    """Goal-driven retrieval and narrow explicit evidence extraction, not a
    general semantic interpreter. A host supplies a subject and information need,
    not a search string or selected source. Backend defaults to actual HTTPS.
    """
    def __init__(self,transport=None,workspace=None):
        self.transport=transport or PublicHTTPS();self.records=[];self.cache={};self.local=LocalDocs(workspace) if workspace is not None else None

    def investigate(self,subject,property_name,*,max_sources=4,refresh=False):
        if not all(isinstance(v,str) and 1<=len(v.strip())<=120 for v in (subject,property_name)) or type(max_sources) is not int or not 1<=max_sources<=8:raise ValueError('invalid research goal')
        key=(subject.casefold().strip(),property_name.casefold().strip())
        if not refresh and key in self.cache:return {**self.cache[key],'recalled':True}
        if refresh:self.cache.pop(key,None)
        wanted=tokens(subject+' '+property_name);trace=[];sources=[];seen=set();budget=max_sources
        if self.local:
            found,n=self.local.find(subject,property_name,max_sources);trace.append({'action':'local_search','files_examined':n,'matches':len(found)});sources.extend(found)
            # Prefer available documentation before contacting a remote service.
            if found:budget=0
        queries=[subject+' '+property_name,subject+' '+property_name+' technical reference']
        for qi,query in enumerate(queries):
            if budget<=0:break
            url='https://html.duckduckgo.com/html/?'+urlencode({'q':query})
            try:
                search=self.transport.get(url);links=extract_links(search)
                trace.append({'action':'search','query':query,'url':search.url,'links':len(links),'sha256':hashlib.sha256(search.body).hexdigest()})
            except (OSError,ValueError,http.client.HTTPException) as e:
                trace.append({'action':'search','query':query,'error':str(e)});continue
            links.sort(key=lambda r:-len(tokens(r[1])&wanted))
            attempts=0;quota=max(1,(budget+1)//2) if qi==0 and budget>1 else budget
            for link,title in links:
                if budget<=0 or attempts>=quota:break
                if link in seen:continue
                seen.add(link);budget-=1;attempts+=1
                try:
                    page=self.transport.get(link);ranked=excerpts(page,subject,property_name)
                    trace.append({'action':'fetch','url':page.url,'matches':len(ranked)})
                    if ranked:
                        sources.append({'url':page.url,'title':title,'sha256':hashlib.sha256(page.body).hexdigest(),'excerpts':ranked[:3],'kind':'public_web'})
                except (OSError,ValueError,http.client.HTTPException) as e:trace.append({'action':'fetch','url':link,'error':str(e)})
            if len({urlsplit(s['url']).hostname for s in sources})>=2:break
        domains=len({urlsplit(s['url']).hostname for s in sources if s['kind']=='public_web'})
        status='retrieved_evidence' if sources else 'unavailable' if not any('links' in r for r in trace) else 'insufficient_evidence'
        record={'goal':{'subject':subject,'property':property_name},'status':status,'sources':sources,'distinct_hosts':domains,'trace':trace,'truth_verified':False,'recalled':False,'observed_at_utc':datetime.now(timezone.utc).isoformat()}
        self.records.append(record)
        if sources:self.cache[key]=record
        return record
