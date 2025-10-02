/* Local heuristic model for The MindfulAI */
(function(global){
  const LS_KEY_USER_LEARN = 'TheMindfulAI_user_vocab_v1';
  const tokenize = (text)=> (text||'')
      .toLowerCase()
      .replace(/[^a-z0-9\s']/g,' ')
      .split(/\s+/).filter(Boolean);
  const unique = arr => [...new Set(arr)];
  function loadJSON(path){
    return fetch(path,{cache:'no-store'}).then(r=>{ if(!r.ok) throw new Error('Failed to load '+path); return r.json(); });
  }
  class VectorSpace {
    constructor(docs){ this.docs=docs; this.df={}; this.N=docs.length; docs.forEach(d=>unique(d.tokens).forEach(t=>{this.df[t]=(this.df[t]||0)+1;})); }
    idf(term){ const df=this.df[term]||0.5; return Math.log((this.N+1)/df); }
    vectorize(tokens){ const tf={}; tokens.forEach(t=>tf[t]=(tf[t]||0)+1); const vec={}; Object.keys(tf).forEach(t=> vec[t]=(tf[t]/tokens.length)*this.idf(t)); return vec; }
    static cosine(a,b){ let dot=0,na=0,nb=0; for(const k in a){ if(b[k]) dot+=a[k]*b[k]; na+=a[k]*a[k]; } for(const k in b){ nb+=b[k]*b[k]; } if(!na||!nb) return 0; return dot/Math.sqrt(na*nb); }
  }
  class MoodDetector { constructor(lex){ this.lex=lex; this.alias={ anxious:'stress', fear:'stress', sad:'negative', anger:'negative', joy:'positive', happy:'positive'}; }
    score(text){ const tokens=tokenize(text); const scores={positive:0,negative:0,stress:0}; tokens.forEach(t=>{ for(const cat in this.lex){ const w=this.lex[cat][t]; if(w){ if(!scores[cat]) scores[cat]=0; scores[cat]+=w; } } }); let best='default', bestVal=0.4; Object.entries(scores).forEach(([k,v])=>{ if(v>bestVal){ best=k; bestVal=v; } }); return {label:best, raw:scores}; }
  }
  class LocalTherapyModel {
    constructor(){ this.ready=false; this.responses=[]; this.vectorSpace=null; this.moodDetector=null; this.userVocab=this.loadUserVocab(); }
    loadUserVocab(){ try { return JSON.parse(localStorage.getItem(LS_KEY_USER_LEARN)||'{}'); } catch { return {}; } }
    saveUserVocab(){ localStorage.setItem(LS_KEY_USER_LEARN, JSON.stringify(this.userVocab)); }
    async init(){ if(this.ready) return; const [respData, lex] = await Promise.all([ loadJSON('data/responses.json'), loadJSON('data/lexicon.json') ]); this.responses = respData.responses.map((r,i)=>({id:i,...r,tokens:tokenize((r.patterns||[]).join(' '))})); this.vectorSpace = new VectorSpace(this.responses.map(r=>({id:r.id,tokens:r.tokens}))); this.moodDetector = new MoodDetector(lex); this.ready=true; }
    reflect(userText){ if(!this.moodDetector) return ''; const tokens=unique(tokenize(userText)).slice(0,40); const emotional=[]; tokens.forEach(t=>{ for(const cat in this.moodDetector.lex){ if(this.moodDetector.lex[cat][t]){ emotional.push(t); break; } } }); if(!emotional.length) return ''; return ' I notice themes around '+emotional.slice(0,3).join(', ')+(emotional.length>3?'...':'')+'.'; }
    retrieve(userText){ const tokens=tokenize(userText); const q=this.vectorSpace.vectorize(tokens); let best={score:0,item:null}; this.responses.forEach(r=>{ const d=this.vectorSpace.vectorize(r.tokens); const s=VectorSpace.cosine(q,d); if(s>best.score) best={score:s,item:r}; }); return best.score>0.05?best.item:null; }
    bestTemplate(intent){ const r=this.responses.find(r=>r.intent===intent); if(!r||!r.templates||!r.templates.length) return null; return r.templates[Math.floor(Math.random()*r.templates.length)]; }
    learn(userText){ const stop=new Set(['the','a','and','or','to','i','it','is','in','of','that','this','for','on','with','was','are','be']); tokenize(userText).forEach(t=>{ if(stop.has(t)) return; this.userVocab[t]=(this.userVocab[t]||0)+1; }); if(Math.random()<0.2) this.saveUserVocab(); }
    fallback(userText,mood){ const templates=["I'm here with you. It sounds like this matters to you.","Thank you for sharing that. It's okay to feel what you feel.","You're not alone in this. We can unpack it step by step.","Noticing this is already a meaningful step forward."]; let base=templates[Math.random()*templates.length|0]; base+=this.reflect(userText); if(mood==='stress') base+=' Maybe name one specific worry right now.'; else if(mood==='negative') base+=' What made this feel especially heavy today?'; else if(mood==='positive') base+=' What personal strength helped you?'; return base.trim(); }
    async predict(userText,history){ if(!this.ready) await this.init(); const moodRes=this.moodDetector.score(userText); const r=this.retrieve(userText); let response; if(r){ const t=this.bestTemplate(r.intent); response=t||this.fallback(userText,moodRes.label); } else response=this.fallback(userText,moodRes.label); this.learn(userText); return {response_text:response, prediction:moodRes.label, mood_debug:moodRes.raw}; }
  }
  global.MindfulLocalModel = new LocalTherapyModel();
})(window);
