import { useEffect, useMemo, useState } from "react";
import { db } from "../firebase";
import { collection, onSnapshot, orderBy, query } from "firebase/firestore";

type Snap = { id:string; url:string; summary:string };

function tokenize(s:string){
  return (s.toLowerCase().match(/[a-z0-9]+/g) || []).filter(w=>w.length>2 && w.length<20);
}
function vocabFrom(snaps:Snap[], max=200){
  const freq:Record<string,number> = {};
  snaps.forEach(s=> tokenize(s.summary).forEach(w=> freq[w]=(freq[w]||0)+1 ));
  const entries = Object.entries(freq).sort((a,b)=> b[1]-a[1]).slice(0,max);
  return entries.map(e=>e[0]);
}
function vectorize(s:string, vocab:string[]){
  const v = new Array(vocab.length).fill(0);
  const toks = tokenize(s);
  toks.forEach(t=>{
    const idx = vocab.indexOf(t);
    if(idx>=0) v[idx]+=1;
  });
  // L2 normalize
  const norm = Math.sqrt(v.reduce((a,c)=>a+c*c,0)) || 1;
  return v.map(x=>x/norm);
}
function cosine(a:number[], b:number[]){
  let s=0; for(let i=0;i<a.length;i++) s += a[i]*b[i]; return s;
}
function kmeans(vectors:number[][], k:number){
  // init: pick first k
  const centers = vectors.slice(0,k).map(v=>v.slice());
  const assign = new Array(vectors.length).fill(0);
  for(let iter=0; iter<8; iter++){
    // assign
    for(let i=0;i<vectors.length;i++){
      let best=0, bestScore=-Infinity;
      for(let c=0;c<k;c++){ const s = cosine(vectors[i], centers[c]); if(s>bestScore){bestScore=s; best=c;} }
      assign[i]=best;
    }
    // recompute
    for(let c=0;c<k;c++){
      const idx = assign.map((a,i)=>a===c?i:-1).filter(i=>i>=0);
      if(idx.length===0) continue;
      const sum = new Array(vectors[0].length).fill(0);
      idx.forEach(i=> vectors[i].forEach((x,j)=> sum[j]+=x ));
      centers[c] = sum.map(x=> x/idx.length);
    }
  }
  return assign;
}

export default function Dashboard({teamId="demo", folderId="main"}){
  const [snaps, setSnaps] = useState<Snap[]>([]);

  useEffect(()=>{
    const q = query(collection(db,"teams",teamId,"folders",folderId,"snapshots"), orderBy("createdAt","desc"));
    return onSnapshot(q, (ss)=>{
      const arr:Snap[] = [];
      ss.forEach(d=> arr.push({id:d.id, url:d.data().url, summary:d.data().summary||""}));
      setSnaps(arr);
    });
  },[teamId, folderId]);

  const clusters = useMemo(()=>{
    if(snaps.length===0) return [] as {name:string; items:Snap[]}[];
    const vocab = vocabFrom(snaps, 200);
    const vecs = snaps.map(s=> vectorize(s.summary, vocab));
    const k = Math.max(1, Math.min(6, Math.ceil(Math.sqrt(snaps.length/2))));
    const assign = kmeans(vecs, k);
    const buckets: {name:string; items:Snap[]}[] = Array.from({length:k},()=>({name:"Cluster",items:[]}));
    assign.forEach((a,i)=> buckets[a].items.push(snaps[i]));
    // name each cluster by top frequent word in it
    buckets.forEach(b=>{
      const freq:Record<string,number> = {};
      b.items.forEach(s=> tokenize(s.summary).forEach(w=> freq[w]=(freq[w]||0)+1 ));
      const top = Object.entries(freq).sort((a,b)=>b[1]-a[1])[0]?.[0] || "Ideas";
      b.name = top[0].toUpperCase()+top.slice(1)+" Cluster";
    });
    return buckets;
  },[snaps]);

  return (
    <div style={{padding:16, fontFamily:"Inter, system-ui"}}>
      <h2>Team Brain</h2>
      <div style={{display:"grid", gridTemplateColumns:`repeat(${Math.max(1,clusters.length)}, 1fr)`, gap:16}}>
        {clusters.map((c,i)=>(
          <div key={i} style={{border:"1px solid #ddd", borderRadius:12, padding:12}}>
            <h3>{c.name}</h3>
            {c.items.map(it=>(
              <div key={it.id} style={{marginBottom:12}}>
                <img src={it.url} alt="" style={{width:"100%", borderRadius:8, display:"block"}}/>
                <div style={{fontSize:12, color:"#555"}}>{it.summary || "…"}</div>
              </div>
            ))}
            {c.items.length===0 && <div style={{color:"#888"}}>No items yet</div>}
          </div>
        ))}
      </div>
    </div>
  );
}