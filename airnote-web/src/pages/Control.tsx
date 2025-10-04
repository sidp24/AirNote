import { useEffect, useRef, useState } from "react";
import { anonLogin } from "../firebase";
import { makeWS } from "../lib/ws";
import type { Incoming, Outgoing } from "../lib/ws";

export default function Control(){
  const ws = useRef<WebSocket|null>(null);
  const [color, setColor] = useState("#00B894");
  const [width, setWidth] = useState(6);
  const [eraser, setEraser] = useState(false);
  const [aiQ, setAiQ] = useState("");
  const [aiA, setAiA] = useState("");
  const [status, setStatus] = useState({ planeLocked:false, fps:0 });

  useEffect(()=>{ (async()=>{
    await anonLogin();
    const s = makeWS((m:Incoming)=>{
      if(m.type==="status") setStatus({planeLocked:m.planeLocked, fps:m.fps});
      if(m.type==="ai_answer") setAiA(m.text);
      if(m.type==="saved") console.log("Saved:", m.url);
    });
    s.onopen = () => {
      send({type:"set_folder", teamId:"demo", folderId:"main"});
      send({type:"tool", tool:"color", value:color});
      send({type:"tool", tool:"width", value:width});
    };
    ws.current = s;
  })(); }, []);

  function send(m:Outgoing){
    if(ws.current && ws.current.readyState===1){
      ws.current.send(JSON.stringify(m));
    }
  }

  return (
    <div style={{fontFamily:"Inter, system-ui", padding:16, maxWidth:520}}>
      <h2>AirNote Control</h2>
      <div style={{marginBottom:12}}>
        <span style={{padding:"4px 8px", border:"1px solid #ddd", borderRadius:8}}>
          Plane: {status.planeLocked ? "Locked ✅" : "Not locked ❌"} · {status.fps} fps
        </span>
      </div>

      <div style={{display:"grid", gap:10}}>
        <label>Color
          <input type="color" value={color}
            onChange={e=>{ setColor(e.target.value); send({type:"tool", tool:"color", value:e.target.value}); }} />
        </label>

        <label>Brush width
          <input type="range" min={2} max={24} value={width}
            onChange={e=>{ const v=Number(e.target.value); setWidth(v); send({type:"tool", tool:"width", value:v}); }} />
        </label>

        <label>
          <input type="checkbox" checked={eraser}
            onChange={e=>{ setEraser(e.target.checked); send({type:"tool", tool:"eraser", value:e.target.checked}); }} />
          Eraser
        </label>

        <button onClick={()=>send({type:"action", name:"save"})}>Save Board</button>

        <h3>Ask AI</h3>
        <textarea rows={3} placeholder="Ask about what’s on the board…" value={aiQ}
          onChange={e=>setAiQ(e.target.value)} />
        <button onClick={()=>send({type:"ask_ai", text: aiQ})}>Ask</button>

        {aiA && (
          <div style={{border:"1px solid #ddd", padding:12, borderRadius:8}}>
            <strong>Answer</strong>
            <ol style={{marginTop:8}}>
              {aiA.split(/\n+/).map((line,i)=> <li key={i}>{line}</li>)}
            </ol>
          </div>
        )}
      </div>
    </div>
  );
}