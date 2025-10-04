export type Outgoing =
  | { type:"tool"; tool:"color"; value:string }
  | { type:"tool"; tool:"width"; value:number }
  | { type:"tool"; tool:"eraser"; value:boolean }
  | { type:"action"; name:"save" }
  | { type:"ask_ai"; text:string }
  | { type:"set_folder"; teamId:string; folderId:string };

export type Incoming =
  | { type:"status"; planeLocked:boolean; fps:number }
  | { type:"saved"; url:string; boardId:string }
  | { type:"ai_answer"; text:string };

export function makeWS(onMessage:(m:Incoming)=>void){
  const sock = new WebSocket("ws://localhost:8765"); // Person A’s Python WS
  sock.onmessage = ev => { try{ onMessage(JSON.parse(ev.data)); }catch{} };
  return sock;
}