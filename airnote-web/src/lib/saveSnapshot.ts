import { db } from "../firebase";
import { collection, addDoc, serverTimestamp } from "firebase/firestore";

export async function recordSnapshot(teamId:string, folderId:string, url:string){
  const col = collection(db, "teams", teamId, "folders", folderId, "snapshots");
  await addDoc(col, {
    teamId, folderId, url,
    summary: "", embedding: [],
    createdAt: serverTimestamp()
  });
}