# ws_bridge.py
import asyncio, websockets, json, requests, os

AI_URL = os.environ.get("AI_URL", "http://localhost:5050/ask_ai")
# TEMP: use any local image until Person A sends the real board bytes
DUMMY_IMAGE = os.environ.get("DUMMY_IMAGE", "/Users/vishal/Desktop/Screenshot.png")

async def handler(ws):
    print("WS client connected")
    # send a status heartbeat so the UI shows something
    await ws.send(json.dumps({"type": "status", "planeLocked": False, "fps": 0}))
    try:
        async for raw in ws:
            msg = json.loads(raw)

            if msg.get("type") == "ask_ai":
                q = msg.get("text", "")
                try:
                    with open(DUMMY_IMAGE, "rb") as f:
                        r = requests.post(
                            AI_URL,
                            files={"image": ("board.png", f, "image/png")},
                            data={"question": q},
                            timeout=60,
                        )
                    ans = r.json().get("answer", "(no answer)")
                except Exception as e:
                    ans = f"Error asking AI: {e}"

                await ws.send(json.dumps({"type": "ai_answer", "text": ans}))

            elif msg.get("type") == "action" and msg.get("name") == "save":
                # simulate a save completed message back to UI
                await ws.send(json.dumps({
                    "type": "saved",
                    "url": "https://picsum.photos/800/500?seed=airnote",
                    "boardId": "demo"
                }))

            # color/width/eraser messages can be ignored in this bridge
    finally:
        print("WS client disconnected")

async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("WS bridge on ws://localhost:8765")
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())