# mic_list_and_test.py
# Lists audio devices, lets you choose an INPUT device index, then runs the live mic debug
# without modifying voice_io.py (we set sd.default.device to route input to the right mic).

import argparse
import sounddevice as sd
from voice_io import record_push_to_talk_debug

def list_devices():
    print("\n=== Audio Devices (PortAudio / sounddevice) ===")
    devs = sd.query_devices()
    default_in = sd.default.device[0] if isinstance(sd.default.device, (list, tuple)) else sd.default.device
    default_out = sd.default.device[1] if isinstance(sd.default.device, (list, tuple)) else None

    for i, d in enumerate(devs):
        ins = d.get("max_input_channels", 0)
        outs = d.get("max_output_channels", 0)
        fs = d.get("default_samplerate", None)
        mark = []
        if i == default_in:  mark.append("DEFAULT_IN")
        if i == default_out: mark.append("DEFAULT_OUT")
        marks = f" [{' '.join(mark)}]" if mark else ""
        print(f"[{i:02d}] in:{ins} out:{outs}  fs:{fs}  name:{d['name']}{marks}")
    print("==============================================\n")

def main():
    ap = argparse.ArgumentParser(description="List and test input devices")
    ap.add_argument("--device", type=int, default=None, help="Input device index from the list")
    ap.add_argument("--samplerate", type=int, default=16000, help="Samplerate (try 44100 or 48000 if needed)")
    ap.add_argument("--hostapi", type=int, default=None, help="Optional host API index (see sd.query_hostapis())")
    ap.add_argument("--save", type=str, default="out/mic_debug.wav", help="WAV path to save")
    args = ap.parse_args()

    # Optional: pin to a specific Host API, which can fix weird Windows routing issues.
    if args.hostapi is not None:
        sd.default.hostapi = args.hostapi

    # Show host APIs (debug)
    try:
        has = sd.query_hostapis()
        print("Host APIs:")
        for idx, ha in enumerate(has):
            print(f"  [{idx}] {ha.get('name')}  (devices: {ha.get('deviceCount')})")
        print()
    except Exception:
        pass

    list_devices()

    if args.device is None:
        print("Pick a device index from the list and rerun with --device <index>.\n"
              "Tip: look for the entry that says your headset/USB mic and has in:>0.")
        return

    # Route input to that device. Leave output alone.
    # sounddevice default.device uses (in, out) ordering, but it’s fine to pass a single int for input on recent versions.
    sd.default.device = (args.device, None)

    print(f"Using INPUT device index: {args.device}")
    print("Hold 'V' to record; release to stop. A live window should show waveform and levels.\n")

    pcm, sr, stats = record_push_to_talk_debug(
        is_down_func=_is_v_down(),
        samplerate=args.samplerate,
        channels=1,
        debug_window=True,
        save_path=args.save,
        return_stats=True,
    )

    print("\n--- Recording Summary ---")
    print(f"Duration:       {stats['duration_sec']:.3f} s")
    print(f"Avg RMS:        {stats['rms_dbfs_avg']:.1f} dBFS")
    print(f"Peak:           {stats['peak_dbfs']:.1f} dBFS")
    print(f"Clipped ratio:  {100.0*stats['clipped_ratio']:.3f}%")
    print(f"Frames captured:{stats['frames_captured']}")
    if pcm and sr:
        print(f"\nSaved WAV -> {args.save}")
    else:
        print("\nNo audio captured. Check device permissions/levels.")

def _is_v_down():
    # same logic as your app: try Windows GetAsyncKeyState, else fallback to OpenCV key
    try:
        import sys
        if sys.platform.startswith("win"):
            import ctypes
            user32 = ctypes.windll.user32
            VK_V = 0x56
            def is_down():
                return (user32.GetAsyncKeyState(VK_V) & 0x8000) != 0
            return is_down
    except Exception:
        pass

    import cv2
    held = {"v": False}
    def is_down():
        k = cv2.waitKey(1) & 0xFF
        if k in (ord('v'), ord('V')):
            held["v"] = True
        if k == 27:  # ESC
            held["v"] = False
        return held["v"]
    return is_down

if __name__ == "__main__":
    main()
